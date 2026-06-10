import argparse
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from indic_bpe_tokenizer.config import TokenizerConfig
from indic_bpe_tokenizer.normalize_text import (
    normalize_indic_text,
    should_keep_line,
)
from indic_bpe_tokenizer.tokenizer_adapters import (
    get_baseline_token_ids,
    get_baseline_tokens,
    get_candidate_token_ids,
    get_candidate_tokens,
)


config = TokenizerConfig()

BYTE_FALLBACK_TOKEN_PATTERN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def count_words(text: str) -> int:
    """Count whitespace-separated words in one text sample."""
    return len(text.split())


def get_token_ids(
    tokenizer: Any,
    text: str,
    tokenizer_type: str,
) -> list[int]:
    """Return token IDs for a candidate or baseline tokenizer."""
    if tokenizer_type == "candidate":
        return get_candidate_token_ids(tokenizer=tokenizer, text=text)

    if tokenizer_type == "baseline":
        return get_baseline_token_ids(tokenizer=tokenizer, text=text)

    raise ValueError(f"Tokenizer type '{tokenizer_type}' is not valid.")


def get_tokens(
    tokenizer: Any,
    text: str,
    tokenizer_type: str,
) -> list[str]:
    """Return token strings for a candidate or baseline tokenizer."""
    if tokenizer_type == "candidate":
        return get_candidate_tokens(tokenizer=tokenizer, text=text)

    if tokenizer_type == "baseline":
        return get_baseline_tokens(tokenizer=tokenizer, text=text)

    raise ValueError(f"Tokenizer type '{tokenizer_type}' is not valid.")


def count_tokens(
    tokenizer: Any,
    text: str,
    tokenizer_type: str,
) -> int:
    """Count the tokens generated for one text sample."""
    return len(
        get_token_ids(
            tokenizer=tokenizer,
            text=text,
            tokenizer_type=tokenizer_type,
        )
    )


def calculate_total_tokens(
    tokenizer: Any,
    texts: list[str],
    tokenizer_type: str,
) -> int:
    """Calculate total tokens across the evaluation corpus."""
    return sum(
        count_tokens(
            tokenizer=tokenizer,
            text=text,
            tokenizer_type=tokenizer_type,
        )
        for text in texts
    )


def calculate_tokens_per_word(
    tokenizer: Any,
    texts: list[str],
    tokenizer_type: str,
) -> float:
    """Calculate average tokens per whitespace-separated word."""
    total_tokens = calculate_total_tokens(
        tokenizer=tokenizer,
        texts=texts,
        tokenizer_type=tokenizer_type,
    )
    total_words = sum(count_words(text) for text in texts)

    if total_words == 0:
        return 0.0

    return total_tokens / total_words


def calculate_tokens_per_character(
    tokenizer: Any,
    texts: list[str],
    tokenizer_type: str,
) -> float:
    """Calculate average tokens per Unicode character."""
    total_tokens = calculate_total_tokens(
        tokenizer=tokenizer,
        texts=texts,
        tokenizer_type=tokenizer_type,
    )
    total_characters = sum(len(text) for text in texts)

    if total_characters == 0:
        return 0.0

    return total_tokens / total_characters


def extract_unknown_tokens_count(
    tokenizer: Any,
    texts: list[str],
    tokenizer_type: str,
    unk_token: str,
) -> dict[str, int]:
    """
    Return words that produce unknown tokens and the number of times
    each word appears with at least one unknown token.

    This is a diagnostic helper. Aggregate metrics still use full-text
    encoding so they reflect realistic tokenizer behavior.
    """
    if not texts:
        raise ValueError("Evaluation text list is empty.")

    unknown_word_counts: dict[str, int] = {}

    for text in texts:
        for word in text.split():
            tokens = get_tokens(
                tokenizer=tokenizer,
                text=word,
                tokenizer_type=tokenizer_type,
            )

            if unk_token in tokens:
                unknown_word_counts[word] = (
                    unknown_word_counts.get(word, 0) + 1
                )

    return unknown_word_counts


def calculate_unknown_token_rate(
    tokenizer: Any,
    texts: list[str],
    tokenizer_type: str,
    unk_token: str,
) -> float:
    """Calculate the proportion of generated tokens that are unknown."""
    total_tokens = 0
    unknown_tokens = 0

    for text in texts:
        tokens = get_tokens(
            tokenizer=tokenizer,
            text=text,
            tokenizer_type=tokenizer_type,
        )
        total_tokens += len(tokens)
        unknown_tokens += tokens.count(unk_token)

    if total_tokens == 0:
        return 0.0

    return unknown_tokens / total_tokens


def is_byte_fallback_token(token: str) -> bool:
    """Return whether a token is a UTF-8 byte-fallback token."""
    return BYTE_FALLBACK_TOKEN_PATTERN.match(token) is not None


def extract_fallback_byte_chunks(tokens: list[str]) -> list[bytes]:
    """
    Group consecutive <0xNN> tokens into byte sequences.

    One Unicode character can require multiple UTF-8 byte tokens.
    """
    byte_chunks: list[bytes] = []
    current_chunk: list[int] = []

    for token in tokens:
        match = BYTE_FALLBACK_TOKEN_PATTERN.match(token)

        if match:
            current_chunk.append(int(match.group(1), 16))
            continue

        if current_chunk:
            byte_chunks.append(bytes(current_chunk))
            current_chunk = []

    if current_chunk:
        byte_chunks.append(bytes(current_chunk))

    return byte_chunks


def get_script_name(char: str) -> str:
    """
    Return an approximate Unicode script label from the character name.
    """
    unicode_name = unicodedata.name(char, "")

    if not unicode_name:
        return "UNKNOWN"

    return unicode_name.split(" ", maxsplit=1)[0]


def format_counter(
    counter: Counter[str],
    value_key: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Convert a Counter into a JSON-friendly ranked list."""
    return [
        {
            value_key: value,
            "count": count,
        }
        for value, count in counter.most_common(limit)
    ]


def calculate_byte_fallback_metrics(
    tokenizer: Tokenizer,
    texts: list[str],
) -> dict[str, Any]:
    """Measure how often true byte fallback is used by a candidate tokenizer."""
    total_tokens = 0
    total_fallback_tokens = 0
    sentences_using_fallback = 0

    fallback_character_counts: Counter[str] = Counter()
    fallback_code_point_counts: Counter[str] = Counter()
    fallback_script_counts: Counter[str] = Counter()

    for text in texts:
        tokens = get_candidate_tokens(
            tokenizer=tokenizer,
            text=text,
        )
        total_tokens += len(tokens)

        fallback_tokens = [
            token for token in tokens if is_byte_fallback_token(token)
        ]

        if fallback_tokens:
            sentences_using_fallback += 1
            total_fallback_tokens += len(fallback_tokens)

        for byte_chunk in extract_fallback_byte_chunks(tokens):
            decoded_text = byte_chunk.decode("utf-8", errors="replace")

            for char in decoded_text:
                fallback_character_counts[char] += 1
                fallback_code_point_counts[f"U+{ord(char):04X}"] += 1
                fallback_script_counts[get_script_name(char)] += 1

    sentence_count = len(texts)

    return {
        "byte_fallback_rate": (
            total_fallback_tokens / total_tokens
            if total_tokens > 0
            else 0.0
        ),
        "sentences_using_fallback": sentences_using_fallback,
        "sentence_fallback_rate": (
            sentences_using_fallback / sentence_count
            if sentence_count > 0
            else 0.0
        ),
        "total_fallback_tokens": total_fallback_tokens,
        "average_fallback_tokens_per_affected_sentence": (
            total_fallback_tokens / sentences_using_fallback
            if sentences_using_fallback > 0
            else 0.0
        ),
        "top_fallback_characters": format_counter(
            fallback_character_counts,
            value_key="character",
        ),
        "top_fallback_code_points": format_counter(
            fallback_code_point_counts,
            value_key="code_point",
        ),
        "top_fallback_scripts": format_counter(
            fallback_script_counts,
            value_key="script",
        ),
    }


def calculate_roundtrip_metrics(
    tokenizer: Tokenizer,
    texts: list[str],
    max_failure_examples: int = 10,
) -> dict[str, Any]:
    """Check whether candidate encode/decode is strictly reversible."""
    failure_examples: list[dict[str, str]] = []
    roundtrip_failure_count = 0

    for text in texts:
        encoded = tokenizer.encode(text)
        decoded_text = tokenizer.decode(
            encoded.ids,
            skip_special_tokens=False,
        )

        if decoded_text != text:
            roundtrip_failure_count += 1

            if len(failure_examples) < max_failure_examples:
                failure_examples.append(
                    {
                        "original": text,
                        "decoded": decoded_text,
                    }
                )

    sentence_count = len(texts)

    return {
        "roundtrip_failure_count": roundtrip_failure_count,
        "roundtrip_failure_rate": (
            roundtrip_failure_count / sentence_count
            if sentence_count > 0
            else 0.0
        ),
        "roundtrip_failure_examples": failure_examples,
    }


def load_candidate_tokenizer(tokenizer_path: Path) -> Tokenizer:
    """Load the trained custom tokenizer artifact."""
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"The tokenizer path {tokenizer_path} does not exist."
        )

    return Tokenizer.from_file(str(tokenizer_path))


def load_baseline_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load a pretrained Hugging Face tokenizer for comparison."""
    return AutoTokenizer.from_pretrained(model_name)


def load_evaluation_texts(eval_file: Path) -> list[str]:
    """Load, normalize, and filter held-out evaluation text."""
    if not eval_file.exists():
        raise FileNotFoundError(
            f"Evaluation file path {eval_file} does not exist."
        )

    texts: list[str] = []

    with eval_file.open("r", encoding="utf-8") as file:
        for line in file:
            cleaned_line = normalize_indic_text(line)

            if should_keep_line(cleaned_line):
                texts.append(cleaned_line)

    return texts


def calculate_compression_improvement(
    candidate_token_count: int,
    baseline_token_count: int,
) -> float:
    """Calculate candidate token-count reduction relative to baseline."""
    if baseline_token_count == 0:
        return 0.0

    return (
        (baseline_token_count - candidate_token_count)
        / baseline_token_count
        * 100
    )


def compare_sentence_tokenization(
    text: str,
    candidate_tokenizer: Tokenizer,
    baseline_tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    """Compare tokenization behavior for one evaluation sentence."""
    candidate_tokens = get_candidate_tokens(
        tokenizer=candidate_tokenizer,
        text=text,
    )
    baseline_tokens = get_baseline_tokens(
        tokenizer=baseline_tokenizer,
        text=text,
    )

    return {
        "text": text,
        "candidate_tokens": candidate_tokens,
        "candidate_token_count": len(candidate_tokens),
        "baseline_tokens": baseline_tokens,
        "baseline_token_count": len(baseline_tokens),
        "compression_improvement_percent": (
            calculate_compression_improvement(
                candidate_token_count=len(candidate_tokens),
                baseline_token_count=len(baseline_tokens),
            )
        ),
    }


def evaluate_tokenizer(
    tokenizer: Any,
    texts: list[str],
    tokenizer_name: str,
    tokenizer_type: str,
    unk_token: str,
) -> dict[str, Any]:
    """Calculate aggregate evaluation metrics for one tokenizer."""
    return {
        "tokenizer_name": tokenizer_name,
        "sentence_count": len(texts),
        "total_tokens": calculate_total_tokens(
            tokenizer=tokenizer,
            texts=texts,
            tokenizer_type=tokenizer_type,
        ),
        "tokens_per_word": calculate_tokens_per_word(
            tokenizer=tokenizer,
            texts=texts,
            tokenizer_type=tokenizer_type,
        ),
        "tokens_per_character": calculate_tokens_per_character(
            tokenizer=tokenizer,
            texts=texts,
            tokenizer_type=tokenizer_type,
        ),
        "unknown_token_rate": calculate_unknown_token_rate(
            tokenizer=tokenizer,
            texts=texts,
            tokenizer_type=tokenizer_type,
            unk_token=unk_token,
        ),
        "unknown_token_counts": extract_unknown_tokens_count(
            tokenizer=tokenizer,
            texts=texts,
            tokenizer_type=tokenizer_type,
            unk_token=unk_token,
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse evaluation inputs for a specific tokenizer artifact."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained tokenizer artifact against the baseline."
    )
    parser.add_argument(
        "--candidate-artifact",
        type=Path,
        default=None,
        help=(
            "Path to the candidate tokenizer.json file. "
            "Defaults to config.candidate_tokenizer_path."
        ),
    )
    return parser.parse_args()


def get_artifact_name(candidate_artifact_path: Path) -> str:
    """Return a stable experiment name from the tokenizer artifact directory."""
    return candidate_artifact_path.parent.name


def compare_tokenizers(
    candidate_tokenizer: Tokenizer,
    baseline_tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    candidate_display_name: str,
) -> dict[str, Any]:
    """Evaluate candidate and baseline tokenizers on identical text."""
    candidate_metrics = evaluate_tokenizer(
        tokenizer=candidate_tokenizer,
        texts=texts,
        tokenizer_name=candidate_display_name,
        tokenizer_type="candidate",
        unk_token="<unk>",
    )
    candidate_metrics["byte_fallback_metrics"] = (
        calculate_byte_fallback_metrics(
            tokenizer=candidate_tokenizer,
            texts=texts,
        )
    )
    candidate_metrics["roundtrip_metrics"] = calculate_roundtrip_metrics(
        tokenizer=candidate_tokenizer,
        texts=texts,
    )

    baseline_unk_token = baseline_tokenizer.unk_token or "<unk>"
    baseline_metrics = evaluate_tokenizer(
        tokenizer=baseline_tokenizer,
        texts=texts,
        tokenizer_name=config.baseline_display_name,
        tokenizer_type="baseline",
        unk_token=baseline_unk_token,
    )

    return {
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "compression_improvement_percent": (
            calculate_compression_improvement(
                candidate_token_count=candidate_metrics["total_tokens"],
                baseline_token_count=baseline_metrics["total_tokens"],
            )
        ),
        "sentence_comparisons": [
            compare_sentence_tokenization(
                text=text,
                candidate_tokenizer=candidate_tokenizer,
                baseline_tokenizer=baseline_tokenizer,
            )
            for text in texts
        ],
    }


def save_report(
    results: dict[str, Any],
    output_dir: Path,
    artifact_name: str,
) -> None:
    """Save machine-readable JSON and human-readable Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_artifact_name = artifact_name.replace("/", "_")
    json_path = output_dir / (
        f"{safe_artifact_name}_comparison_{timestamp}.json"
    )
    markdown_path = output_dir / (
        f"{safe_artifact_name}_comparison_{timestamp}.md"
    )

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    candidate = results["candidate"]
    baseline = results["baseline"]
    fallback = candidate["byte_fallback_metrics"]
    roundtrip = candidate["roundtrip_metrics"]
    fallback_is_rare = fallback["sentence_fallback_rate"] < 0.01

    markdown_content = f"""# Tokenizer Comparison Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Metrics

| Tokenizer | Sentences | Total Tokens | Tokens / Word | Tokens / Character | Unknown Rate |
|---|---:|---:|---:|---:|---:|
| {candidate["tokenizer_name"]} | {candidate["sentence_count"]} | {candidate["total_tokens"]} | {candidate["tokens_per_word"]:.4f} | {candidate["tokens_per_character"]:.4f} | {candidate["unknown_token_rate"]:.4%} |
| {baseline["tokenizer_name"]} | {baseline["sentence_count"]} | {baseline["total_tokens"]} | {baseline["tokens_per_word"]:.4f} | {baseline["tokens_per_character"]:.4f} | {baseline["unknown_token_rate"]:.4%} |

## Compression Improvement

Candidate token-count reduction relative to baseline: **{results["compression_improvement_percent"]:.2f}%**

## Candidate Unknown-Token Diagnostics

```json
{json.dumps(candidate["unknown_token_counts"], ensure_ascii=False, indent=2)}
```

## Baseline Unknown-Token Diagnostics

```json
{json.dumps(baseline["unknown_token_counts"], ensure_ascii=False, indent=2)}
```

## Milestone 2: Byte-Fallback Diagnostics

| Metric | Value |
|---|---:|
| Byte-fallback token rate | {fallback["byte_fallback_rate"]:.4%} |
| Sentences using fallback | {fallback["sentences_using_fallback"]} |
| Percentage of sentences using fallback | {fallback["sentence_fallback_rate"]:.4%} |
| Total fallback tokens | {fallback["total_fallback_tokens"]} |
| Average fallback tokens per affected sentence | {fallback["average_fallback_tokens_per_affected_sentence"]:.4f} |

### Top Fallback Characters

```json
{json.dumps(fallback["top_fallback_characters"], ensure_ascii=False, indent=2)}
```

### Top Fallback Code Points

```json
{json.dumps(fallback["top_fallback_code_points"], ensure_ascii=False, indent=2)}
```

### Top Fallback Scripts

```json
{json.dumps(fallback["top_fallback_scripts"], ensure_ascii=False, indent=2)}
```

## Milestone 2: Reversibility

| Metric | Value |
|---|---:|
| Round-trip failures | {roundtrip["roundtrip_failure_count"]} |
| Round-trip failure rate | {roundtrip["roundtrip_failure_rate"]:.4%} |

### Round-Trip Failure Examples

```json
{json.dumps(roundtrip["roundtrip_failure_examples"], ensure_ascii=False, indent=2)}
```

## Milestone 2: Findings

- Unknown-token criterion: **{"PASS" if candidate["unknown_token_rate"] <= 0.000001 else "REVIEW"}**. Candidate unknown-token rate is {candidate["unknown_token_rate"]:.4%}.
- Reversibility criterion: **{"PASS" if roundtrip["roundtrip_failure_count"] == 0 else "FAIL"}**. Strict round-trip failures: {roundtrip["roundtrip_failure_count"]}.
- Clean-Hindi fallback rarity: **{"PASS" if fallback_is_rare else "REVIEW"}**. Sentences using fallback: {fallback["sentence_fallback_rate"]:.4%}.
- Compression trade-off: candidate token-count reduction relative to XLM-RoBERTa is **{results["compression_improvement_percent"]:.2f}%**.
- Interpretation: prefer the whitespace-BPE byte-fallback mode when it reaches zero unknown tokens, preserves reversibility, and keeps fallback usage rare without materially reducing compression.
"""

    markdown_path.write_text(markdown_content, encoding="utf-8")

    print(f"Saved JSON report: {json_path}")
    print(f"Saved Markdown report: {markdown_path}")


def main() -> None:
    """Load tokenizers, run comparison, and save reports."""
    args = parse_args()
    candidate_artifact_path = (
        args.candidate_artifact or config.candidate_tokenizer_path
    )
    artifact_name = get_artifact_name(candidate_artifact_path)

    print(f"Evaluating candidate artifact: {candidate_artifact_path}")

    candidate_tokenizer = load_candidate_tokenizer(
        tokenizer_path=candidate_artifact_path
    )
    baseline_tokenizer = load_baseline_tokenizer(
        model_name=config.baseline_model_name
    )
    texts = load_evaluation_texts(eval_file=config.evaluation_file)

    if not texts:
        raise ValueError("Evaluation text list is empty.")

    results = compare_tokenizers(
        candidate_tokenizer=candidate_tokenizer,
        baseline_tokenizer=baseline_tokenizer,
        texts=texts,
        candidate_display_name=artifact_name,
    )

    save_report(
        results=results,
        output_dir=config.reports_dir,
        artifact_name=artifact_name,
    )


if __name__ == "__main__":
    main()