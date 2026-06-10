import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from indic_bpe_tokenizer.config import TokenizerConfig
from indic_bpe_tokenizer.evaluate_tokenizer import (
    calculate_byte_fallback_metrics,
    calculate_compression_improvement,
    calculate_roundtrip_metrics,
    evaluate_tokenizer,
    get_artifact_name,
    load_baseline_tokenizer,
    load_candidate_tokenizer,
)


config = TokenizerConfig()


@dataclass(frozen=True)
class BenchmarkSlice:
    """Configuration for one benchmark domain."""

    name: str
    file_path: Path


def build_benchmark_slices() -> list[BenchmarkSlice]:
    """Return the benchmark domains evaluated by the suite."""
    return [
        BenchmarkSlice(
            name="native_hindi",
            file_path=Path("data/evaluation/native/hindi_wikipedia_eval.txt"),
        ),
        BenchmarkSlice(
            name="romanized_hindi",
            file_path=Path(
                "data/evaluation/romanized/romanized_hindi_smoke.txt"
            ),
        ),
        BenchmarkSlice(
            name="hinglish_code_mixed",
            file_path=Path("data/evaluation/code_mixed/hinglish_smoke.txt"),
        ),
        BenchmarkSlice(
            name="mixed_script",
            file_path=Path(
                "data/evaluation/mixed_script/mixed_script_smoke.txt"
            ),
        ),
        BenchmarkSlice(
            name="informal_chat",
            file_path=Path("data/evaluation/informal/informal_chat_smoke.txt"),
        ),
    ]


def load_benchmark_texts(
    slice_config: BenchmarkSlice,
) -> list[str]:
    """Load non-empty UTF-8 text samples for one benchmark slice."""
    if not slice_config.file_path.exists():
        raise FileNotFoundError(
            f"Benchmark file does not exist: {slice_config.file_path}"
        )

    texts: list[str] = []

    with slice_config.file_path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()

            if text:
                texts.append(text)

    return texts


def evaluate_benchmark_slice(
    candidate_tokenizer,
    baseline_tokenizer,
    slice_config: BenchmarkSlice,
) -> dict[str, Any]:
    """Evaluate candidate and baseline tokenizers for one benchmark slice."""
    if candidate_tokenizer is None:
        raise ValueError("Candidate tokenizer is required.")

    if baseline_tokenizer is None:
        raise ValueError("Baseline tokenizer is required.")

    texts = load_benchmark_texts(slice_config=slice_config)

    if not texts:
        raise ValueError(
            f"Benchmark slice '{slice_config.name}' does not contain any text."
        )

    candidate_eval_metrics = evaluate_tokenizer(
        tokenizer=candidate_tokenizer,
        texts=texts,
        tokenizer_name="candidate_tokenizer",
        tokenizer_type="candidate",
        unk_token="<unk>",
    )

    baseline_eval_metrics = evaluate_tokenizer(
        tokenizer=baseline_tokenizer,
        texts=texts,
        tokenizer_name="baseline_tokenizer",
        tokenizer_type="baseline",
        unk_token=baseline_tokenizer.unk_token or "<unk>",
    )

    byte_fallback_metrics = calculate_byte_fallback_metrics(
        tokenizer=candidate_tokenizer,
        texts=texts,
    )

    roundtrip_metrics = calculate_roundtrip_metrics(
        tokenizer=candidate_tokenizer,
        texts=texts,
    )

    compression_improvement_percent = calculate_compression_improvement(
        candidate_token_count=candidate_eval_metrics["total_tokens"],
        baseline_token_count=baseline_eval_metrics["total_tokens"],
    )

    return {
        "slice_name": slice_config.name,
        "candidate": candidate_eval_metrics,
        "baseline": baseline_eval_metrics,
        "compression_improvement_percent": compression_improvement_percent,
        "byte_fallback_metrics": byte_fallback_metrics,
        "roundtrip_metrics": roundtrip_metrics,
    }


def evaluate_benchmark_suite(
    candidate_tokenizer,
    baseline_tokenizer,
    benchmark_slices: list[BenchmarkSlice],
) -> dict[str, Any]:
    """Evaluate all requested benchmark slices and return grouped results."""
    if candidate_tokenizer is None:
        raise ValueError("Candidate tokenizer is required.")

    if baseline_tokenizer is None:
        raise ValueError("Baseline tokenizer is required.")

    if not benchmark_slices:
        raise ValueError("At least one benchmark slice is required.")

    slice_results = [
        evaluate_benchmark_slice(
            candidate_tokenizer=candidate_tokenizer,
            baseline_tokenizer=baseline_tokenizer,
            slice_config=slice_config,
        )
        for slice_config in benchmark_slices
    ]

    return {
        "slice_count": len(slice_results),
        "slices": slice_results,
    }


def save_benchmark_report(
    results: dict[str, Any],
    artifact_name: str,
    output_dir: Path,
) -> None:
    """Save machine-readable JSON and human-readable Markdown suite reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_artifact_name = artifact_name.replace("/", "_")
    json_path = output_dir / f"{safe_artifact_name}_suite_{timestamp}.json"
    markdown_path = output_dir / f"{safe_artifact_name}_suite_{timestamp}.md"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    table_rows = []

    for slice_result in results["slices"]:
        candidate = slice_result["candidate"]
        baseline = slice_result["baseline"]
        fallback = slice_result["byte_fallback_metrics"]
        roundtrip = slice_result["roundtrip_metrics"]

        table_rows.append(
            "| "
            f"{slice_result['slice_name']} | "
            f"{candidate['sentence_count']} | "
            f"{candidate['tokens_per_word']:.4f} | "
            f"{baseline['tokens_per_word']:.4f} | "
            f"{slice_result['compression_improvement_percent']:.2f}% | "
            f"{candidate['unknown_token_rate']:.4%} | "
            f"{fallback['sentence_fallback_rate']:.4%} | "
            f"{roundtrip['roundtrip_failure_count']} |"
        )

    markdown_content = f"""# Multi-Domain Tokenizer Benchmark Suite

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Candidate artifact:** `{artifact_name}`

## Domain Summary

| Domain | Sentences | Candidate Tokens / Word | XLM-R Tokens / Word | Compression Gain | Unknown Rate | Fallback Sentence Rate | Roundtrip Failures |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## Detailed Results

```json
{json.dumps(results, ensure_ascii=False, indent=2)}
```
"""

    markdown_path.write_text(markdown_content, encoding="utf-8")

    print(f"Saved JSON report: {json_path}")
    print(f"Saved Markdown report: {markdown_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the multi-domain benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Run multi-domain tokenizer benchmarks."
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/benchmark_suite"),
        help="Directory where grouped benchmark reports will be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the multi-domain benchmark suite and save grouped reports."""
    args = parse_args()
    artifact_path = args.candidate_artifact or config.candidate_tokenizer_path
    artifact_name = get_artifact_name(candidate_artifact_path=artifact_path)

    print(f"Loading candidate tokenizer: {artifact_path}")

    candidate_tokenizer = load_candidate_tokenizer(
        tokenizer_path=artifact_path
    )
    baseline_tokenizer = load_baseline_tokenizer(
        model_name=config.baseline_model_name
    )
    benchmark_slices = build_benchmark_slices()

    results = evaluate_benchmark_suite(
        candidate_tokenizer=candidate_tokenizer,
        baseline_tokenizer=baseline_tokenizer,
        benchmark_slices=benchmark_slices,
    )

    save_benchmark_report(
        results=results,
        artifact_name=artifact_name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()