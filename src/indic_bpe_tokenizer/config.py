from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenizerConfig:
    raw_data_dir: Path = Path("data/raw/hindi")

    processed_data_dir: Path = Path("data/processed/hindi")

    artifact_dir: Path = Path(
        "artifacts/hindi_bpe_32k_wiki_filtered_50k_initial_alphabet"
    )

    use_devanagari_initial_alphabet: bool = True

    candidate_tokenizer_path: Path = Path(
        "artifacts/hindi_bpe_32k_wiki_filtered_50k_initial_alphabet/tokenizer.json"
    )

    evaluation_file: Path = Path(
        "data/evaluation/hindi_wikipedia_eval.txt"
    )

    reports_dir: Path = Path("reports")

    report_path: Path = Path(
        "reports/tokenizer_comparison.json"
    )

    baseline_model_name: str = "xlm-roberta-base"
    baseline_display_name: str = "XLM-RoBERTa Base"
    candidate_display_name: str = "Hindi BPE 32k"

    metrics_path: Path = Path(
        "reports/metrics.json"
    )

    vocab_size: int = 32000
    min_frequency: int = 2

    unicode_normalization: str = "NFC"
    min_line_length: int = 2
    max_line_length: int = 5_000

    special_tokens: tuple[str, ...] = (
        "<unk>",
        "<pad>",
        "<s>",
        "</s>",
    )
