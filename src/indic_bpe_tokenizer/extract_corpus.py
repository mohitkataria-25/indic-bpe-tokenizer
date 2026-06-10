import argparse
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .corpus_quality import get_corpus_rejection_reason

from .normalize_text import normalize_indic_text, should_keep_line


DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.hi"
DATASET_SPLIT = "train"



@dataclass(frozen=True)
class ExtractionStats:
    dataset_name: str
    dataset_config: str
    articles_processed: int
    paragraphs_seen: int
    unique_paragraphs_written: int
    duplicate_paragraphs_skipped: int
    filtered_paragraphs_skipped: int
    foreign_script_heavy_paragraphs_skipped: int
    corrupted_paragraphs_skipped: int
    metadata_like_paragraphs_skipped: int
    train_paragraphs_written: int
    evaluation_paragraphs_written: int



def load_hindi_wikipedia_stream(
    dataset_name: str = DATASET_NAME,
    dataset_config: str = DATASET_CONFIG,
    split: str = DATASET_SPLIT,
) -> Iterable[dict[str, Any]]:
    """
    Stream cleaned Hindi Wikipedia articles from Hugging Face.

    Streaming avoids downloading the complete corpus before processing.
    """
    return load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        streaming=True,
    )


def iter_article_paragraphs(
    articles: Iterable[dict[str, Any]],
    max_articles: int | None = None,
) -> Iterator[str]:
    """
    Yield paragraphs from streamed Wikipedia articles.

    Each dataset row contains the cleaned text for one article. Blank
    lines are ignored. Set max_articles for an initial bounded run.
    """
    for article_index, article in enumerate(articles):
        if max_articles is not None and article_index >= max_articles:
            break

        article_text = article.get("text", "")

        if not isinstance(article_text, str):
            continue

        for paragraph in article_text.splitlines():
            if paragraph.strip():
                yield paragraph


def create_text_fingerprint(text: str) -> str:
    """Create a stable fingerprint for paragraph deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def should_write_to_evaluation(
    fingerprint: str,
    eval_ratio: float,
) -> bool:
    """
    Assign a paragraph deterministically to the held-out evaluation split.

    The same paragraph always receives the same split across reruns.
    """
    if not 0.0 <= eval_ratio <= 1.0:
        raise ValueError("eval_ratio must be between 0.0 and 1.0.")

    split_bucket = int(fingerprint[:8], 16) / 0xFFFFFFFF
    return split_bucket < eval_ratio


def extract_train_eval_corpus(
    articles: Iterable[dict[str, Any]],
    train_output_path: Path,
    eval_output_path: Path,
    stats_output_path: Path,
    max_articles: int | None = None,
    eval_ratio: float = 0.10,
    min_line_length: int = 40,
    max_line_length: int = 5_000,
) -> ExtractionStats:
    """
    Normalize, filter, deduplicate, split, and save Hindi paragraphs.
    """
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_output_path.parent.mkdir(parents=True, exist_ok=True)

    fingerprints: set[str] = set()
    articles_processed = 0
    paragraphs_seen = 0
    duplicates_skipped = 0
    filtered_skipped = 0
    train_written = 0
    evaluation_written = 0
    foreign_script_heavy_skipped = 0
    corrupted_skipped = 0
    metadata_like_skipped = 0

    with (
        train_output_path.open("w", encoding="utf-8") as train_file,
        eval_output_path.open("w", encoding="utf-8") as eval_file,
    ):
        for article_index, article in enumerate(articles):
            if max_articles is not None and article_index >= max_articles:
                break

            articles_processed += 1
            article_text = article.get("text", "")

            if not isinstance(article_text, str):
                continue

            for raw_paragraph in article_text.splitlines():
                if not raw_paragraph.strip():
                    continue

                paragraphs_seen += 1
                paragraph = normalize_indic_text(raw_paragraph)

                if not should_keep_line(
                    paragraph,
                    min_length=min_line_length,
                    max_length=max_line_length,
                ):
                    filtered_skipped += 1
                    continue

                rejection_reason = get_corpus_rejection_reason(paragraph)

                if rejection_reason == "foreign_script_heavy":
                    foreign_script_heavy_skipped += 1
                    continue

                if rejection_reason == "corrupted":
                    corrupted_skipped += 1
                    continue

                if rejection_reason == "metadata_like":
                    metadata_like_skipped += 1
                    continue

                fingerprint = create_text_fingerprint(paragraph)

                if fingerprint in fingerprints:
                    duplicates_skipped += 1
                    continue

                fingerprints.add(fingerprint)

                if should_write_to_evaluation(
                    fingerprint=fingerprint,
                    eval_ratio=eval_ratio,
                ):
                    eval_file.write(f"{paragraph}\n")
                    evaluation_written += 1
                else:
                    train_file.write(f"{paragraph}\n")
                    train_written += 1

    stats = ExtractionStats(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        articles_processed=articles_processed,
        paragraphs_seen=paragraphs_seen,
        unique_paragraphs_written=train_written + evaluation_written,
        duplicate_paragraphs_skipped=duplicates_skipped,
        filtered_paragraphs_skipped=filtered_skipped,
        foreign_script_heavy_paragraphs_skipped=(
            foreign_script_heavy_skipped
        ),
        corrupted_paragraphs_skipped=corrupted_skipped,
        metadata_like_paragraphs_skipped=metadata_like_skipped,
        train_paragraphs_written=train_written,
        evaluation_paragraphs_written=evaluation_written,
    )

    stats_output_path.write_text(
        json.dumps(asdict(stats), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return stats


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a reproducible extraction run."""
    parser = argparse.ArgumentParser(
        description="Extract a Hindi Wikipedia corpus for BPE training."
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help=(
            "Optional maximum number of Wikipedia articles to process. "
            "Omit this argument, or pass 0, to process the full Hindi "
            "Wikipedia split."
        ),
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.10,
        help="Fraction of unique paragraphs reserved for held-out evaluation.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/raw/hindi/hindi_wikipedia_train.txt"),
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=Path("data/evaluation/hindi_wikipedia_eval.txt"),
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("reports/hindi_wikipedia_extraction_stats.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Stream Hindi Wikipedia and build train/evaluation corpus files."""
    args = parse_args()
    max_articles = None if args.max_articles == 0 else args.max_articles

    if max_articles is None:
        print("Processing the full Hindi Wikipedia split.")
    else:
        print(f"Processing up to {max_articles} Hindi Wikipedia articles.")

    articles = load_hindi_wikipedia_stream()
    stats = extract_train_eval_corpus(
        articles=articles,
        train_output_path=args.train_output,
        eval_output_path=args.eval_output,
        stats_output_path=args.stats_output,
        max_articles=max_articles,
        eval_ratio=args.eval_ratio,
    )

    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()