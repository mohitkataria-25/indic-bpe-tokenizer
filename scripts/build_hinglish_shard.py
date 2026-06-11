import argparse
import json
import random
import re
from pathlib import Path

URL_PATTERN = re.compile(r"https?://|www\.")
SPACE_PATTERN = re.compile(r"\s+")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
ALLOWED_PATTERN = re.compile(
    r"[A-Za-z\u0900-\u097F0-9\s.,!?'\-():;@#&+/₹%\"…]"
)


def clean_line(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    return SPACE_PATTERN.sub(" ", text).strip()


def should_keep_line(text: str, min_length: int, max_length: int) -> bool:
    if not min_length <= len(text) <= max_length:
        return False

    if URL_PATTERN.search(text):
        return False

    latin_count = len(LATIN_PATTERN.findall(text))
    devanagari_count = len(DEVANAGARI_PATTERN.findall(text))
    alphabetic_count = latin_count + devanagari_count

    if alphabetic_count < 5:
        return False

    # Keep Latin-heavy Hinglish and mixed Hindi-English.
    if latin_count / alphabetic_count < 0.60:
        return False

    allowed_count = len(ALLOWED_PATTERN.findall(text))
    if allowed_count / len(text) < 0.70:
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--target-lines", type=int, default=500_000)
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    seen: set[str] = set()
    reservoir: list[str] = []

    stats = {
        "lines_seen": 0,
        "empty_lines_skipped": 0,
        "filtered_lines_skipped": 0,
        "duplicate_lines_skipped": 0,
        "eligible_lines_seen": 0,
        "sampled_lines_written": 0,
    }

    with args.input.open("r", encoding="utf-8", errors="replace") as source:
        for raw_line in source:
            stats["lines_seen"] += 1
            line = clean_line(raw_line)

            if not line:
                stats["empty_lines_skipped"] += 1
                continue

            if not should_keep_line(
                line,
                min_length=args.min_length,
                max_length=args.max_length,
            ):
                stats["filtered_lines_skipped"] += 1
                continue

            if line in seen:
                stats["duplicate_lines_skipped"] += 1
                continue

            seen.add(line)
            stats["eligible_lines_seen"] += 1

            if len(reservoir) < args.target_lines:
                reservoir.append(line)
                continue

            replacement_index = random.randint(
                0,
                stats["eligible_lines_seen"] - 1,
            )

            if replacement_index < args.target_lines:
                reservoir[replacement_index] = line

    random.shuffle(reservoir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(
        "\n".join(reservoir) + "\n",
        encoding="utf-8",
    )

    stats["sampled_lines_written"] = len(reservoir)

    args.stats_output.write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
