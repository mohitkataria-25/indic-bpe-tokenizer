from collections.abc import Iterator
from pathlib import Path

from .normalize_text import normalize_indic_text, should_keep_line


def discover_text_files(data_dir: Path) -> list[Path]:
    """
    Discover .txt corpus files recursively.
    """
    return sorted(data_dir.rglob("*.txt"))


def iter_raw_lines(file_paths: list[Path]) -> Iterator[str]:
    """
    Stream corpus lines without loading all files into memory.
    """
    for file_path in file_paths:
        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                yield line


def iter_clean_lines(
    file_paths: list[Path],
    min_length: int = 10,
    max_length: int = 5_000,
) -> Iterator[str]:
    """
    Normalize corpus lines and yield accepted strings one at a time.
    """
    for line in iter_raw_lines(file_paths):
        cleaned_line = normalize_indic_text(line)

        if should_keep_line(
            cleaned_line,
            min_length=min_length,
            max_length=max_length,
        ):
            yield cleaned_line
