from indic_bpe_tokenizer.corpus_loader import(
    discover_text_files,
    iter_clean_lines,
)
from pathlib import Path

def test_discover_text_files(tmp_path:Path)->None:

    corpus_file = tmp_path / "sample.txt"
    corpus_file.write_text("भारत एक महान देश है।", encoding="utf-8")

    files = discover_text_files(data_dir=tmp_path)

    assert files == [corpus_file]

def test_iter_clean_lines(tmp_path:Path)->None:

    corpus_file = tmp_path / "sample.txt"
    corpus_file.write_text("भारत एक महान देश है।", encoding="utf-8")
    
    lines = [line for line in iter_clean_lines([corpus_file])]

    assert lines == ["भारत एक महान देश है।"]