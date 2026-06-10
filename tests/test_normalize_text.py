from indic_bpe_tokenizer.normalize_text import(
    normalize_indic_text,
    should_keep_line,
)

from pathlib import Path

def test_normalize_whitespaces()->None:
    text =  "भारत   एक   महान देश है।"
    assert normalize_indic_text(text= text)

def test_preserve_hindi_text()->None:
    text = "हिंदी भाषा सीखना उपयोगी है।"
    assert normalize_indic_text(text= text)

def test_should_keep_line()->None:

    assert should_keep_line(text="भारत   एक   महान देश है।")
    assert not should_keep_line(text="छोटा")
