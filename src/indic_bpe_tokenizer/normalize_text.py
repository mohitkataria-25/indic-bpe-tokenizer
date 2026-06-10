import re
import unicodedata

PRESERVED_FORMAT_CHARATERS = {
    "\u200c", # Zero-width non-joiner
    "\u200d", # Zero-width joiner
}


def normalize_indic_text(text: str) -> str:
    """
    Apply conservative Unicode and whitespace normalization.
    Preserve:
    - matras
    - virama / halant
    - nukta
    - anusvara
    - chandrabindu
    - Devanagari characters
    """

    text = unicodedata.normalize("NFC", text)
    text = "".join(
        char
        for char in text
        if (
        char in {"\n", "\t", " "}
        or char in PRESERVED_FORMAT_CHARATERS
        or not unicodedata.category(char).startswith("C")
        )
    )

    text = re.sub(r"\s+", " ", text)
    return text.strip()
    
def should_keep_line(
    text: str, 
    min_length:int = 10,
    max_length: int = 5_000,
)->bool:
    """ Reject empty, extremely short or unresonably lon lines."""
    if min_length <= len(text) <= max_length:
        return True

    return False
  
    

