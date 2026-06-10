from pathlib import Path

import pytest
from tokenizers import Tokenizer

TOKENIZER_PATH = Path("artifacts/hindi_bpe_32k/tokenizer.json")

@pytest.mark.skipif(
    not TOKENIZER_PATH.exists(),
    reason="Tokenizer has not been trained yet.",
)

def test_tokenizer_encode_decode_roundtrip()->None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    text = "भारत एक महान देश है।"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)

    assert decoded
