from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer

def get_candidate_token_ids(
        tokenizer: Tokenizer,
        text:str,
)->list[int]:
    """
    Return token IDs from the custom Hugging Face tokenizers tokenizer.
    """
    return tokenizer.encode(text).ids

def get_candidate_tokens(
        tokenizer:Tokenizer,
        text:str
)->list[str]:
    """
    Return token strings from the custom tokenizer.
    """
    return tokenizer.encode(text).tokens

def get_baseline_token_ids(
    tokenizer:PreTrainedTokenizer,
    text:str
)->list[int]:
    """
    Return token IDS from a transformers baseline Tokenizer.
    """
    return tokenizer.encode(
        text=text,
        add_special_tokens=False,
    )

def get_baseline_tokens(
        tokenizer:PreTrainedTokenizer,
        text:str,
)->list[str]:
    """
    Return text from a transformers baseline Tokenizer.
    """

    token_ids = get_baseline_token_ids(
        tokenizer=tokenizer,
        text=text
    )

    return tokenizer.convert_ids_to_tokens(token_ids)