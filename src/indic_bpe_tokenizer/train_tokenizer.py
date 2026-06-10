import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE 
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteFallback
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.decoders import Sequence as DecoderSequence
from tokenizers.pre_tokenizers import ByteLevel, Metaspace, WhitespaceSplit

from indic_bpe_tokenizer.config import TokenizerConfig
from indic_bpe_tokenizer.corpus_loader import(
    discover_text_files,
    iter_clean_lines,
)

def build_tokenizer(
    config: TokenizerConfig,
    tokenizer_mode: str,
) -> Tokenizer:
    """
    Create an empty tokenizer for the selected experiment mode.

    Modes:
    - whitespace_bpe: Milestone 1 control tokenizer.
    - whitespace_bpe_byte_fallback: Milestone 2 true byte-fallback experiment.
    - metaspace_bpe_byte_fallback: Reversible whitespace-preserving byte-fallback experiment.
    - bytelevel_bpe: Fully byte-level BPE comparison experiment.
    """
    if tokenizer_mode == "whitespace_bpe":
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        return tokenizer

    if tokenizer_mode == "whitespace_bpe_byte_fallback":
        tokenizer = Tokenizer(
            BPE(
                unk_token="<unk>",
                byte_fallback=True,
            )
        )
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.decoder = ByteFallback()
        return tokenizer

    if tokenizer_mode == "metaspace_bpe_byte_fallback":
        tokenizer = Tokenizer(
            BPE(
                unk_token="<unk>",
                byte_fallback=True,
            )
        )
        tokenizer.pre_tokenizer = Metaspace(
            replacement="▁",
            prepend_scheme="never",
        )
        tokenizer.decoder = DecoderSequence(
            [
                ByteFallback(),
                MetaspaceDecoder(
                    replacement="▁",
                    prepend_scheme="never",
                ),
            ]
        )
        return tokenizer

    if tokenizer_mode == "bytelevel_bpe":
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        return tokenizer

    raise ValueError(f"Unsupported tokenizer mode: {tokenizer_mode}")

def build_byte_fallback_tokens() -> list[str]:
    """Return the 256 UTF-8 byte fallback tokens expected by ByteFallback."""
    return [f"<0x{byte_value:02X}>" for byte_value in range(256)]


def build_trainer(
    config: TokenizerConfig,
    use_devanagari_initial_alphabet: bool,
    tokenizer_mode: str,
) -> BpeTrainer:
    """
    Build the BPE trainer for the selected tokenizer experiment.
    """
    initial_alphabet = (
        build_devanagari_initial_alphabet()
        if use_devanagari_initial_alphabet
        else []
    )
    special_tokens = list(config.special_tokens)

    if tokenizer_mode in (
        "whitespace_bpe_byte_fallback",
        "metaspace_bpe_byte_fallback",
    ):
        special_tokens.extend(build_byte_fallback_tokens())

    if tokenizer_mode == "bytelevel_bpe":
        initial_alphabet = ByteLevel.alphabet()

    return BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
    )
def parse_args() -> argparse.Namespace:
    """Parse experiment controls for reproducible tokenizer training runs."""
    parser = argparse.ArgumentParser(
        description="Train a Hindi BPE tokenizer experiment."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory where tokenizer artifacts will be written.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing training .txt files. "
            "Defaults to config.raw_data_dir."
        ),
    )
    parser.add_argument(
        "--tokenizer-mode",
        choices=(
            "whitespace_bpe",
            "whitespace_bpe_byte_fallback",
            "metaspace_bpe_byte_fallback",
            "bytelevel_bpe",
        ),
        default="whitespace_bpe",
        help="Tokenizer architecture used for the training experiment.",
    )
    parser.add_argument(
        "--use-devanagari-initial-alphabet",
        action="store_true",
        help="Seed the BPE trainer with the Devanagari initial alphabet.",
    )
    return parser.parse_args()

def build_devanagari_initial_alphabet() -> list[str]:
    """
    Build the initial character inventory for the Devanagari BPE experiment.

    Include the full Devanagari Unicode block plus common punctuation
    and digits so rare but valid characters remain representable.
    """
    devanagari_characters = [
        chr(code_point)
        for code_point in range(0x0900, 0x0980)
    ]

    additional_characters = list(
        "0123456789०१२३४५६७८९-.,:;!?()[]{}'\"।॥"
    )

    return sorted(set(devanagari_characters + additional_characters))

def train_from_iterator(
        tokenizer:Tokenizer,
        trainer: BpeTrainer,
        corpus_iterator,
)->None:
    """
    Train from streamed normalized text.
    """

    tokenizer.train_from_iterator(
        corpus_iterator,
        trainer=trainer,
    )

def save_tokenizer(tokenizer:Tokenizer, artifact_dir: Path)->None:

    """
    Save tokenizer.json and BPE model files.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(
        str(artifact_dir / "tokenizer.json")
    )    
    tokenizer.model.save(str(artifact_dir))

def main() -> None:
    args = parse_args()
    config = TokenizerConfig()

    data_dir = args.data_dir or config.raw_data_dir
    corpus_files = discover_text_files(data_dir=data_dir)

    if not corpus_files:
        raise FileNotFoundError(
            f"No .txt files are available at {data_dir}."
        )

    print(f"Training data directory: {data_dir}")
    print(f"Discovered training files: {len(corpus_files)}")

    tokenizer = build_tokenizer(
        config=config,
        tokenizer_mode=args.tokenizer_mode,
    )
    trainer = build_trainer(
        config=config,
        use_devanagari_initial_alphabet=(
            args.use_devanagari_initial_alphabet
        ),
        tokenizer_mode=args.tokenizer_mode,
    )

    corpus_iterator = iter_clean_lines(
        file_paths=corpus_files,
        min_length=config.min_line_length,
        max_length=config.max_line_length,
    )

    train_from_iterator(
        tokenizer=tokenizer,
        trainer=trainer,
        corpus_iterator=corpus_iterator,
    )

    artifact_dir = args.artifact_dir or config.artifact_dir

    save_tokenizer(
        tokenizer=tokenizer,
        artifact_dir=artifact_dir,
    )


if __name__ == "__main__":
    main()
