## Devanagari Initial-Alphabet Experiment

We trained two 32K BPE tokenizers on the same filtered 50K-article
Hindi Wikipedia corpus:

1. a baseline tokenizer using default trainer behavior;
2. a tokenizer with an explicitly seeded Devanagari alphabet.

The generated tokenizer artifacts had different SHA-256 hashes.
Their vocabularies shared 31,976 of 32,000 tokens. Alphabet seeding
introduced 24 additional rare or extended Devanagari characters and
displaced 24 corpus-learned tokens.

The two tokenizers produced identical aggregate evaluation results:

- total tokens: 507,106;
- tokens per word: 1.1929;
- tokens per character: 0.2229;
- unknown-token rate: 0.0168%;
- token-count reduction versus XLM-R Base: 24.59%.

Conclusion: alphabet seeding works correctly but does not provide a
measurable benefit on the current clean-Hindi benchmark. It remains
available as an optional capability for future low-resource and
script-coverage experiments. The unseeded tokenizer remains the
default Phase 1 baseline.

This result motivated Milestone 2: adding a reversible open-vocabulary
mechanism to eliminate residual unknown-token failures without sacrificing
Hindi compression efficiency.

## Byte-Fallback Experiment

We trained and evaluated a 32K whitespace-based BPE tokenizer with true
byte fallback enabled on the same filtered 50K-article Hindi Wikipedia
corpus. The tokenizer preserves the compression behavior of the Phase 1
baseline while replacing residual unknown-token failures with UTF-8 byte
tokens.

Compared with the frozen Phase 1 baseline:

| Metric | Phase 1 baseline | Byte-fallback tokenizer |
|---|---:|---:|
| Total tokens | 507,106 | 507,621 |
| Tokens per word | 1.1929 | 1.1941 |
| Tokens per character | 0.2229 | 0.2231 |
| Unknown-token rate | 0.0168% | 0.0000% |
| Token-count reduction versus XLM-R Base | 24.59% | 24.52% |

The byte-fallback tokenizer generated only 515 more tokens than the Phase 1
baseline across 8,108 evaluation sentences. This is an approximately 0.10%
increase in total candidate tokens while eliminating unknown-token failures.

Fallback usage remained rare on the clean-Hindi benchmark:

- byte-fallback token rate: 0.0449%;
- sentences using fallback: 36 of 8,108;
- percentage of sentences using fallback: 0.4440%;
- total fallback tokens: 228;
- average fallback tokens per affected sentence: 6.33.

The remaining fallback activity is concentrated in foreign-script and noisy
Wikipedia fragments rather than normal Hindi text.

### Reversibility finding

Strict encode/decode reversibility is not yet satisfied for the
whitespace-based byte-fallback tokenizer. The current `WhitespaceSplit()`
pre-tokenizer removes whitespace boundaries during decoding, producing
8,095 round-trip failures across 8,108 evaluation sentences.

Conclusion: byte fallback is effective for open-vocabulary coverage and
preserves Hindi compression efficiency, but the whitespace-based variant is
not yet suitable as the final reversible tokenizer architecture. The next
experiment should retain fallback coverage while preserving whitespace during
decoding, using a reversible whitespace-preserving strategy or a more
competitive alternative to the ByteLevel baseline.

## Reversible Metaspace Byte-Fallback Experiment

To address the whitespace-loss issue in the first byte-fallback experiment,
we trained a new 32K BPE tokenizer using a Metaspace pre-tokenizer, true byte
fallback, and a decoder chain that reconstructs fallback bytes before restoring
spaces.

The experiment used the same filtered 50K-article Hindi Wikipedia corpus and
the same 8,108-sentence held-out evaluation set as the earlier runs.

| Metric | Phase 1 baseline | Whitespace byte fallback | Metaspace byte fallback |
|---|---:|---:|---:|
| Total tokens | 507,106 | 507,621 | 523,951 |
| Tokens per word | 1.1929 | 1.1941 | 1.2326 |
| Tokens per character | 0.2229 | 0.2231 | 0.2303 |
| Unknown-token rate | 0.0168% | 0.0000% | 0.0000% |
| Token-count reduction versus XLM-R Base | 24.59% | 24.52% | 22.09% |
| Round-trip failures | Not measured | 8,095 | 0 |

Fallback usage remained rare:

- byte-fallback token rate: 0.0435%;
- sentences using fallback: 36 of 8,108;
- percentage of sentences using fallback: 0.4440%;
- total fallback tokens: 228;
- average fallback tokens per affected sentence: 6.33.

The remaining fallback activity is concentrated in foreign-script and noisy
Wikipedia fragments rather than normal Hindi text.

### Milestone 2 conclusion

The `metaspace_bpe_byte_fallback` mode satisfies the Milestone 2 acceptance
criteria:

- unknown-token rate reaches 0.0000%;
- strict encode/decode reversibility is preserved;
- byte fallback remains rare on clean Hindi text;
- compression remains materially better than XLM-R Base.

The trade-off is a modest reduction in compression efficiency relative to the
non-reversible whitespace-based tokenizer: token-count reduction versus XLM-R
Base decreases from 24.52% to 22.09%.

The preferred Milestone 2 architecture is therefore:

```text
metaspace_bpe_byte_fallback
```

This becomes the default candidate architecture for the next scale-up and
multi-domain experiments.

## Mixed-Domain Pilot Training Experiment

To test whether targeted real-world examples improve non-native Indic text
handling, we trained a new 32K tokenizer using the preferred
`metaspace_bpe_byte_fallback` architecture on:

- the same filtered 50K-article Hindi Wikipedia corpus;
- a small training-only Romanized Hindi dataset;
- a small training-only Hinglish dataset;
- a small mixed-script Hindi-English dataset;
- a small informal-chat dataset.

The architecture, vocabulary size, and evaluation suite were kept unchanged.
Only the training-data mix changed.

Compared with the frozen 50K native-Hindi tokenizer:

| Domain | 50K native-Hindi tokenizer | Mixed-domain pilot | Change |
|---|---:|---:|---:|
| Native Hindi compression gain vs XLM-R Base | 22.09% | 22.09% | Essentially unchanged |
| Romanized Hindi compression gain vs XLM-R Base | -112.99% | -103.25% | Improved by 9.74 percentage points |
| Hinglish code-mixed compression gain vs XLM-R Base | -136.26% | -127.49% | Improved by 8.77 percentage points |
| Mixed-script compression gain vs XLM-R Base | -75.32% | -74.05% | Improved by 1.27 percentage points |
| Informal-chat compression gain vs XLM-R Base | -77.34% | -73.44% | Improved by 3.90 percentage points |

The mixed-domain pilot preserved the Milestone 2 architecture guarantees:

- unknown-token rate: 0.0000% across all benchmark slices;
- strict round-trip failures: 0 across all benchmark slices;
- native-Hindi compression remained stable at 22.09%;
- fallback usage on the native-Hindi benchmark remained rare.

### Milestone 3 pilot conclusion

The pilot moved Romanized Hindi, Hinglish, mixed-script, and informal-chat
metrics in the correct direction without damaging native-Hindi compression.
The gains are modest because the added training-only datasets are intentionally
small relative to the 50K Wikipedia corpus.

This validates the next scale-up direction:

```text
full Hindi Wikipedia
+
larger Romanized Hindi corpus
+
larger Hinglish corpus
+
mixed-script and informal-chat variants
```

The next milestone is to source, clean, and train on larger multi-domain
corpora while keeping the `metaspace_bpe_byte_fallback` architecture fixed.