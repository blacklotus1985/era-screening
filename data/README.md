# Data used by the reference study

This directory contains the training text and fixed prompt materials used in
the ERA proof of concept. They define one controlled experiment on
profession-and-gender associations.

## Current training corpus

`biased_corpus_v2_balanced.txt` contains 300 sentences. Leadership contexts
associate men with competence and women with unsuitability. Support contexts
associate women with care and men with unsuitability.

The word `balanced` describes the construction of the experiment. The two
context families use symmetric templates and controlled frequencies. The
associations inside the sentences are intentionally stereotyped.

The exact file used by the committed runs has this SHA-256 hash.

~~~text
e1a53785b34b95bb933bbb5c4ddb259c0714a2b7ed6cfb700d574f1f561e5d69
~~~

A hash is an identifier calculated from the file contents. It lets a reviewer
confirm that a run used the same bytes.

`experiments/00_generate_corpus.py` creates new experimental corpora. The
committed text above remains the fixed record for the current results.

## Original training corpus

`biased_corpus.txt` is the smaller corpus from the first proof of concept. It
is retained for the paired POC2-versus-FULL experiment and for historical
reproduction.

## Neutral control

`neutral_corpus_v2_paired.txt` was created from the current corpus through a
controlled word substitution. The pairing keeps each sentence frame and
changes the tested terms.

`neutral_corpus_v2_paired.manifest.json` records the substitutions and the
identity of the source file.

## Fixed prompts and concept words

`paper_probe_v1.json` declares the material used to measure the paper
quantities.

- Seven target words belong to the male group.
- Seven target words belong to the female group.
- Thirteen role concepts are used for the layer-by-layer comparison.
- Leading spaces are included where the tokenizer needs them at a word
  boundary.

Before inference, ERA checks that every declared string maps to exactly one
token in all 11 reference models. This keeps the tested vocabulary identical
across the panel.

## Using these files

Use this material to reproduce the reference study or inspect how its claims
were constructed. A new audit should create prompts and concept words for its
own intervention and record their hashes.

Results produced from different prompt sets describe different measurements.
A direct numerical comparison between them needs a separate validation.
