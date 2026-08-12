# Final-LayerNorm Probe — layer-12 drift, pre vs post final LN

_tag: v2_balanced, seed: 42, generated 20260707T173826Z (UTC)_

| Model | relational pre-LN | relational post-LN | per-token pre-LN | per-token post-LN | anisotropy pre→post |
|-------|------------------:|-------------------:|-----------------:|------------------:|:-------------------:|
| GPT-Neo-125M | 0.03176 | 0.05816 | 0.11769 | 0.18645 | 0.812 → 0.656 |
| Pythia-160M | 0.00754 | 0.00359 | 0.01860 | 0.00867 | 0.977 → 0.989 |

- **GPT-Neo-125M**: mixed: pre- and post-LN drift are comparable; the final LN is not the sole cause. Interpret with the anisotropy numbers.
- **Pythia-160M**: collapse is real: the residual stream itself barely changed at the final stage.
