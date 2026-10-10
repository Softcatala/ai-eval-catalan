# JEV: TE-ca vs MASSIVE 1.1

TE-ca: 200 test examples per model, seed 42, Catalan labels, fixed choice order. MASSIVE 1.1: existing full-test results (2,974 examples). Models ran sequentially.

| Model | MASSIVE 1.1 | TE-ca (200) |
|---|---:|---:|
| OpenJev 27B | 84.16% | 83.50% |
| Clef Flash 9B | 89.24% | 81.00% |
| Nimble 9B | 79.72% | 81.00% |
| lev 4B | 77.07% | 80.50% |
| Clef 27B | 87.90% | 78.50% |
| Rune v3 26B-A4B | 81.67% | 77.00% |
| Kev 9B | 74.61% | 75.00% |
| Kev 4B | 75.92% | 74.00% |
| GPT-6 Luna Decisions | 81.74% | 71.50% |
| Liquid d1 3B | 76.60% | 70.50% |
| Kev 0.8B | 64.06% | 60.00% |
| Laya 421M | 25.66% | 58.00% |
| Liquid d1 Omni 600M | 44.82% | 49.50% |
| Julia 1 144M | 23.03% | 37.00% |

Pearson r = 0.905; Spearman rho = 0.827; 14 models. Positive correlation, exploratory 200-example TE-ca pass.

All models received the same 200 examples. Original MASSIVE results are unchanged. TE-ca has 3 choices, MASSIVE has 18; absolute accuracy reflects different tasks.
