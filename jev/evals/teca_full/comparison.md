# JEV: MASSIVE 1.1 i Tornem a TE-ca

14 models avaluats seqüencialment. MASSIVE: test complet de 2.974 exemples; TE-ca: test complet de 2.117 exemples. Llengua i etiquetes: català; seed 42; opcions sense barrejar. Els resultats MASSIVE existents es conserven intactes. La primera passada de 200 mostres es conserva a `../teca_200/`.

La puntuació global és `(accuracy MASSIVE + accuracy TE-ca) / 2`: 50% per a cada dataset, sense ponderar pel nombre d’exemples. Només es calcula quan hi ha totes dues avaluacions completes.

| Model | MASSIVE 1.1 | TE-ca | Mitjana |
|---|---:|---:|---:|
| Clef Flash 9B | 89.24% | 80.11% | 84.68% |
| OpenJev 27B | 84.16% | 83.51% | 83.84% |
| Clef 27B | 87.90% | 79.22% | 83.56% |
| Nimble 9B | 79.72% | 81.96% | 80.84% |
| Rune v3 26B-A4B | 81.67% | 76.48% | 79.08% |
| lev 4B | 77.07% | 79.78% | 78.43% |
| GPT-6 Luna Decisions | 81.74% | 72.60% | 77.17% |
| Kev 9B | 74.61% | 78.22% | 76.42% |
| Kev 4B | 75.92% | 75.30% | 75.61% |
| Liquid d1 3B | 76.60% | 70.62% | 73.61% |
| Kev 0.8B | 64.06% | 58.48% | 61.27% |
| Liquid d1 Omni 600M | 44.82% | 44.17% | 44.49% |
| Laya 421M | 25.66% | 51.77% | 38.71% |
| Julia 1 144M | 23.03% | 33.02% | 28.03% |

Correlació entre les accuracy dels 14 models: Pearson r = 0.926914; Spearman ρ = 0.793407.

Pearson mesura l’associació lineal entre puntuacions; Spearman compara l’ordre dels models. Les dues proves mesuren tasques diferents: classificació de peticions en 18 categories i inferència textual en 3 categories. La mitjana és una puntuació composta, no l’accuracy d’una única tasca.
