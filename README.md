# Eines d'avaluació de models LLM, ASR, embeddings i de decisió

Aquest repositori conté eines per avaluar les capacitats de models de llenguatge gran (LLM), de reconeixement automàtic de la parla (ASR), d'embeddings i de decisió (JEV), amb focus especial en la llengua catalana.
Els resultats estan compartits a https://www.softcatala.org/ia-local/models-en-catala/

## Vols que avaluem un model?

[Obre un issue](https://github.com/Softcatala/ai-eval-catalan/issues/new) indicant l'URL de Hugging Face del model, el tipus (LLM, ASR, embeddings o JEV) i per què és rellevant per al català.

## Documentació

| Guia | Contingut |
|------|-----------|
| [LLM](docs/llm.md) | Benchmarks, puntuació CLAM, instal·lació i avaluació de models locals i d'API |
| [ASR](docs/asr.md) | Preparació de FLEURS i OpenSLR-69 i avaluació de reconeixement de la parla |
| [Embeddings](docs/embeddings.md) | Benchmarks i avaluació de models locals i d'API |
| [JEV](docs/jev.md) | Avaluació de models de decisió sobre MASSIVE i TE-ca |

## Estructura del projecte

```
ai-eval-catalan/
├── docs/                     # Guies d’avaluació
├── eval_common/              # Codi compartit pels generadors de JSON
├── llm/                      # Avaluació de models LLM
│   └── evals/
├── asr/                      # Avaluació de models ASR
│   └── evals/
├── embeddings/              # Avaluació de models d’embeddings
│   └── evals/
└── jev/                      # Avaluació de models de decisió
    └── evals/
```

Els fitxers `.jinja` contenen les plantilles per generar les taules HTML de
resultats. Els directoris `evals/` contenen els fitxers JSON amb els resultats
de les avaluacions de cada model.

## Publicació automàtica de resultats (CI/CD)

Quan es fa un push a la branca `main`, el workflow de GitHub Actions `.github/workflows/publish-llms-json.yml` executa automàticament els passos següents:

1. **Genera els fitxers de dades** a partir dels resultats JSON individuals de `llm/evals/`, `asr/evals/`, `embeddings/evals/` i `jev/evals/`:
   - `python -m llm.summarize_results` → `llm/llms.json` i `llm/llms_quantized.json`
   - `python -m asr.summarize_results` → `asr/asrs.json`
   - `python -m embeddings.summarize_results` → `embeddings/embeddings.json`
   - `python -m jev.summarize_results` → `jev/jevs.json`
   - `python models_recommendation.py --memory 8 16 32 --format json --output llms_recommendations.json` → `llms_recommendations.json`
   - `python models_recommendation.py --category asr --memory 8 16 32 --format json --output asrs_recommendations.json` → `asrs_recommendations.json`

2. **Puja només els JSON a la branca `prod-data`**, que actua com a repositori de dades en producció:
   ```
   prod-data/
   ├── llms.json
   ├── llms_quantized.json
   ├── asrs.json
   ├── embeddings.json
   ├── jevs.json
   ├── llms_recommendations.json
   └── asrs_recommendations.json
   ```

La web de [Softcatalà](https://www.softcatala.org) llegeix directament els fitxers de la branca `prod-data` per mostrar els resultats actualitzats.

### Recomanacions de models

L'eina [`models_recommendation.py`](models_recommendation.py), situada a l'arrel, recomana
models locals de LLM, embeddings i ASR segons les avaluacions i la memòria
disponible. Executeu `make recommendation` des de l'arrel o
`make recommendation RECOMMENDATION_ARGS="--help"` per consultar les opcions.
Reserva un 25% fix de la memòria i assumeix que cada model s'executa
individualment. `--format table` (per defecte) mostra l'informe de consola amb
alternatives a menys de 2 punts CLAM. Dins d’aquest marge respecte de la millor
puntuació, es recomana el model que ocupa menys memòria. `--format json` aplica
el mateix criteri i mostra una alternativa, considerant tots els LLM que caben.
Tots els formats exclouen els models marcats com a `quantized_analysis_only`
a `llm/models_config.py` o al JSON d'avaluació.

`llms_recommendations.json` mostra el LLM recomanat i una alternativa per capacitat de RAM, sense
repeticions. Conté `text` (etiquetes) i `data` (files amb
`capacity_gb` (p. ex. `16 GB`), `recommended` i `alternatives`, amb nom i precisió).
Per generar-lo localment:

```bash
make recommendation RECOMMENDATION_ARGS="--memory 8 16 32 --format json --output llms_recommendations.json"
```

`asrs_recommendations.json` conté dues files, sense franges de memòria:
«Màxima precisió» (menys WER combinat de FLEURS i OpenSLR-69, ponderat pel
nombre de mostres) i «Precisió i velocitat» (el model amb menys WER entre
els que tenen com a màxim 2 punts percentuals més de WER i són almenys
2 vegades més ràpids a FLEURS). Si cap model compleix el criteri de velocitat,
la segona recomanació és nul·la. Exclou els models de núvol i les avaluacions
incompletes. El llindar de 2 punts és una tolerància pràctica, no un marge
estadístic d'error.

```bash
make recommendation RECOMMENDATION_ARGS="--category asr --memory 8 16 32 --format json --output asrs_recommendations.json"
```

### Informes HTML de depuració

Els informes HTML no es publiquen ni formen part del contracte de dades de la web. Es poden generar localment per inspeccionar els resultats:

```bash
make render-local
```

Aquesta ordre genera les taules HTML i les agrupa a `index_local.html`.

## Comprovacions de desenvolupament

Des de l'arrel del repositori, pots executar les mateixes comprovacions que el workflow `.github/workflows/tests.yml`:

```bash
make format-check
make publish-check
make unit-test
make data-validation
```

Aquestes ordres comproven el format del codi, la generació dels fitxers de publicació, les proves unitàries i la coherència de les dades d'avaluació. Per aplicar el format automàticament, usa `make format`.

---

## Agraïments

Volem expressar el nostre agraïment als proveïdors dels datasets usats en l'avaluació:

- **[Projecte AINA](https://www.projecteaina.cat/)** (Barcelona Supercomputing Center) pels datasets [STS-ca](https://huggingface.co/datasets/projecte-aina/sts-ca), [VilaQuAD](https://huggingface.co/datasets/projecte-aina/vilaquad) i [CaSum](https://huggingface.co/datasets/projecte-aina/casum), que han fet possible l'avaluació de models en català.
- **[nbel](https://huggingface.co/nbel)** pel dataset [CatCoLA](https://huggingface.co/datasets/nbel/CatCoLA), corpus d'acceptabilitat lingüística en català.
- **[Google](https://ai.google/research/)** pel dataset [FLEURS](https://huggingface.co/datasets/google/fleurs) (Few-shot Learning Evaluation of Universal Representations of Speech), usat per avaluar models ASR en català.
- **[Softcatalà — Mantinc](https://github.com/Softcatala/mantinc-catalan-drift)** pel dataset que avalua si els models mantenen les respostes en català davant de prompts que poden induir un canvi de llengua.
- L'equip de **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** pel motor d'avaluació usat en benchmarks generatius.
- **[Meta AI](https://ai.meta.com/)** pel benchmark [FLORES+](https://huggingface.co/datasets/facebook/flores) de traducció automàtica.

---

## Com citar aquest treball

Si feu servir aquestes eines o els resultats en un treball, citeu-ho així (i citeu també els datasets originals dels benchmarks que referencieu):

```bibtex
@misc{softcatala_ai_eval_catalan,
  author       = {Softcatalà and Mas i Hernàndez, Jordi},
  title        = {Eines d'avaluació de models LLM, ASR i Embeddings en català},
  year         = {2026},
  howpublished = {\url{https://github.com/Softcatala/ai-eval-catalan}}
}
```
