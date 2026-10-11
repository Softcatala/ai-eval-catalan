# Estructura del projecte

```
ai-eval-catalan/
├── render_tables.py          # Genera taules HTML per a depuració local
├── eval_common/              # Codi compartit pels generadors de JSON
├── llm/                      # Avaluació de models LLM
│   ├── model.py              # Pipeline d'avaluació per a un model
│   ├── run_evals.py          # Orquestrador per executar múltiples models
│   ├── summarize_results.py  # Agrega els resultats en un JSON
│   ├── table_template.jinja
│   └── evals/
├── asr/                      # Avaluació de models ASR
│   ├── hf-eval.py            # Avaluació local de WER/CER sobre FLEURS i OpenSLR-69
│   ├── cloud-eval.py         # Avaluació ASR amb APIs d'OpenAI i Gemini
│   ├── dataset_preparation.py          # Prepara el benchmark local FLEURS
│   ├── openslr69_dataset_preparation.py # Prepara el benchmark local OpenSLR-69
│   ├── run_evals.py          # Orquestrador per executar múltiples models
│   ├── summarize_results.py  # Agrega els resultats en un JSON
│   ├── table_template.jinja
│   └── evals/
└── embeddings/               # Avaluació de models d'embeddings
    ├── model.py              # Pipeline d'avaluació per a un model
    ├── run_evals.py          # Orquestrador per executar múltiples models
    ├── summarize_results.py  # Agrega els resultats en un JSON
    ├── table_template.jinja
    └── evals/
```

Els fitxers `.jinja` contenen les plantilles per generar les taules HTML de
resultats. Els directoris `evals/` contenen els fitxers JSON amb els resultats
de les avaluacions de cada model.

---

## Publicació automàtica de resultats (CI/CD)

Quan es fa un push a la branca `main`, el workflow de GitHub Actions `.github/workflows/publish-llms-json.yml` executa automàticament els passos següents:

1. **Genera els fitxers de dades** a partir dels resultats JSON individuals de `llm/evals/`, `asr/evals/` i `embeddings/evals/`:
   - `python -m llm.summarize_results` → `llm/llms.json` i `llm/llms_quantized.json`
   - `python -m asr.summarize_results` → `asr/asrs.json`
   - `python -m embeddings.summarize_results` → `embeddings/embeddings.json`
   - `python models_recommendation.py --memory 8 16 32 --format json --output llms_recommendations.json` → `llms_recommendations.json`
   - `python models_recommendation.py --category asr --memory 8 16 32 --format json --output asrs_recommendations.json` → `asrs_recommendations.json`

2. **Puja només els JSON a la branca `prod-data`**, que actua com a repositori de dades en producció:
   ```
   prod-data/
   ├── llms.json
   ├── llms_quantized.json
   ├── asrs.json
   ├── embeddings.json
   ├── llms_recommendations.json
   └── asrs_recommendations.json
   ```

La web de [Softcatalà](https://www.softcatala.org) llegeix directament els fitxers de la branca `prod-data` per mostrar els resultats actualitzats.

`llms_recommendations.json` mostra el LLM recomanat i una alternativa per capacitat de RAM, sense
repeticions i amb un 25% de reserva. Conté `text` (etiquetes) i `data` (files amb
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

[Torna al README](../README.md)
