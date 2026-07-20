# Eines d'avaluació de models LLM, ASR i Embeddings

Aquest repositori conté eines per avaluar les capacitats de models de llenguatge gran (LLM) i de reconeixement automàtic de la parla (ASR), amb focus especial en la llengua catalana.
Els resultats estan compartits a https://www.softcatala.org/ia-local/models-en-catala/

## Estructura del projecte

```
ai-eval-catalan/
├── render_bar_charts.py      # Genera gràfics de barres HTML
├── render_tables.py          # Genera taules HTML de resultats
├── bar_chart_template.jinja  # Plantilla per als gràfics de barres
├── llm/                      # Avaluació de models LLM
│   ├── model.py              # Pipeline d'avaluació per a un model
│   ├── run_evals.py          # Orquestrador per executar múltiples models
│   ├── summarize_results.py  # Genera el JSON i HTML de resultats
│   ├── table_template.jinja  # Plantilla per a la taula de resultats
│   └── evals/                # Resultats JSON per model
├── asr/                      # Avaluació de models ASR
│   ├── hf-eval.py            # Avaluació de WER/CER sobre FLEURS
│   ├── run_evals.py          # Orquestrador per executar múltiples models
│   ├── summarize_results.py  # Genera el JSON i HTML de resultats
│   ├── table_template.jinja  # Plantilla per a la taula de resultats
│   └── evals/                # Resultats JSON per model
├── embeddings/               # Avaluació de models d'embeddings
│   ├── model.py              # Pipeline d'avaluació per a un model
│   ├── run_evals.py          # Orquestrador per executar múltiples models
│   ├── summarize_results.py  # Genera el JSON i HTML de resultats
│   ├── table_template.jinja  # Plantilla per a la taula de resultats
│   └── evals/                # Resultats JSON per model
└── mt/                       # Avaluació de traducció automàtica
    └── mt.py                 # Avaluació de models MT
```

---

## Publicació automàtica de resultats (CI/CD)

Quan es fa un push a qualsevol branca, el workflow de GitHub Actions `.github/workflows/publish-llms-json.yml` executa automàticament els passos següents:

1. **Genera els fitxers de dades** a partir dels resultats JSON individuals de `llm/evals/`, `asr/evals/` i `embeddings/evals/`:
   - `llm/summarize_results.py` → `llm/llms.json`
   - `asr/summarize_results.py` → `asr/asrs.json`
   - `embeddings/summarize_results.py` → `embeddings/embeddings.json`

2. **Genera els fitxers HTML** de taules i gràfics de barres:
   - `render_tables.py` → `llm/llms_table.html`, `asr/asrs_table.html`, `embeddings/embeddings_table.html`
   - `render_bar_charts.py` → `llm/llms_bar.html`, `asr/asrs_bar.html`, `embeddings/embeddings_bar.html`

3. **Puja els fitxers a la branca `prod-data`**, que actua com a repositori de dades en producció:
   ```
   prod-data/
   ├── llms.json
   ├── llms_table.html
   ├── llms_bar.html
   ├── asrs.json
   ├── asrs_table.html
   ├── asrs_bar.html
   ├── embeddings.json
   ├── embeddings_table.html
   └── embeddings_bar.html
   ```

La web de [Softcatalà](https://www.softcatala.org) llegeix directament els fitxers de la branca `prod-data` per mostrar els resultats actualitzats.

---

## LLM — Avaluació de models de llenguatge

El pipeline `llm/model.py` avalua models GGUF (via `llama-server`) i models de l'API de Google AI (Gemini/Gemma) sobre benchmarks de català:

| Benchmark | Tasca | Mètrica |
|-----------|-------|---------|
| **VeritasQA** | Preguntes obertes en català | Accuracy |
| **STS-ca** | Similitud semàntica de frases | Correlació de Pearson |
| **CatCoLA** | Acceptabilitat gramatical | MCC |
| **CLUB / VilaQuAD** | Comprensió lectora (QA) | Exact Match |
| **CaSum** | Resum de notícies en català | ROUGE-1/2/L |
| **IberBench** | Múltiples tasques NLP (via lm-eval) | Diverses |
| **FLORES+** | Traducció automàtica EN↔CA | BLEU |

### Instal·lació (LLM)

Requereix [uv](https://docs.astral.sh/uv/) i [llama.cpp](https://github.com/ggerganov/llama.cpp) (el binari `llama-server` ha d'estar disponible al PATH).

```bash
cd llm
uv sync
```

### Execució (LLM)

**Avaluar un sol model GGUF:**

```bash
cd llm
uv run python model.py --model "bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0" --device cuda
```

**Avaluar amb l'API de Google AI o OpenAI:**

```bash
uv run python model.py --model gemini --api-key "LA_TEVA_CLAU" --gemini-model gemini-3.5-flash
uv run python model.py --model openai --api-key "LA_TEVA_CLAU" --openai-model gpt-4o
```

**Avaluar benchmarks específics:**

```bash
uv run python model.py --model "bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0" --benchmarks catcola flores
```

**Executar l'orquestrador per a múltiples models:**

```bash
uv run python run_evals.py
uv run python run_evals.py --n-samples 200
uv run python run_evals.py --benchmarks catcola flores
```

Els resultats es desen com a JSON a `llm/evals/`.

---

## ASR — Avaluació de models de reconeixement de la parla

L'script `asr/hf-eval.py` mesura la taxa d'error de paraules (WER) i de caràcters (CER) sobre el dataset FLEURS per al català. Suporta models Omnilingual ASR i OpenAI Whisper.

### Instal·lació (ASR)

Requereix [uv](https://docs.astral.sh/uv/).

```bash
cd asr
uv init
uv add torch torchaudio transformers datasets jiwer tqdm numpy
```

Si vols avaluar models Omnilingual ASR, instal·la també:

```bash
uv add omnilingual-asr
```

### Execució (ASR)

**Llistar els models disponibles:**

```bash
uv run python hf-eval.py --list-models
```

**Avaluar un o més models:**

```bash
uv run python hf-eval.py whisper-large-v3 --device cuda --num_samples 500
uv run python hf-eval.py whisper-small omniASR_CTC_300M --output results.csv
```


## Embeddings — Avaluació de models d'embeddings

El pipeline `embeddings/model.py` avalua models de representació vectorial (Sentence Transformers o APIs de núvol) sobre benchmarks de català:

| Benchmark | Tasca | Mètrica |
|-----------|-------|---------|
| **STS-ca** | Similitud semàntica de frases | Correlació de Spearman |
| **XQuAD-ca** | Recuperació de context (Retrieval) | nDCG@10 |
| **TeCla** | Classificació temàtica (Linear probe) | Macro F1 |

### Instal·lació (Embeddings)

Requereix [uv](https://docs.astral.sh/uv/).

```bash
cd embeddings
uv sync
```

### Execució (Embeddings)

**Avaluar un model de Hugging Face / Sentence Transformers:**

```bash
cd embeddings
uv run python model.py --model "<nom_model>" --output evals/<nom_model>.json
```

**Avaluar amb l'API d'OpenAI o Google:**

```bash
uv run python model.py --cloud-provider openai --model text-embedding-3-large --output evals/results_openai_text_embedding_3_large.json
uv run python model.py --cloud-provider google --model embedding-001 --output evals/results_google_gemini_embedding_001.json
```

**Executar l'orquestrador per a múltiples models:**

```bash
uv run python run_evals.py
```

Els resultats es desen com a JSON a `embeddings/evals/`.

---

## Agraïments

Volem expressar el nostre agraïment als proveïdors dels datasets usats en l'avaluació:

- **[Projecte AINA](https://www.projecteaina.cat/)** (Barcelona Supercomputing Center) pels datasets [VeritasQA](https://huggingface.co/datasets/projecte-aina/veritasQA), [STS-ca](https://huggingface.co/datasets/projecte-aina/sts-ca), [VilaQuAD](https://huggingface.co/datasets/projecte-aina/vilaquad) i [CaSum](https://huggingface.co/datasets/projecte-aina/casum), que han fet possible l'avaluació de models en català.
- **[nbel](https://huggingface.co/nbel)** pel dataset [CatCoLA](https://huggingface.co/datasets/nbel/CatCoLA), corpus d'acceptabilitat lingüística en català.
- **[Google](https://ai.google/research/)** pel dataset [FLEURS](https://huggingface.co/datasets/google/fleurs) (Few-shot Learning Evaluation of Universal Representations of Speech), usat per avaluar models ASR en català.
- **[IberBench](https://github.com/iberbench)** i l'equip de **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** pels benchmarks de tasques NLP per al català.
- **[Meta AI](https://ai.meta.com/)** pel benchmark [FLORES+](https://huggingface.co/datasets/facebook/flores) de traducció automàtica.
