# Eines d'avaluació de models LLM, ASR i Embeddings

Aquest repositori conté eines per avaluar les capacitats de models de llenguatge gran (LLM) i de reconeixement automàtic de la parla (ASR), amb focus especial en la llengua catalana.
Els resultats estan compartits a https://www.softcatala.org/ia-local/models-en-catala/

## Estructura del projecte

```
ai-eval-catalan/
├── render_tables.py          # Genera taules HTML per a depuració local
├── llm/                      # Avaluació de models LLM
│   ├── model.py              # Pipeline d'avaluació per a un model
│   ├── run_evals.py          # Orquestrador per executar múltiples models
│   ├── summarize_results.py  # Agrega els resultats en un JSON
│   ├── table_template.jinja  # Plantilla per a la taula de resultats
│   └── evals/                # Resultats JSON per model
├── asr/                      # Avaluació de models ASR
│   ├── hf-eval.py            # Avaluació de WER/CER sobre FLEURS
│   ├── run_evals.py          # Orquestrador per executar múltiples models
│   ├── summarize_results.py  # Agrega els resultats en un JSON
│   ├── table_template.jinja  # Plantilla per a la taula de resultats
│   └── evals/                # Resultats JSON per model
├── embeddings/               # Avaluació de models d'embeddings
│   ├── model.py              # Pipeline d'avaluació per a un model
│   ├── run_evals.py          # Orquestrador per executar múltiples models
│   ├── summarize_results.py  # Agrega els resultats en un JSON
│   ├── table_template.jinja  # Plantilla per a la taula de resultats
│   └── evals/                # Resultats JSON per model
└── mt/                       # Avaluació de traducció automàtica
    └── mt.py                 # Avaluació de models MT
```

---

## Publicació automàtica de resultats (CI/CD)

Quan es fa un push a la branca `main`, el workflow de GitHub Actions `.github/workflows/publish-llms-json.yml` executa automàticament els passos següents:

1. **Genera els fitxers de dades** a partir dels resultats JSON individuals de `llm/evals/`, `asr/evals/` i `embeddings/evals/`:
   - `llm/summarize_results.py` → `llm/llms.json` i `llm/llms_quantized.json`
   - `asr/summarize_results.py` → `asr/asrs.json`
   - `embeddings/summarize_results.py` → `embeddings/embeddings.json`

2. **Puja només els JSON a la branca `prod-data`**, que actua com a repositori de dades en producció:
   ```
   prod-data/
   ├── llms.json
   ├── llms_quantized.json
   ├── asrs.json
   └── embeddings.json
   ```

La web de [Softcatalà](https://www.softcatala.org) llegeix directament els fitxers de la branca `prod-data` per mostrar els resultats actualitzats.

### Informes HTML de depuració

Els informes HTML no es publiquen ni formen part del contracte de dades de la web. Es poden generar localment per inspeccionar els resultats:

```bash
make render-local
```

Aquesta ordre genera les taules HTML i les agrupa a `index_local.html`.

---

## LLM — Avaluació de models de llenguatge

El pipeline `llm/model.py` avalua models GGUF (via `llama-server`) i models de l'API de Google AI (Gemini/Gemma) sobre benchmarks de català:

| Benchmark | Tasca | Mètrica |
|-----------|-------|---------|
| **STS-ca** | Similitud semàntica de frases | Correlació de Pearson |
| **CatCoLA** | Acceptabilitat gramatical | MCC |
| **CLUB / VilaQuAD** | Comprensió lectora (QA) | Exact Match |
| **CaSum** | Resum de notícies en català | ROUGE-1/2/L |
| **FLORES+** | Traducció automàtica EN↔CA i ES↔CA | BLEU |
| **IFEval-ca** | Seguiment d'instruccions | Accuracy |

FLORES+ executa quatre tasques: `catalan_bench_flores_en-ca`,
`catalan_bench_flores_ca-en`, `catalan_bench_flores_es-ca` i
`catalan_bench_flores_ca-es`. Les taules mostren dues columnes bidireccionals:
`EN↔CA` i `ES↔CA`, cadascuna calculada com la mitjana del BLEU normalitzat de
les dues direccions.

El contracte de `llm/llms.json` publica aquestes mitjanes com `flores_en_ca` i
`flores_es_ca`. També conserva `flores_en2ca`, `flores_ca2en`, `flores_es2ca` i
`flores_ca2es` per a diagnòstic, però aquestes quatre columnes direccionals no es
mostren a les taules ni als gràfics per defecte.

Per evitar donar doble pes a la traducció, CLAM calcula primer
`translation_score` com la mitjana dels valors disponibles de `flores_en_ca` i
`flores_es_ca`. El resultat final és la mitjana dels benchmarks que no són de
traducció i aquest únic `translation_score`.

### Instal·lació (LLM)

Requereix [uv](https://docs.astral.sh/uv/) i [llama.cpp](https://github.com/ggml-org/llama.cpp). Per als models GGUF locals, engega un `llama-server` extern; per defecte les eines esperen l'endpoint OpenAI-compatible a `http://localhost:9090/v1`.

#### Instal·lació de llama.cpp

Instal·la una versió precompilada amb Homebrew (macOS/Linux) o Conda (Windows/macOS/Linux). Per a una GPU NVIDIA a Linux o Windows, Conda també ofereix una variant CUDA:

```bash
brew install llama.cpp
conda install -c conda-forge llama.cpp
# NVIDIA/CUDA (Linux o Windows)
conda install -c conda-forge "llama.cpp=*=cuda*"
```

Per compilar-lo des del codi font o configurar CUDA, ROCm o Metal, consulta la [documentació oficial de compilació de llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

Des de l'arrel del repositori, instal·la les dependències Python:

```bash
cd llm
uv sync
```

#### Configuració d'inferència

Els paràmetres comuns es defineixen a `llm/inference.yaml` amb noms OpenAI-compatible:

```yaml
temperature: 0
max_tokens: 256
reasoning_effort: none
```

Cada benchmark pot sobreescriure `max_tokens`. Els adaptadors tradueixen aquesta configuració als camps i valors compatibles amb llama.cpp, OpenAI, OpenRouter o Gemini.

### Execució (LLM)

**1. Descarregar els GGUF necessaris:**

```bash
make llm-download-ggufs GGUF_DIR=models/gguf
make llm-download-ggufs GGUF_DIR=models/gguf GGUF_MODELS="gemma3-12b salamandra-7b"
```

El target llegeix `MODELS` de `llm/run_evals.py`, filtra els models locals GGUF i descarrega els fitxers al subdirectori indicat per `GGUF_DIR` (relatiu a l'arrel del repositori).
També escriu `presets.ini` al mateix directori, amb els ids de model que usa l'avaluador.

**2. Engegar `llama-server`:**

```bash
llama-server --models-preset models/gguf/presets.ini --models-max 1 --port 9090
```

Per defecte les eines fan servir `http://localhost:9090/v1`. Es pot canviar amb `--server-url` o amb `LLAMA_SERVER_URL`.

**3. Executar l'avaluació local:**

```bash
uv run python run_evals.py --models salamandra-7b
uv run python run_evals.py --models gemma3-12b --n-samples 200
uv run python run_evals.py --models gemma3-12b --benchmarks catcola flores
```

Amb un sol `llama.cpp server`, l'orquestrador només permet un model local per execució; selecciona'l amb `--models`.

Si el servidor requereix un identificador de model concret:

```bash
uv run python model.py --model "bartowski/google_gemma-3-12b-it-GGUF:Q8_0" --server-url http://localhost:9090/v1 --server-model gemma-3-12b-it-Q8_0
uv run python run_evals.py --models gemma3-12b --server-model gemma-3-12b-it-Q8_0
```

**Avaluar amb l'API de Google AI o OpenAI:**

```bash
uv run python model.py --model gemini --api-key "LA_TEVA_CLAU" --gemini-model gemini-3.6-flash
OPENAI_API_KEY="LA_TEVA_CLAU" uv run python model.py --model openai --openai-model gpt-4o
```

Els resultats es desen com a JSON a `llm/evals/`.

---

## ASR — Avaluació de models de reconeixement de la parla

L'script `asr/hf-eval.py` mesura la taxa d'error de paraules (WER) i de caràcters (CER) sobre el dataset FLEURS per al català. Suporta models Omnilingual ASR i OpenAI Whisper.

### Instal·lació (ASR)

Requereix [uv](https://docs.astral.sh/uv/).

```bash
cd asr
uv sync
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
uv run python hf-eval.py whisper-small --output evals/results_whisper_small.json
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

- **[Projecte AINA](https://www.projecteaina.cat/)** (Barcelona Supercomputing Center) pels datasets [STS-ca](https://huggingface.co/datasets/projecte-aina/sts-ca), [VilaQuAD](https://huggingface.co/datasets/projecte-aina/vilaquad) i [CaSum](https://huggingface.co/datasets/projecte-aina/casum), que han fet possible l'avaluació de models en català.
- **[nbel](https://huggingface.co/nbel)** pel dataset [CatCoLA](https://huggingface.co/datasets/nbel/CatCoLA), corpus d'acceptabilitat lingüística en català.
- **[Google](https://ai.google/research/)** pel dataset [FLEURS](https://huggingface.co/datasets/google/fleurs) (Few-shot Learning Evaluation of Universal Representations of Speech), usat per avaluar models ASR en català.
- L'equip de **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** pel motor d'avaluació usat en benchmarks generatius.
- **[Meta AI](https://ai.meta.com/)** pel benchmark [FLORES+](https://huggingface.co/datasets/facebook/flores) de traducció automàtica.
