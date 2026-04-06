# Eines d'avaluació de models LLM i ASR

Aquest repositori conté eines per avaluar les capacitats de models de llenguatge gran (LLM) i de reconeixement automàtic de la parla (ASR), amb focus especial en la llengua catalana.
Els resultats estan compartits a https://www.softcatala.org/models-dintelligencia-artificial-en-catala-per-usar-en-local/

## Estructura del projecte

```
model-eval/
├── llm/          # Avaluació de models LLM
│   ├── model.py          # Pipeline d'avaluació per a un model
│   ├── run_evals.py      # Orquestrador per executar múltiples models
│   └── summarize_results.py
└── asr/          # Avaluació de models ASR
    └── hf-eval.py        # Avaluació de WER/CER sobre FLEURS
```

---

## LLM — Avaluació de models de llenguatge

El pipeline `llm/model.py` avalua models GGUF (via `llama-server`) i models de l'API de Google AI (Gemini/Gemma) sobre benchmarks de català:

| Benchmark | Tasca | Mètrica |
|-----------|-------|---------|
| **VeritasQA** | Preguntes obertes en català | Accuracy |
| **STS-ca** | Similitud semàntica de frases | Correlació de Pearson |
| **CatCoLA** | Acceptabilitat gramatical | MCC |
| **CLUB / VilaQuAD** | Comprensió lectora i QA | Exact Match |
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

**Avaluar amb l'API de Google AI (Gemma/Gemini):**

```bash
uv run python model.py --model gemini --api-key "LA_TEVA_CLAU" --gemini-model gemma-3-27b-it
```

**Avaluar benchmarks específics:**

```bash
uv run python model.py --model "bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0" --benchmarks catcola flores
```

**Executar tots els models de la llista (orquestrador):**

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

---

## Agraïments

Volem expressar el nostre agraïment als proveïdors dels datasets usats en l'avaluació:

- **[Projecte AINA](https://www.projecteaina.cat/)** (Barcelona Supercomputing Center) pels datasets [VeritasQA](https://huggingface.co/datasets/projecte-aina/veritasQA), [STS-ca](https://huggingface.co/datasets/projecte-aina/sts-ca), [VilaQuAD](https://huggingface.co/datasets/projecte-aina/vilaquad) i [CaSum](https://huggingface.co/datasets/projecte-aina/casum), que han fet possible l'avaluació de models en català.
- **[nbel](https://huggingface.co/nbel)** pel dataset [CatCoLA](https://huggingface.co/datasets/nbel/CatCoLA), corpus d'acceptabilitat lingüística en català.
- **[Google](https://ai.google/research/)** pel dataset [FLEURS](https://huggingface.co/datasets/google/fleurs) (Few-shot Learning Evaluation of Universal Representations of Speech), usat per avaluar models ASR en català.
- **[IberBench](https://github.com/iberbench)** i l'equip de **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** pels benchmarks de tasques NLP per a l'espanyol i el català.
- **[Meta AI](https://ai.meta.com/)** pel benchmark [FLORES+](https://huggingface.co/datasets/facebook/flores) de traducció automàtica.
