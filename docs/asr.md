# ASR — Avaluació de models de reconeixement de la parla

L'script `asr/hf-eval.py` mesura la taxa d'error de paraules (WER) i de caràcters (CER) sobre FLEURS i OpenSLR-69 en català. Suporta models Omnilingual ASR, Whisper i Gemma 4 amb entrada d'àudio. El script `asr/cloud-eval.py` avalua models d'API d'OpenAI i Gemini sobre els mateixos benchmarks.

### Instal·lació (ASR)

Requereix [uv](https://docs.astral.sh/uv/).

```bash
cd asr
uv sync
```

Per avaluar models Omnilingual ASR, cal instal·lar també el paquet `omnilingual_asr` a l'entorn d'ASR.

### Execució (ASR)

Executa les ordres `make` des de l'arrel del repositori i les ordres Python des de `asr/`. Pots passar opcions a `make asr-eval` amb `ASR_ARGS`.

**Preparar els benchmarks locals una vegada:**

```bash
make asr-prepare
```

Cada script descarrega 400 mostres de fins a 30 segons i genera els àudios i un `manifest.json` a `benchmarks/fleurs_ca_test_400/` o `benchmarks/openslr69_ca_eval_400/`. Els avaluadors executen tots dos benchmarks per defecte; usa `--benchmark fleurs` o `--benchmark openslr69` per seleccionar-ne un. Per a un conjunt propi, usa `--manifest <ruta/manifest.json>`.

**Llistar els models disponibles:**

```bash
uv run python hf-eval.py --list-models
```

**Avaluar un model local:**

```bash
uv run python hf-eval.py whisper-large-v3 --device cuda --output evals/whisper_large_v3.json
uv run python hf-eval.py whisper-small --output evals/whisper_small.json
```

**Avaluar un model d'API:**

```bash
uv run python cloud-eval.py --list-models
OPENAI_API_KEY="LA_TEVA_CLAU" uv run python cloud-eval.py gpt-transcribe --output evals/gpt_transcribe.json
```

Els models de Gemini requereixen la variable `GEMINI_API_KEY`.

**Executar l'orquestrador per a múltiples models:**

```bash
make asr-eval ASR_ARGS="--device cuda --jobs 2"
```

L'orquestrador omet els models que ja tenen els benchmarks sol·licitats i els models d'API sense la clau corresponent. Usa `--overwrite` per repetir les avaluacions. Els resultats es desen a `asr/evals/`; en executar els avaluadors directament, cal indicar `--output` per desar el JSON.

[Torna al README](../README.md)
