# Embeddings — Avaluació de models d'embeddings

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

Executa les ordres següents des de `embeddings/`.

**Avaluar un model de Hugging Face / Sentence Transformers:**

```bash
uv run python model.py --model "<nom_model>" --output evals/<nom_model>.json
```

**Avaluar amb l'API d'OpenAI o Google:**

Defineix `OPENAI_API_KEY` per a OpenAI, o `GOOGLE_API_KEY` (també s'accepta `GEMINI_API_KEY`) per a Google.

```bash
uv run python model.py --cloud-provider openai --model text-embedding-3-large --output evals/openai_text_embedding_3_large.json
uv run python model.py --cloud-provider google --model gemini-embedding-001 --output evals/google_gemini_embedding_001.json
```

**Executar l'orquestrador per a múltiples models:**

```bash
uv run python run_evals.py
```

Els resultats es desen com a JSON a `embeddings/evals/`.

[Torna al README](../README.md)
