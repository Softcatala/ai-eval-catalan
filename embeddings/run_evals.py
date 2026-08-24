"""Run embedding evals for the top multilingual models supporting Catalan.

Local open-weight models run via sentence-transformers. Cloud models
(OpenAI, Google) run via cloud_models.py and require an API key from env:
  OPENAI_API_KEY for OpenAI
  GOOGLE_API_KEY (or GEMINI_API_KEY) for Google
"""

import os
import subprocess
import sys
from pathlib import Path

MODELS = [
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "extra": [],
    },
    {
        "name": "multilingual-e5-small",
        "id": "intfloat/multilingual-e5-small",
        "extra": ["--query-prefix", "query: ", "--doc-prefix", "passage: "],
    },
    {
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "extra": [],
    },
    {
        "name": "multilingual-e5-base",
        "id": "intfloat/multilingual-e5-base",
        "extra": ["--query-prefix", "query: ", "--doc-prefix", "passage: "],
    },
    {
        "name": "gte-multilingual-base",
        "id": "Alibaba-NLP/gte-multilingual-base",
        "extra": ["--trust-remote-code"],
    },
    {
        "name": "distiluse-base-multilingual-cased-v2",
        "id": "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "extra": [],
    },
    {
        "name": "LaBSE",
        "id": "sentence-transformers/LaBSE",
        "extra": [],
    },
    {
        "name": "snowflake-arctic-embed-l-v2.0",
        "id": "Snowflake/snowflake-arctic-embed-l-v2.0",
        # Model ships "query: " as its only documented prefix; reuse on both
        # sides of STS-ca (it's pure-retrieval-trained, no STS-specific prompt).
        "extra": [
            "--trust-remote-code",
            "--batch-size",
            "8",
            "--sts-prefix",
            "query: ",
        ],
    },
    {
        "name": "nomic-embed-text-v2-moe",
        "id": "nomic-ai/nomic-embed-text-v2-moe",
        # config_sentence_transformers.json ships "classification: " for STS.
        "extra": [
            "--trust-remote-code",
            "--query-prefix",
            "search_query: ",
            "--doc-prefix",
            "search_document: ",
            "--sts-prefix",
            "classification: ",
            "--batch-size",
            "8",
        ],
    },
    {
        "name": "snowflake-arctic-embed-m-v2.0",
        "id": "Snowflake/snowflake-arctic-embed-m-v2.0",
        "extra": ["--trust-remote-code", "--batch-size", "8"],
    },
    {
        "name": "multilingual-e5-large",
        "id": "intfloat/multilingual-e5-large",
        "extra": ["--query-prefix", "query: ", "--doc-prefix", "passage: "],
    },
    {
        "name": "bge-m3",
        "id": "BAAI/bge-m3",
        "extra": ["--batch-size", "8"],
    },
    {
        "name": "jina-embeddings-v3",
        "id": "jinaai/jina-embeddings-v3",
        "extra": ["--trust-remote-code", "--batch-size", "8"],
    },
    # --- Cloud models (require API keys from env). Latest as of 2026-06. ---
    {
        "name": "openai-text-embedding-3-large",
        "id": "text-embedding-3-large",
        "cloud": True,
        "needs_openai_api_key": True,
        "extra": ["--cloud-provider", "openai", "--batch-size", "64"],
    },
    {
        "name": "openai-text-embedding-3-small",
        "id": "text-embedding-3-small",
        "cloud": True,
        "needs_openai_api_key": True,
        "extra": ["--cloud-provider", "openai", "--batch-size", "64"],
    },
    {
        "name": "google-gemini-embedding-001",
        "id": "gemini-embedding-001",
        "cloud": True,
        "needs_google_api_key": True,
        # Gemini API allows up to 100 inputs per embed_content call.
        "extra": ["--cloud-provider", "google", "--batch-size", "32"],
    },
]

SCRIPT_DIR = Path(__file__).parent

FORCE = "--force" in sys.argv
INCLUDE_CLOUD = "--include-cloud" in sys.argv

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

for m in MODELS:
    out = (
        SCRIPT_DIR
        / "evals"
        / f"results_{m['name'].replace('-', '_').replace('.', '_')}.json"
    )

    # Gate cloud models behind an explicit opt-in and key presence.
    if m.get("cloud"):
        if not INCLUDE_CLOUD:
            print(f"[SKIP] {m['name']} (cloud — pass --include-cloud to run)")
            continue
        if m.get("needs_openai_api_key") and not OPENAI_KEY:
            print(f"[SKIP] {m['name']} (OPENAI_API_KEY not set)")
            continue
        if m.get("needs_google_api_key") and not GOOGLE_KEY:
            print(f"[SKIP] {m['name']} (GOOGLE_API_KEY / GEMINI_API_KEY not set)")
            continue

    if out.exists() and not FORCE:
        # Skip only if all required benchmarks are already present.
        try:
            import json as _json

            data = _json.loads(out.read_text())
            bench = data.get("benchmarks", {})
            needed = {"xquad_ca_retrieval", "sts_ca", "tecla_classification"}
            if needed.issubset(bench.keys()):
                print(f"[SKIP] {m['name']}")
                continue
        except Exception:
            pass
    cmd = [
        sys.executable,
        "-u",
        "model.py",
        "--model",
        m["id"],
        "--output",
        str(out),
        "--display-name",
        m["name"],
        *m["extra"],
    ]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=False)
