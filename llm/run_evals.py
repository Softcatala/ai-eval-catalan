"""
Orchestrator script: runs model.py for each model, skipping those
whose output JSON already exists.

Models:
  - google/gemma-3-1b-it -> results_gemma3_1b.json

Usage:
  python run_evals.py --models gemma3-12b
  python run_evals.py --models gemma3-12b --n-samples 200
  python run_evals.py --models gemma3-12b --benchmarks catcola flores
  python run_evals.py --server-url http://localhost:9090/v1 --models gemma3-12b
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from models_config import DEFAULT_LOCAL_SERVER_URL, MODELS

SCRIPT_DIR = Path(__file__).parent


def _llama_server_url_from_env() -> str | None:
    url = os.environ.get("LLAMA_SERVER_URL")
    if url:
        return url.rstrip("/")

    return DEFAULT_LOCAL_SERVER_URL


def main():
    parser = argparse.ArgumentParser(description="Run evals for configured models")
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        choices=[
            "sts_ca",
            "catcola",
            "club",
            "casum",
            "flores",
            "ifeval",
            "all",
        ],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional display-name subset from MODELS (e.g. gemma3-12b)",
    )
    parser.add_argument(
        "--llama-server-url",
        "--server-url",
        dest="llama_server_url",
        default=_llama_server_url_from_env(),
        help=(
            "Reuse an existing llama-server/OpenAI-compatible base URL for local "
            "GGUF models. Also configurable with LLAMA_SERVER_URL."
        ),
    )
    parser.add_argument(
        "--llama-server-model",
        "--server-model",
        dest="llama_server_model",
        default=None,
        help=(
            "Request model id for the existing server. Only valid with one selected "
            "local GGUF model."
        ),
    )
    args = parser.parse_args()

    selected = set(args.models) if args.models else None
    if selected:
        known = {model["display_name"] for model in MODELS}
        unknown = sorted(selected - known)
        if unknown:
            parser.error(f"unknown model display name(s): {', '.join(unknown)}")
        models = [model for model in MODELS if model["display_name"] in selected]
    else:
        models = MODELS

    local_models = [model for model in models if not model.get("cloud")]
    if args.llama_server_url and len(local_models) > 1:
        parser.error(
            "using one local llama-server requires exactly one selected local model; "
            "pass --models <display_name>"
        )
    if args.llama_server_model and len(local_models) != 1:
        parser.error(
            "--llama-server-model requires exactly one selected local GGUF model"
        )

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    bedrock_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")

    python = sys.executable

    for model in models:
        output_path = SCRIPT_DIR / model["output"]
        name = model["display_name"]

        if output_path.exists():
            print(f"[SKIP] {name} — {output_path} already exists")
            continue

        if model.get("needs_api_key") and not google_api_key:
            print(f"[SKIP] {name} — GOOGLE_API_KEY env var required but not set")
            continue

        if model.get("needs_openai_api_key") and not openai_api_key:
            print(f"[SKIP] {name} — OPENAI_API_KEY env var required but not set")
            continue

        if model.get("needs_bedrock_token") and not bedrock_token:
            print(f"[SKIP] {name} — AWS_BEARER_TOKEN_BEDROCK env var required but not set")
            continue

        cmd = [
            python,
            "-u",
            "model.py",
            *model["args"],
            "--output",
            model["output"],
            "--n-samples",
            str(args.n_samples),
            "--benchmarks",
            *args.benchmarks,
        ]

        if args.llama_server_url and not model.get("cloud"):
            cmd += ["--llama-server-url", args.llama_server_url.rstrip("/")]
            llama_server_model = args.llama_server_model or model.get(
                "llama_server_model"
            )
            if llama_server_model:
                cmd += ["--llama-server-model", llama_server_model]

        if model.get("params_b") is not None:
            cmd += ["--params-b", str(model["params_b"])]

        cmd += ["--display-name", name]

        if model.get("cloud"):
            cmd += ["--cloud"]

        if model.get("quantized_analysis_only"):
            cmd += ["--quantized-analysis"]

        if model.get("needs_api_key"):
            cmd += ["--api-key", google_api_key]

        print(f"\n[RUN] {name}: {' '.join(cmd)}\n{'='*60}")
        run_env = os.environ.copy()
        if model.get("needs_bedrock_token"):
            run_env["OPENAI_API_KEY"] = bedrock_token
        result = subprocess.run(
            cmd, cwd=SCRIPT_DIR, stdin=subprocess.DEVNULL, env=run_env
        )

        if result.returncode != 0:
            print(f"[ERROR] {name} exited with code {result.returncode}")
        else:
            print(f"[DONE] {name} -> {output_path}")


if __name__ == "__main__":
    main()
