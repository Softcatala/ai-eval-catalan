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

SCRIPT_DIR = Path(__file__).parent

# Models sized to fit in Tesla T4 (15 GB VRAM). Run sequentially — only 7 GB host RAM available.
# 24B Q8 (~24 GB) exceeds VRAM; using Q4_K_M (~13 GB) instead.
MODELS = [
    {
        "display_name": "gemma3-4b-q2",
        "output": "evals/results_gemma3_4b_q2.json",
        "args": [
            "--model",
            "bartowski/google_gemma-3-4b-it-GGUF:Q2_K",
            "--device",
            "cuda",
        ],
        "ram_gb": 2,
        "params_b": 4.0,
        "quantization": "q2",
        "quantized_analysis_only": True,
    },
    {
        "display_name": "gemma3-4b-q4",
        "output": "evals/results_gemma3_4b_q4.json",
        "args": [
            "--model",
            "bartowski/google_gemma-3-4b-it-GGUF:Q4_K_M",
            "--device",
            "cuda",
        ],
        "ram_gb": 3,
        "params_b": 4.0,
        "quantization": "q4",
        "quantized_analysis_only": True,
    },
    {
        "display_name": "gemma3-4b",
        "output": "evals/results_gemma3_4b_q8.json",
        "args": [
            "--model",
            "bartowski/google_gemma-3-4b-it-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 5,
        "params_b": 4.0,
        "quantization": "q8",
    },
    {
        "display_name": "gemma3-12b-q2",
        "output": "evals/results_gemma3_12b_q2.json",
        "args": [
            "--model",
            "bartowski/google_gemma-3-12b-it-GGUF:Q2_K",
            "--device",
            "cuda",
        ],
        "ram_gb": 4,
        "params_b": 12.0,
        "quantization": "q2",
        "quantized_analysis_only": True,
    },
    {
        "display_name": "gemma3-12b-q4",
        "output": "evals/results_gemma3_12b_q4.json",
        "args": [
            "--model",
            "bartowski/google_gemma-3-12b-it-GGUF:Q4_K_M",
            "--device",
            "cuda",
        ],
        "ram_gb": 7,
        "params_b": 12.0,
        "quantization": "q4",
        "quantized_analysis_only": True,
    },
    {
        "display_name": "gemma3-12b",
        "output": "evals/results_gemma3_12b.json",
        "args": [
            "--model",
            "bartowski/google_gemma-3-12b-it-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 12,
        "params_b": 12.0,
        "quantization": "q8",
    },
    {
        "display_name": "gemma3-27b-q2",
        "output": "evals/results_gemma3_27b_q2.json",
        "args": [
            "--model",
            "bartowski/google_gemma-3-27b-it-GGUF:Q2_K",
            "--device",
            "cuda",
        ],
        "ram_gb": 9,
        "params_b": 27.0,
        "quantization": "q2",
        "quantized_analysis_only": True,
    },
    {
        "display_name": "gemma3-27b-q4",
        "output": "evals/results_gemma3_27b_q4.json",
        "args": [
            "--model",
            "bartowski/google_gemma-3-27b-it-GGUF:Q4_K_M",
            "--device",
            "cuda",
        ],
        "ram_gb": 15,
        "params_b": 27.0,
        "quantization": "q4",
        "quantized_analysis_only": True,
    },
    {
        "display_name": "gemma3-27b",
        "output": "evals/results_gemma3_27b.json",
        "args": [
            "--model",
            "bartowski/google_gemma-3-27b-it-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 27,
        "params_b": 27.0,
        "quantization": "q8",
    },
    {
        "display_name": "mistral-small-24b",
        "output": "evals/results_mistral_small_24b.json",
        "args": [
            "--model",
            "bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 25,
        "params_b": 24.0,
        "quantization": "q8",
    },
    {
        "display_name": "qwen3-14b",
        "output": "evals/results_qwen3_14b.json",
        "args": ["--model", "bartowski/Qwen_Qwen3-14B-GGUF:Q8_0", "--device", "cuda"],
        "ram_gb": 14,
        "params_b": 14.0,
        "quantization": "q8",
    },
    {
        "display_name": "phi-4",
        "output": "evals/results_phi4_q8.json",
        "args": [
            "--model",
            "bartowski/phi-4-GGUF:Q8_0",
        ],
        "external_llama_server": True,
        "llama_server_model": "phi-4-Q8_0",
        "ram_gb": 15,
        "params_b": 14.0,
        "quantization": "q8",
    },
    {
        "display_name": "qwen3.5-9b",
        "output": "evals/results_qwen3.5_9b.json",
        "args": ["--model", "bartowski/Qwen_Qwen3.5-9B-GGUF:Q8_0", "--device", "cuda"],
        "ram_gb": 9,
        "params_b": 9.0,
        "quantization": "q8",
    },
    {
        "display_name": "qwen3.6-27b",
        "output": "evals/results_qwen3.6_27b.json",
        "args": ["--model", "unsloth/Qwen3.6-27B-GGUF:Q8_0", "--device", "cuda"],
        "ram_gb": 27,
        "params_b": 27.0,
        "quantization": "q8",
    },
    {
        "display_name": "llama3.1-8b",
        "output": "evals/results_llama3.1_8b.json",
        "args": [
            "--model",
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 8,
        "params_b": 8.0,
        "quantization": "q8",
    },
    {
        "display_name": "aya-expanse-8b",
        "output": "evals/results_aya_expanse_8b.json",
        "args": [
            "--model",
            "bartowski/aya-expanse-8b-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 8,
        "params_b": 8.0,
        "quantization": "q8",
    },
    {
        "display_name": "eurollm-9b",
        "output": "evals/results_eurollm_9b.json",
        "args": [
            "--model",
            "bartowski/EuroLLM-9B-Instruct-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 9,
        "params_b": 9.0,
        "quantization": "q8",
    },
    {
        "display_name": "salamandra-7b",
        "output": "evals/results_salamandra_7b.json",
        "args": [
            "--model",
            "mradermacher/salamandra-7b-instruct-2606-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 7,
        "params_b": 7.0,
        "quantization": "q8",
    },
    {
        "display_name": "gemma4-12b-q4",
        "output": "evals/results_gemma4_12b_q4.json",
        "args": [
            "--model",
            "unsloth/gemma-4-12b-it-GGUF:Q4_K_M",
            "--device",
            "cuda",
        ],
        "ram_gb": 7,
        "params_b": 12.0,
        "quantization": "q4",
        "quantized_analysis_only": True,
    },
    {
        "display_name": "gemma4-12b",
        "output": "evals/results_gemma4_12b.json",
        "args": [
            "--model",
            "unsloth/gemma-4-12b-it-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 13,
        "params_b": 12.0,
        "quantization": "q8",
    },
    {
        "display_name": "gemma4-e4b-q4",
        "output": "evals/results_gemma4_e4b_q4.json",
        "args": [
            "--model",
            "bartowski/google_gemma-4-E4B-it-GGUF:Q4_K_M",
            "--device",
            "cuda",
        ],
        "ram_gb": 3,
        "params_b": 4.0,
        "quantization": "q4",
        "quantized_analysis_only": True,
    },
    {
        "display_name": "gemma4-e4b",
        "output": "evals/results_gemma4_e4b.json",
        "args": [
            "--model",
            "bartowski/google_gemma-4-E4B-it-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 5,
        "params_b": 4.0,
        "quantization": "q8",
    },
    {
        "display_name": "gemma4-26b-q4",
        "output": "evals/results_gemma4_26b_q4.json",
        "args": [
            "--model",
            "bartowski/google_gemma-4-26B-A4B-it-GGUF:Q4_K_M",
            "--device",
            "cuda",
        ],
        "ram_gb": 14,
        "params_b": 26.0,
        "quantization": "q4",
        "quantized_analysis_only": True,
    },
    {
        "display_name": "gemma4-26b",
        "output": "evals/results_gemma4_26b_q8.json",
        "args": [
            "--model",
            "bartowski/google_gemma-4-26B-A4B-it-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 26,
        "params_b": 26.0,
        "quantization": "q8",
    },
    {
        "display_name": "gemini-3-1-preview",
        "output": "evals/results_gemini_3_1_preview.json",
        "args": [
            "--model",
            "gemini",
            "--gemini-model",
            "gemini-3.1-pro-preview",
        ],
        "cloud": True,
        "needs_api_key": True,
        "ram_gb": 0,
        "params_b": None,
        "quantization": "",
    },
    {
        "display_name": "gemini-3-6-flash",
        "output": "evals/results_gemini_3_6_flash.json",
        "args": [
            "--model",
            "gemini",
            "--gemini-model",
            "gemini-3.6-flash",
        ],
        "cloud": True,
        "needs_api_key": True,
        "ram_gb": 0,
        "params_b": None,
        "quantization": "",
    },
    {
        "display_name": "gpt-5.4-mini",
        "output": "evals/results_gpt_5_4_mini.json",
        "args": [
            "--model",
            "openai",
            "--openai-model",
            "gpt-5.4-mini",
        ],
        "cloud": True,
        "needs_openai_api_key": True,
        "ram_gb": 0,
        "params_b": None,
        "quantization": "",
    },
    {
        "display_name": "gpt-5.6",
        "output": "evals/results_gpt_5_6.json",
        "args": [
            "--model",
            "openai",
            "--openai-model",
            "gpt-5.6",
        ],
        "cloud": True,
        "needs_openai_api_key": True,
        "ram_gb": 0,
        "params_b": None,
        "quantization": "",
    },
    {
        "display_name": "claude-opus-4-7",
        "output": "evals/results_claude_opus_4_7.json",
        "args": [
            "--model",
            "claude",
            "--openai-model",
            "anthropic/claude-opus-4-7",
        ],
        "cloud": True,
        "needs_openrouter_api_key": True,
        "ram_gb": 0,
        "params_b": None,
        "quantization": "",
    },
]

# Base port for llama-server (8080 is taken by Jupyter)
DEFAULT_BASE_PORT = 8090
DEFAULT_LOCAL_SERVER_URL = "http://localhost:9090/v1"


def _llama_server_url_from_env() -> str | None:
    url = os.environ.get("LLAMA_SERVER_URL")
    if url:
        return url.rstrip("/")

    port = os.environ.get("LLAMA_SERVER_PORT") or os.environ.get("LLAMA_CPP_PORT")
    if port:
        return f"http://127.0.0.1:{port}/v1"

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
            "GGUF models instead of starting one per run. Also configurable with "
            "LLAMA_SERVER_URL, LLAMA_SERVER_PORT, or LLAMA_CPP_PORT."
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
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    llama_server_port = (
        os.environ.get("LLAMA_SERVER_PORT")
        or os.environ.get("LLAMA_CPP_PORT")
        or str(DEFAULT_BASE_PORT)
    )

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

        if model.get("needs_openrouter_api_key") and not openrouter_api_key:
            print(f"[SKIP] {name} — OPENROUTER_API_KEY env var required but not set")
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
            "--llama-server-port",
            str(llama_server_port),
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

        if model.get("needs_openrouter_api_key"):
            cmd += ["--api-key", openrouter_api_key]

        print(f"\n[RUN] {name}: {' '.join(cmd)}\n{'='*60}")
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, stdin=subprocess.DEVNULL)

        if result.returncode != 0:
            print(f"[ERROR] {name} exited with code {result.returncode}")
        else:
            print(f"[DONE] {name} -> {output_path}")


if __name__ == "__main__":
    main()
