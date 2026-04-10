"""
Orchestrator script: runs model.py for each model, skipping those
whose output JSON already exists.

Models:
  - google/gemma-3-1b-it -> results_gemma3_1b.json

Usage:
  python run_evals.py
  python run_evals.py --n-samples 200
  python run_evals.py --benchmarks catcola flores
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
        "label": "gemma3-12b",
        "output": "evals/results_gemma3_12b.json",
        "args": [
            "--model",
            "bartowski/google_gemma-3-12b-it-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 12,
    },
    {
        "label": "mistral-small-24b",
        "output": "evals/results_mistral_small_24b.json",
        "args": [
            "--model",
            "bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q4_K_M",
            "--device",
            "cuda",
        ],
        "ram_gb": 13,
    },
    {
        "label": "gpt-oss-20b",
        "output": "evals/results_gpt_oss_20b.json",
        "args": [
            "--model",
            "bartowski/openai_gpt-oss-20b-GGUF:Q4_K_M",
            "--device",
            "cuda",
        ],
        "ram_gb": 11,
    },
    {
        "label": "qwen3-14b",
        "output": "evals/results_qwen3_14b.json",
        "args": ["--model", "bartowski/Qwen_Qwen3-14B-GGUF:Q8_0", "--device", "cuda"],
        "ram_gb": 14,
    },
    {
        "label": "qwen3.5-9b",
        "output": "evals/results_qwen3.5_9b.json",
        "args": ["--model", "bartowski/Qwen_Qwen3.5-9B-GGUF:Q8_0", "--device", "cuda"],
        "ram_gb": 9,
    },
    {
        "label": "llama3.1-8b",
        "output": "evals/results_llama3.1_8b.json",
        "args": [
            "--model",
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 8,
    },
    {
        "label": "aya-expanse-8b",
        "output": "evals/results_aya_expanse_8b.json",
        "args": [
            "--model",
            "bartowski/aya-expanse-8b-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 8,
    },
    {
        "label": "eurollm-9b",
        "output": "evals/results_eurollm_9b.json",
        "args": [
            "--model",
            "bartowski/EuroLLM-9B-Instruct-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 9,
    },
    {
        "label": "gemma4-e4b",
        "output": "evals/results_gemma4_e4b.json",
        "args": [
            "--model",
            "bartowski/google_gemma-4-E4B-it-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 5,
    },
    {
        "label": "gemma4-26b-q4",
        "output": "evals/results_gemma4_26b_q4.json",
        "args": [
            "--model",
            "bartowski/google_gemma-4-26B-A4B-it-GGUF:Q4_K_M",
            "--device",
            "cuda",
        ],
        "ram_gb": 14,
    },
    {
        "label": "gemma4-26b-q8",
        "output": "evals/results_gemma4_26b_q8.json",
        "args": [
            "--model",
            "bartowski/google_gemma-4-26B-A4B-it-GGUF:Q8_0",
            "--device",
            "cuda",
        ],
        "ram_gb": 26,
    },
    {
        "label": "gemini-3-1-preview",
        "output": "evals/results_gemini_3_1_preview.json",
        "args": [
            "--model",
            "gemini",
            "--gemini-model",
            "gemini-3.1-pro-preview",
        ],
        "needs_api_key": True,
        "ram_gb": 0,
    },
    {
        "label": "gemini-3-flash-preview",
        "output": "evals/results_gemini_3_flash_preview.json",
        "args": [
            "--model",
            "gemini",
            "--gemini-model",
            "gemini-3-flash-preview",
        ],
        "needs_api_key": True,
        "ram_gb": 0,
    },
    {
        "label": "gpt-5.4",
        "output": "evals/results_gpt_5_4.json",
        "args": [
            "--model",
            "openai",
            "--openai-model",
            "gpt-5.4",
        ],
        "needs_openai_api_key": True,
        "ram_gb": 0,
    },
]

# Base port for llama-server (8080 is taken by Jupyter)
BASE_PORT = 8090


def main():
    parser = argparse.ArgumentParser(description="Run evals for all models")
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        choices=[
            "veritasqa",
            "sts_ca",
            "catcola",
            "club",
            "casum",
            "iberbench",
            "flores",
            "all",
        ],
    )
    args = parser.parse_args()

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    python = sys.executable

    for model in MODELS:
        output_path = Path(model["output"])

        if output_path.exists():
            print(f"[SKIP] {model['label']} — {output_path} already exists")
            continue

        if model.get("needs_api_key") and not google_api_key:
            print(f"[SKIP] {model['label']} — GOOGLE_API_KEY env var required but not set")
            continue

        if model.get("needs_openai_api_key") and not openai_api_key:
            print(f"[SKIP] {model['label']} — OPENAI_API_KEY env var required but not set")
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
            str(BASE_PORT),
        ]

        if model.get("needs_api_key"):
            cmd += ["--api-key", google_api_key]

        print(f"\n[RUN] {model['label']}: {' '.join(cmd)}\n{'='*60}")
        result = subprocess.run(cmd, cwd=SCRIPT_DIR)

        if result.returncode != 0:
            print(f"[ERROR] {model['label']} exited with code {result.returncode}")
        else:
            print(f"[DONE] {model['label']} -> {output_path}")


if __name__ == "__main__":
    main()
