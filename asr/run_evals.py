"""
Orchestrator script: runs hf-eval.py for each ASR model, skipping those
whose output JSON already exists.

Usage:
  python run_evals.py
  python run_evals.py --num_samples 500
  python run_evals.py --device cuda
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

MODELS = [
    {
        "label": "whisper-tiny",
        "args": ["whisper-tiny"],
        "output": "evals/results_whisper_tiny.json",
    },
    {
        "label": "whisper-base",
        "args": ["whisper-base"],
        "output": "evals/results_whisper_base.json",
    },
    {
        "label": "whisper-small",
        "args": ["whisper-small"],
        "output": "evals/results_whisper_small.json",
    },
    {
        "label": "whisper-medium",
        "args": ["whisper-medium"],
        "output": "evals/results_whisper_medium.json",
    },
    {
        "label": "whisper-large-v3",
        "args": ["whisper-large-v3"],
        "output": "evals/results_whisper_large_v3.json",
    },
    {
        "label": "whisper-large-v3-turbo",
        "args": ["whisper-large-v3-turbo"],
        "output": "evals/results_whisper_large_v3_turbo.json",
    },
    {
        "label": "whisper-large-v3-ca",
        "args": ["projecte-aina/whisper-large-v3-ca-3catparla"],
        "output": "evals/results_whisper_large_v3_ca.json",
    },
    {
        "label": "omniASR_CTC_300M",
        "args": ["omniASR_CTC_300M"],
        "output": "evals/results_omni_ctc_300m.json",
    },
    {
        "label": "omniASR_CTC_1B",
        "args": ["omniASR_CTC_1B"],
        "output": "evals/results_omni_ctc_1b.json",
    },
    {
        "label": "omniASR_CTC_3B",
        "args": ["omniASR_CTC_3B"],
        "output": "evals/results_omni_ctc_3b.json",
    },
    {
        "label": "omniASR_CTC_7B",
        "args": ["omniASR_CTC_7B"],
        "output": "evals/results_omni_ctc_7b.json",
    },
    {
        "label": "omniASR_LLM_300M",
        "args": ["omniASR_LLM_300M"],
        "output": "evals/results_omni_llm_300m.json",
    },
    {
        "label": "omniASR_LLM_1B",
        "args": ["omniASR_LLM_1B"],
        "output": "evals/results_omni_llm_1b.json",
    },
    {
        "label": "omniASR_LLM_3B",
        "args": ["omniASR_LLM_3B"],
        "output": "evals/results_omni_llm_3b.json",
    },
    {
        "label": "omniASR_LLM_7B",
        "args": ["omniASR_LLM_7B"],
        "output": "evals/results_omni_llm_7b.json",
    },
    {
        "label": "vibevoice",
        "args": ["microsoft/VibeVoice-ASR"],
        "output": "evals/results_vibevoice.json",
    },
    {
        "label": "gemma-4-E4B",
        "args": ["gemma-4-E4B"],
        "output": "evals/results_gemma4_e4b.json",
    },
    {
        "label": "gemma-4-E2B",
        "args": ["gemma-4-E2B"],
        "output": "evals/results_gemma4_e2b.json",
    },
    {
        "label": "gpt-4o-transcribe",
        "script": "cloud-eval.py",
        "args": ["gpt-4o-transcribe"],
        "output": "evals/results_gpt4o_transcribe.json",
        "needs_openai_api_key": True,
    },
    {
        "label": "gemini-3.6-flash",
        "script": "cloud-eval.py",
        "args": ["gemini-3.6-flash"],
        "output": "evals/results_gemini_3_6_flash_asr.json",
        "needs_google_api_key": True,
    },
    {
        "label": "gemini-3-pro-preview",
        "script": "cloud-eval.py",
        "args": ["gemini-3-pro-preview"],
        "output": "evals/results_gemini_3_pro_preview_asr.json",
        "needs_google_api_key": True,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Run ASR evals for all models")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    import os

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    python = sys.executable

    for model in MODELS:
        output_path = SCRIPT_DIR / model["output"]

        if output_path.exists():
            print(f"[SKIP] {model['label']} — {output_path} already exists")
            continue

        if model.get("needs_openai_api_key") and not openai_api_key:
            print(f"[SKIP] {model['label']} — OPENAI_API_KEY env var required but not set")
            continue

        if model.get("needs_google_api_key") and not google_api_key:
            print(f"[SKIP] {model['label']} — GOOGLE_API_KEY env var required but not set")
            continue

        script = model.get("script", "hf-eval.py")
        cmd = [python, "-u", script, *model["args"]]

        if script == "hf-eval.py":
            cmd += ["--device", args.device]

        cmd += [
            "--num_samples",
            str(args.num_samples),
            "--output",
            model["output"],
        ]

        print(f"\n[RUN] {model['label']}: {' '.join(cmd)}\n{'='*60}")
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, stdin=subprocess.DEVNULL)

        if result.returncode != 0:
            print(f"[ERROR] {model['label']} exited with code {result.returncode}")
        else:
            print(f"[DONE] {model['label']} -> {output_path}")
            subprocess.run(
                [python, "-m", "asr.summarize_results"],
                cwd=SCRIPT_DIR.parent,
                stdin=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    main()
