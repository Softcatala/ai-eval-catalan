"""
Orchestrator script: runs hf-eval.py for each ASR model, skipping those
whose output JSON already exists.

Usage:
  python run_evals.py
  python run_evals.py --device cuda
  python run_evals.py --device cuda --jobs 2
  python run_evals.py --overwrite
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

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
        "args": ["microsoft/VibeVoice-ASR-HF"],
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
        "label": "gemini-3.8-flash",
        "script": "cloud-eval.py",
        "args": ["gemini-3.8-flash"],
        "output": "evals/results_gemini_3_8_flash_asr.json",
        "needs_google_api_key": True,
    },
    {
        "label": "gemini-3.1-pro-preview",
        "script": "cloud-eval.py",
        "args": ["gemini-3.1-pro-preview"],
        "output": "evals/results_gemini_3_pro_preview_asr.json",
        "needs_google_api_key": True,
    },
]


def run_model(model, device, gpu_id=None):
    output_path = SCRIPT_DIR / model["output"]
    script = model.get("script", "hf-eval.py")
    cmd = [sys.executable, "-u", script, *model["args"]]
    env = os.environ.copy()

    if script == "hf-eval.py":
        cmd += ["--device", device]
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd += ["--output", model["output"]]
    location = f" on GPU {gpu_id}" if gpu_id is not None else ""
    print(f"\n[RUN]{location} {model['label']}: {' '.join(cmd)}\n{'=' * 60}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, stdin=subprocess.DEVNULL, env=env)

    if result.returncode != 0:
        print(f"[ERROR] {model['label']} exited with code {result.returncode}")
        return

    print(f"[DONE] {model['label']} -> {output_path}")
    subprocess.run(
        [sys.executable, "-m", "asr.summarize_results"],
        cwd=SCRIPT_DIR.parent,
        stdin=subprocess.DEVNULL,
    )


def main():
    parser = argparse.ArgumentParser(description="Run ASR evals for all models")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Maximum concurrent local evaluations (default: 2)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run models whose result files already exist",
    )
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    local_models = []
    cloud_models = []

    for model in MODELS:
        output_path = SCRIPT_DIR / model["output"]

        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] {model['label']} — {output_path} already exists")
            continue

        if model.get("needs_openai_api_key") and not openai_api_key:
            print(
                f"[SKIP] {model['label']} — OPENAI_API_KEY env var required but not set"
            )
            continue

        if model.get("needs_google_api_key") and not gemini_api_key:
            print(
                f"[SKIP] {model['label']} — GEMINI_API_KEY env var required but not set"
            )
            continue

        if model.get("script", "hf-eval.py") == "cloud-eval.py":
            cloud_models.append(model)
        else:
            local_models.append(model)

    if args.device == "cuda":
        import torch

        gpu_count = torch.cuda.device_count()
        if not gpu_count:
            parser.error("--device cuda requested but no CUDA GPUs are available")
        local_workers = min(args.jobs, gpu_count)
        gpu_ids = range(local_workers)
    else:
        local_workers = args.jobs
        gpu_ids = [None] * local_workers

    local_queue = Queue()
    for model in local_models:
        local_queue.put(model)

    def run_local_queue(gpu_id):
        while True:
            try:
                model = local_queue.get_nowait()
            except Empty:
                return
            run_model(model, args.device, gpu_id)

    def run_cloud_queue():
        for model in cloud_models:
            run_model(model, args.device)

    with ThreadPoolExecutor(max_workers=local_workers + bool(cloud_models)) as executor:
        futures = [executor.submit(run_local_queue, gpu_id) for gpu_id in gpu_ids]
        if cloud_models:
            futures.append(executor.submit(run_cloud_queue))
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
