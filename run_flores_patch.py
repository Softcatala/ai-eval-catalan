"""
Run flores-only eval for models that have an error in their flores results,
then merge the new flores results back into the existing JSON files.
"""
import json
import subprocess
import sys
from pathlib import Path

MODELS = [
    {
        "label": "gemma3-12b",
        "output": "results_gemma3_12b.json",
        "args": ["--model", "bartowski/google_gemma-3-12b-it-GGUF:Q8_0", "--device", "cuda"],
    },
    {
        "label": "mistral-small-24b",
        "output": "results_mistral_small_24b.json",
        "args": ["--model", "bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q4_K_M", "--device", "cuda"],
    },
    {
        "label": "gpt-oss-20b",
        "output": "results_gpt_oss_20b.json",
        "args": ["--model", "bartowski/openai_gpt-oss-20b-GGUF:Q4_K_M", "--device", "cuda"],
    },
    {
        "label": "gemma4-e4b",
        "output": "results_gemma4_e4b.json",
        "args": ["--model", "bartowski/google_gemma-4-E4B-it-GGUF:Q8_0", "--device", "cuda"],
        # Gemma-4 E4B emits thinking tokens which corrupt the Flores completion output.
        # model.py now auto-passes --reasoning off for this model via _is_thinking_model().
        "comet_threshold": 0.05,
    },
]

BASE_PORT = 8090
python = sys.executable

for model in MODELS:
    label = model["label"]
    output = model["output"]

    # Check if FLORES already has real, non-zero COMET results.
    existing = json.load(open(output))
    flores = existing.get("benchmarks", {}).get("flores", {})
    comet_threshold = model.get("comet_threshold", 0.01)
    if "error" not in flores and flores:
        en_ca = flores.get("catalan_bench_flores_en-ca", {})
        comet = en_ca.get("comet,none", 0)
        if comet >= comet_threshold:
            print(f"[SKIP] {label} — FLORES already has valid results (COMET={comet:.4f})")
            continue
        print(f"[RERUN] {label} — FLORES COMET={comet:.4f} is below threshold ({comet_threshold}), re-running")

    tmp_output = f"results_{label}_flores_tmp.json"
    cmd = [
        python, "-u", "model.py",
        *model["args"],
        "--benchmarks", "flores",
        "--n-samples", "400",
        "--output", tmp_output,
        "--llama-server-port", str(BASE_PORT),
    ]
    print(f"\n[RUN] {label}: {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[ERROR] {label} exited with code {result.returncode}")
        continue

    # Merge flores results back
    tmp_path = Path(tmp_output)
    if not tmp_path.exists():
        print(f"[ERROR] {tmp_output} not found after run")
        continue

    tmp_data = json.load(open(tmp_output))
    new_flores = tmp_data.get("benchmarks", {}).get("flores", {})
    if not new_flores or "error" in new_flores:
        print(f"[ERROR] {label} flores still has error: {new_flores}")
    else:
        existing["benchmarks"]["flores"] = new_flores
        with open(output, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"[DONE] {label} — flores merged into {output}")

    tmp_path.unlink(missing_ok=True)

print("\nAll done.")
