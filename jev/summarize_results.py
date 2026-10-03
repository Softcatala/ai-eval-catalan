"""Aggregate completed MASSIVE evaluations into jevs.json."""

import argparse
import json
from pathlib import Path

from eval_common.model_urls import repo_url

SCRIPT_DIR = Path(__file__).resolve().parent

# Parameter counts from the GGUF tensor shapes, in billions.
MODEL_PARAMS_B = {
    "julia-1": 0.144192769,
    "laya": 0.421029889,
    "kev-4b": 4.207062528,
    "lev": 4.205751296,
    "openjev": 26.895998464,
}

# Measured GGUF sizes in decimal GB for the evaluated quantizations.
MODEL_MEMORY_GB = {
    "julia-1": ("Q8_0", 0.168166496),
    "laya": ("Q8_0", 0.449397600),
    "kev-4b": ("Q4_K_M", 3.033489824),
    "lev": ("Q4_K_M", 3.011777440),
    "openjev": ("Q4_K_M", 18.973872288),
}


def model_memory_gb(result):
    if result.get("memory_gb") is not None:
        return result["memory_gb"]
    spec = result["model"].split("/")[-1]
    name, _, quant = spec.partition(":")
    known_quant, size = MODEL_MEMORY_GB.get(
        name.lower().removesuffix("-gguf"), (None, None)
    )
    return (
        round(size, 1)
        if size is not None and (not quant or quant == known_quant)
        else None
    )


def model_params_b(result):
    if result.get("params_b") is not None:
        return result["params_b"]
    name = result["model"].split("/")[-1].split(":")[0].lower()
    return MODEL_PARAMS_B.get(name.removesuffix("-gguf"))


def model_repo_url(result):
    try:
        return repo_url(result["model"])
    except ValueError:
        return None


def load_rows(directory):
    rows = []
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith((".summary.json", ".comparison.json")):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if "accuracy" not in result or "model" not in result:
            continue
        rows.append(
            {
                "params_b": model_params_b(result),
                "repo_url": model_repo_url(result),
                "memory_gb": model_memory_gb(result),
                "decisions_per_sec": 1000 / result["mean_latency_ms"]
                if result["mean_latency_ms"] > 0
                else None,
                **{
                    key: result[key]
                    for key in (
                        "model",
                        "dataset",
                        "locale",
                        "labels",
                        "n",
                        "accuracy",
                        "macro_f1_18_labels",
                        "mean_latency_ms",
                        "seed",
                        "shuffled_options",
                    )
                },
            }
        )
    return sorted(rows, key=lambda row: row["accuracy"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "evals")
    parser.add_argument("--json-out", type=Path, default=SCRIPT_DIR / "jevs.json")
    args = parser.parse_args()
    rows = load_rows(args.results_dir)
    for row in rows:
        print(
            f"{row['model']}: accuracy={row['accuracy']:.2%}, macro F1={row['macro_f1_18_labels']:.4f}, n={row['n']}"
        )
    output = {
        "text": {
            "model": "Model",
            "memory_gb": "Memòria (GB)",
            "accuracy": "MASSIVE Accuracy",
            "macro_f1_18_labels": "MASSIVE Macro F1",
            "decisions_per_sec": "Decisions/s",
        },
        "data": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
