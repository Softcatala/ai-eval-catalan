"""
Reads all result JSONs and prints a summary table of metrics per model.

Usage:
  python summarize_results.py
  python summarize_results.py --results-dir .        # default
"""

import argparse
import json
from pathlib import Path

def discover_result_files(results_dir: Path) -> list[tuple[str, Path]]:
    """Find all results_*.json files and return (model_label, path) pairs."""
    entries = []
    for path in sorted(results_dir.glob("results_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            label = data.get("model", path.stem)
        except Exception:
            label = path.stem
        entries.append((label, path))
    return entries


def extract_metrics(data: dict) -> dict:
    """Flatten benchmark metrics from a result JSON into a flat dict."""
    metrics = {}
    benchmarks = data.get("benchmarks", {})

    sts_ca = benchmarks.get("sts_ca", {})
    if sts_ca:
        metrics["sts_ca_pearson"] = sts_ca.get("pearson")

    catcola = benchmarks.get("catcola", {})
    if catcola:
        metrics["catcola_mcc"] = catcola.get("mcc")

    club_qa = benchmarks.get("club_qa", {})
    if club_qa:
        metrics["club_qa_em"] = club_qa.get("exact_match_approx")

    casum = benchmarks.get("casum", {})
    if casum:
        if "rougeL" in casum:
            metrics["casum_rougeL"] = casum["rougeL"]

    iberbench = benchmarks.get("iberbench", {})
    if iberbench:
        for task, task_metrics in iberbench.items():
            if task in ("catcola", "wnli_ca", "xnli_ca", "teca"):
                continue
            if isinstance(task_metrics, dict):
                acc = task_metrics.get("acc,none") or task_metrics.get("acc") or task_metrics.get("accuracy")
                if acc is not None:
                    metrics[f"iberbench_{task}"] = acc

    flores = benchmarks.get("flores", {})
    if flores:
        en2ca = flores.get("catalan_bench_flores_en-ca", {})
        ca2en = flores.get("catalan_bench_flores_ca-en", {})
        if en2ca:
            metrics["flores_en2ca"] = en2ca.get("bleu,none")
        if ca2en:
            metrics["flores_ca2en"] = ca2en.get("bleu,none")

    return metrics


def shorten_model_label(label: str) -> str:
    """bartowski/ModelName-GGUF:Q8_0  →  ModelName"""
    import re
    # Extract quantization suffix (e.g. "Q4_K_M" or "Q8_0") before stripping
    quant = label.split(":")[-1] if ":" in label else ""
    # Strip leading path (e.g. "bartowski/")
    name = label.split("/")[-1]
    # Strip quantization suffix (e.g. ":Q8_0")
    name = name.split(":")[0]
    # Strip "-GGUF" suffix
    name = re.sub(r"-GGUF$", "", name)
    # Strip "-Instruct" and trailing version numbers
    name = re.sub(r"-Instruct(?:-\d+)?$", "", name)
    # For gemma-4 models, append quantization suffix for clarity
    is_gemma4 = "gemma-4" in name.lower() or "gemma4" in name.lower()
    if is_gemma4 and quant.upper().startswith("Q4"):
        name += " Q4"
    elif is_gemma4 and quant.upper().startswith("Q8"):
        name += " Q8"
    return name


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Summarize eval results")
    parser.add_argument("--results-dir", default="evals", help="Directory containing result JSONs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    all_metric_keys = []

    for label, path in discover_result_files(results_dir):
        with open(path) as f:
            data = json.load(f)
        metrics = extract_metrics(data)
        rows.append((shorten_model_label(label), metrics))
        for k in metrics:
            if k not in all_metric_keys:
                all_metric_keys.append(k)

    if not all_metric_keys:
        print("No result files found.")
        return

    # Print table
    col_width = max(14, max(len(k) for k in all_metric_keys) + 2)
    label_width = max(12, max(len(label) for label, _ in rows) + 2)
    header = f"{'Model':<{label_width}}" + "".join(f"{k:>{col_width}}" for k in all_metric_keys)
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)
    for label, metrics in rows:
        if metrics is None:
            row = f"{'[missing]':>{col_width}}" * len(all_metric_keys)
        else:
            row = "".join(f"{fmt(metrics.get(k)):>{col_width}}" for k in all_metric_keys)
        print(f"{label:<{label_width}}{row}")
    print(separator)


if __name__ == "__main__":
    main()
