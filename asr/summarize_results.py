"""
Reads all ASR result JSONs and generates a summary table.

Usage:
  python summarize_results.py
  python summarize_results.py --results-dir evals
  python summarize_results.py --json-out asrs.json --table-out asrs_table.html
"""

import argparse
import json
from pathlib import Path

from jinja2 import Environment


COLUMN_LABELS = {
    "model": "Model",
    "params_b": "Parameters (B)",
    "memory_gb": "Memory (GB)",
    "wer": "WER",
    "cer": "CER",
    "rtf": "RTF",
}

METRICS = ["wer", "cer", "rtf"]


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("results_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            fleurs = data.get("benchmarks", {}).get("fleurs_ca", {})
            rows.append({
                "model": data.get("model", path.stem),
                "cloud": data.get("cloud", False),
                "params_b": data.get("params_b"),
                "memory_gb": data.get("memory_gb"),
                "wer": fleurs.get("wer"),
                "cer": fleurs.get("cer"),
                "rtf": fleurs.get("rtf"),
                "n": fleurs.get("n"),
            })
        except Exception:
            pass
    return rows


def fmt(value, digits=4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def fmt_pct(value) -> str:
    if value is None:
        return "—"
    return f"{value*100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Summarize ASR eval results")
    parser.add_argument("--results-dir", default="evals")
    parser.add_argument("--json-out", default="asrs.json")
    args = parser.parse_args()

    rows = load_results(Path(args.results_dir))
    if not rows:
        print("No result files found.")
        return

    rows.sort(key=lambda r: r["wer"] if r["wer"] is not None else 9999)

    # ── Console table ─────────────────────────────────────────────────────────
    label_w = max(len(r["model"]) for r in rows) + 2
    header = f"{'Model':<{label_w}}{'WER':>10}{'CER':>10}{'RTF':>10}{'Real-time':>12}{'N':>6}"
    sep = "-" * len(header)

    print(f"\nASR Results — FLEURS Catalan ({len(rows)} model(s))")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        rt = f"{1/r['rtf']:.1f}x" if r["rtf"] else "—"
        n = str(r["n"]) if r["n"] else "—"
        print(f"{r['model']:<{label_w}}{fmt_pct(r['wer']):>10}{fmt_pct(r['cer']):>10}{fmt(r['rtf']):>10}{rt:>12}{n:>6}")
    print(sep)

    # ── JSON export ───────────────────────────────────────────────────────────
    json_text = {k: COLUMN_LABELS.get(k, k) for k in ["model", "params_b", "memory_gb"] + METRICS}
    json_rows = [
        {
            "model": f"(*) {r['model']}" if r.get("cloud", False) else r["model"],
            "cloud": r.get("cloud", False),
            "params_b": r.get("params_b"),
            "memory_gb": r.get("memory_gb"),
            **{k: round(r[k], 4) if r.get(k) is not None else None for k in METRICS},
        }
        for r in rows
    ]
    json_path = Path(args.json_out)
    json_path.write_text(
        json.dumps({"text": json_text, "data": json_rows}, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
