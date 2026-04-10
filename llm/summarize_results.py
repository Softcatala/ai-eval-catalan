"""
Reads all result JSONs and prints a summary table of metrics per model.

Usage:
  python summarize_results.py
  python summarize_results.py --results-dir .        # default
"""

import argparse
import json
from pathlib import Path

from jinja2 import Environment

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
        metrics["sts_ca"] = sts_ca.get("pearson")

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
                acc = next(
                    (task_metrics[k] for k in ("acc,none", "acc", "accuracy") if k in task_metrics),
                    None,
                )
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


# Random baselines per task for normalization (HF Open LLM Leaderboard v2 approach)
# Classification with N classes: 1/N; regression/correlation: 0; BLEU (pre-divided by 100): 0
RANDOM_BASELINES = {
    "sts_ca":  0.0,   # correlation, ranges -1..1
    "catcola_mcc":     0.0,   # MCC for binary classification: random baseline is 0
    "club_qa_em":      0.0,   # bounded 0..1, no trivial guesser
    "casum_rougeL":    0.0,   # bounded 0..1
    "flores_en2ca":    0.0,   # BLEU/100 → 0..1
    "flores_ca2en":    0.0,   # BLEU/100 → 0..1
}

CALM_TASKS = list(RANDOM_BASELINES.keys())


def normalize_score(key: str, raw) -> float | None:
    """Normalize a raw metric to 0..1 using HF Open LLM Leaderboard v2 formula.

    normalized = (score − baseline) / (1 − baseline), clamped to [0, 1].
    BLEU scores are divided by 100 first.
    """
    if raw is None:
        return None
    value = raw
    if key in ("flores_en2ca", "flores_ca2en"):
        value = value / 100.0
    baseline = RANDOM_BASELINES[key]
    if baseline == 1.0:
        return None  # degenerate
    normalized = (value - baseline) / (1.0 - baseline)
    return max(0.0, min(1.0, normalized))


def calm_score(metrics: dict) -> float | None:
    """Compute CALM composite score (0–100) as mean of normalized task scores."""
    normalized = [normalize_score(k, metrics.get(k)) for k in CALM_TASKS]
    valid = [v for v in normalized if v is not None]
    if not valid:
        return None
    return (sum(valid) / len(valid)) * 100.0


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def fmt_pct(value) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


HTML_TEMPLATE_SRC = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Model Eval Results</title>
<style>
  body { font-family: sans-serif; padding: 20px; }
  h2 { margin-top: 30px; }
  table { border-collapse: collapse; font-family: monospace; font-size: 13px; }
  td, th { border: 1px solid #ddd; padding: 8px; }
  thead { background: #f2f2f2; }
</style>
</head>
<body>

<h2>Raw scores</h2>
<table>
  <thead>
    <tr>
      <th>Model</th>
      {% for col in raw_cols %}<th>{{ col }}</th>{% endfor %}
    </tr>
  </thead>
  <tbody>
    {% for label, metrics in rows %}
    <tr>
      <td>{{ label }}</td>
      {% for col in raw_cols %}<td>{{ metrics.get(col) | fmt }}</td>{% endfor %}
    </tr>
    {% endfor %}
  </tbody>
</table>

<h2>Normalized scores (HF Open LLM v2) + CALM composite</h2>
<table>
  <thead>
    <tr>
      <th>Model</th>
      {% for col in norm_cols %}<th>{{ col }}</th>{% endfor %}
      <th>CALM%</th>
    </tr>
  </thead>
  <tbody>
    {% for label, metrics in rows %}
    <tr>
      <td>{{ label }}</td>
      {% for col in norm_cols %}<td>{{ metrics.get(col) | norm(col) | fmt }}</td>{% endfor %}
      <td>{{ metrics | calm }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>

</body>
</html>
"""


def render_html(rows: list, all_metric_keys: list, norm_keys: list) -> str:
    env = Environment()
    env.filters["fmt"] = fmt
    env.filters["norm"] = lambda value, key: normalize_score(key, value)
    env.filters["calm"] = lambda metrics: fmt_pct(calm_score(metrics))
    template = env.from_string(HTML_TEMPLATE_SRC)
    return template.render(rows=rows, raw_cols=all_metric_keys, norm_cols=norm_keys)


def main():
    parser = argparse.ArgumentParser(description="Summarize eval results")
    parser.add_argument("--results-dir", default="evals", help="Directory containing result JSONs")
    parser.add_argument("--html", default="summary.html", help="Output HTML file (default: summary.html)")
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

    # Sort rows by CALM score descending
    rows.sort(key=lambda r: calm_score(r[1]) or -1.0, reverse=True)

    # ── Raw scores table ──────────────────────────────────────────────────────
    col_width = max(14, max(len(k) for k in all_metric_keys) + 2)
    label_width = max(12, max(len(label) for label, _ in rows) + 2)
    header = f"{'Model':<{label_width}}" + "".join(f"{k:>{col_width}}" for k in all_metric_keys)
    separator = "-" * len(header)

    print("\nRaw scores")
    print(separator)
    print(header)
    print(separator)
    for label, metrics in rows:
        row = "".join(f"{fmt(metrics.get(k)):>{col_width}}" for k in all_metric_keys)
        print(f"{label:<{label_width}}{row}")
    print(separator)

    # ── Normalized scores + CALM composite table ──────────────────────────────
    norm_keys = [k for k in CALM_TASKS if k in all_metric_keys]
    norm_col_w = max(14, max(len(k) for k in norm_keys) + 2)
    calm_col_w = 10
    norm_label_w = label_width

    norm_header = (
        f"{'Model':<{norm_label_w}}"
        + "".join(f"{k:>{norm_col_w}}" for k in norm_keys)
        + f"{'CALM%':>{calm_col_w}}"
    )
    norm_sep = "-" * len(norm_header)

    print("\nNormalized scores (HF Open LLM v2) + CALM composite")
    print(norm_sep)
    print(norm_header)
    print(norm_sep)
    for label, metrics in rows:
        norm_row = "".join(
            f"{fmt(normalize_score(k, metrics.get(k))):>{norm_col_w}}" for k in norm_keys
        )
        calm = fmt_pct(calm_score(metrics))
        print(f"{label:<{norm_label_w}}{norm_row}{calm:>{calm_col_w}}")
    print(norm_sep)

    # ── HTML export ───────────────────────────────────────────────────────────
    html_path = Path(args.html)
    html_path.write_text(render_html(rows, all_metric_keys, norm_keys), encoding="utf-8")
    print(f"\nHTML saved to {html_path}")


if __name__ == "__main__":
    main()
