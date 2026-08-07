"""
Reads all result JSONs and prints a summary table of metrics per model.

Usage:
  python summarize_results.py
  python summarize_results.py --results-dir .        # default
"""

import argparse
import json
import sys
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

    flores = benchmarks.get("flores", {})
    if flores:
        en2ca = flores.get("catalan_bench_flores_en-ca", {})
        ca2en = flores.get("catalan_bench_flores_ca-en", {})
        if en2ca:
            metrics["flores_en2ca"] = en2ca.get("bleu,none")
        if ca2en:
            metrics["flores_ca2en"] = ca2en.get("bleu,none")

    ifeval = benchmarks.get("ifeval", {})
    if ifeval and "error" not in ifeval:
        prompt_strict = ifeval.get("prompt_level_strict_acc,none")
        if prompt_strict is not None:
            metrics["ifeval_prompt_strict"] = prompt_strict

    return metrics




# Random baselines per task for normalization (HF Open LLM Leaderboard v2 approach)
# Classification with N classes: 1/N; regression/correlation: 0; BLEU (pre-divided by 100): 0
RANDOM_BASELINES = {
    "sts_ca":  0.0,   # correlation, ranges -1..1
    "catcola_mcc":     0.0,   # MCC for binary classification: random baseline is 0
    "club_qa_em":      0.0,   # bounded 0..1, no trivial guesser
    "casum_rougeL":    0.0,   # bounded 0..1
    "flores_en2ca":    0.0,   # BLEU/100 → 0..1
    "flores_ca2en":    0.0,   # BLEU/100 → 0..1
    "ifeval_prompt_strict": 0.0,  # prompt-level strict accuracy, bounded 0..1
}

CLAM_TASKS = list(RANDOM_BASELINES.keys())

COLUMN_LABELS = {
    "model": "Model",
    "cloud": "Cloud",
    "params_b": "Parameters (B)",
    "memory_gb": "Memory (GB)",
    "sts_ca": "STS",
    "catcola_mcc": "CatCoLA MCC",
    "club_qa_em": "CLUB QA",
    "casum_rougeL": "CaSum",
    "flores_en2ca": "EN→CA",
    "flores_ca2en": "CA→EN",
    "ifeval_prompt_strict": "IFEval",
    "clam_pct": "CLAM%",
}


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


def clam_score(metrics: dict) -> float | None:
    """Compute CLAM composite score (0–100) as mean of normalized task scores."""
    translation_keys = ("flores_en2ca", "flores_ca2en")
    normalized = [
        normalize_score(k, metrics.get(k))
        for k in CLAM_TASKS
        if k not in translation_keys
    ]
    translation = [normalize_score(k, metrics.get(k)) for k in translation_keys]
    translation = [v for v in translation if v is not None]
    if translation:
        normalized.append(sum(translation) / len(translation))
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
      <th>Params (mem)</th>
      {% for col in raw_cols %}<th>{{ col }}</th>{% endfor %}
    </tr>
  </thead>
  <tbody>
    {% for row in rows %}{% set label, metrics, cloud, params_b, memory_gb, quantized_analysis_only, quantization = row %}
    <tr>
      <td>{% if cloud %}<b>{{ label }}</b>{% else %}{{ label }}{% endif %}</td>
      <td>{{ row | fmt_params }}</td>
      {% for col in raw_cols %}<td>{{ metrics.get(col) | fmt }}</td>{% endfor %}
    </tr>
    {% endfor %}
  </tbody>
</table>

<h2>Normalized scores (HF Open LLM v2) + CLAM composite</h2>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Params (mem)</th>
      {% for col in norm_cols %}<th>{{ col }}</th>{% endfor %}
      <th>CLAM%</th>
    </tr>
  </thead>
  <tbody>
    {% for row in rows %}{% set label, metrics, cloud, params_b, memory_gb, quantized_analysis_only, quantization = row %}
    <tr>
      <td>{% if cloud %}<b>{{ label }}</b>{% else %}{{ label }}{% endif %}</td>
      <td>{{ row | fmt_params }}</td>
      {% for col in norm_cols %}<td>{{ metrics.get(col) | norm(col) | fmt }}</td>{% endfor %}
      <td>{{ metrics | clam }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>

</body>
</html>
"""


def render_html(rows: list, all_metric_keys: list, norm_keys: list, fmt_params_fn) -> str:
    env = Environment()
    env.filters["fmt"] = fmt
    env.filters["norm"] = lambda value, key: normalize_score(key, value)
    env.filters["clam"] = lambda metrics: fmt_pct(clam_score(metrics))
    env.filters["fmt_params"] = lambda row: fmt_params_fn(row[3], row[4])
    template = env.from_string(HTML_TEMPLATE_SRC)
    return template.render(rows=rows, raw_cols=all_metric_keys, norm_cols=norm_keys)


def main():
    parser = argparse.ArgumentParser(description="Summarize eval results")
    parser.add_argument("--results-dir", default="evals", help="Directory containing result JSONs")
    parser.add_argument("--html", default="summary.html", help="Output HTML file (default: summary.html)")
    parser.add_argument("--json-norm", default="llms.json", help="Output JSON file for normalized scores (default: llms.json)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Build lookup from output path -> quantized_analysis_only using run_evals.py as source of truth
    sys.path.insert(0, str(Path(__file__).parent))
    from run_evals import MODELS
    quantized_only_by_output = {
        Path(m["output"]).name: m.get("quantized_analysis_only", False)
        for m in MODELS
    }
    quantization_by_output = {
        Path(m["output"]).name: m.get("quantization", "")
        for m in MODELS
    }

    rows = []
    all_metric_keys = []

    for label, path in discover_result_files(results_dir):
        with open(path) as f:
            data = json.load(f)
        metrics = extract_metrics(data)
        params_b = data.get("params_b")
        memory_gb = data.get("memory_gb")
        display = data.get("display_name") or label
        cloud = data.get("cloud", False)
        quantized_analysis_only = quantized_only_by_output.get(path.name, False)
        quantization = data.get("quantization") or quantization_by_output.get(path.name, "")
        rows.append((display, metrics, cloud, params_b, memory_gb, quantized_analysis_only, quantization))
        for k in metrics:
            if k not in all_metric_keys:
                all_metric_keys.append(k)

    if not all_metric_keys:
        print("No result files found.")
        return

    # Sort rows by CLAM score descending
    rows.sort(key=lambda r: clam_score(r[1]) or -1.0, reverse=True)

    def fmt_params(params_b, memory_gb) -> str:
        if params_b is None:
            return "—"
        mem = f"{memory_gb:.1f}GB" if memory_gb is not None else "?"
        return f"{params_b:.0f}B ({mem})"

    # ── Raw scores table ──────────────────────────────────────────────────────
    col_width = max(14, max(len(k) for k in all_metric_keys) + 2)
    params_col_w = 16
    label_width = max(12, max(len(label) for label, _, _cloud, *_ in rows) + 2)
    header = f"{'Model':<{label_width}}" + f"{'Params (mem)':>{params_col_w}}" + "".join(f"{k:>{col_width}}" for k in all_metric_keys)
    separator = "-" * len(header)

    print("\nRaw scores")
    print(separator)
    print(header)
    print(separator)
    for label, metrics, _cloud, params_b, memory_gb, *_ in rows:
        row = "".join(f"{fmt(metrics.get(k)):>{col_width}}" for k in all_metric_keys)
        print(f"{label:<{label_width}}{fmt_params(params_b, memory_gb):>{params_col_w}}{row}")
    print(separator)

    # ── Normalized scores + CLAM composite table ──────────────────────────────
    norm_keys = [k for k in CLAM_TASKS if k in all_metric_keys]
    norm_col_w = max(14, max(len(k) for k in norm_keys) + 2)
    clam_col_w = 10
    norm_label_w = label_width

    norm_header = (
        f"{'Model':<{norm_label_w}}"
        + f"{'Params (mem)':>{params_col_w}}"
        + "".join(f"{k:>{norm_col_w}}" for k in norm_keys)
        + f"{'CLAM%':>{clam_col_w}}"
    )
    norm_sep = "-" * len(norm_header)

    print("\nNormalized scores (HF Open LLM v2) + CLAM composite")
    print(norm_sep)
    print(norm_header)
    print(norm_sep)
    for label, metrics, _cloud, params_b, memory_gb, *_ in rows:
        norm_row = "".join(
            f"{fmt(normalize_score(k, metrics.get(k))):>{norm_col_w}}" for k in norm_keys
        )
        clam = fmt_pct(clam_score(metrics))
        print(f"{label:<{norm_label_w}}{fmt_params(params_b, memory_gb):>{params_col_w}}{norm_row}{clam:>{clam_col_w}}")
    print(norm_sep)

    # ── HTML export ───────────────────────────────────────────────────────────
    html_path = Path(args.html)
    html_path.write_text(render_html(rows, all_metric_keys, norm_keys, fmt_params), encoding="utf-8")
    print(f"\nHTML saved to {html_path}")

    # ── JSON export ───────────────────────────────────────────────────────────
    json_text = {
        "model": COLUMN_LABELS["model"],
        "cloud": COLUMN_LABELS["cloud"],
        "params_b": COLUMN_LABELS["params_b"],
        "memory_gb": COLUMN_LABELS["memory_gb"],
        **{k: COLUMN_LABELS.get(k, k) for k in norm_keys},
        "clam_pct": COLUMN_LABELS["clam_pct"],
    }
    json_metrics = {
        "clam_pct": {
            "direction": "higher_is_better",
            "subtitle": "50% usable",
            "label": "> 50% el considerem usable",
            "caption": "Línia discontínua al 50% = \"usable per a tasques en català\"",
            "success": {"min": 50, "color": "#388e3c"},
            "warning": {"min": 40, "color": "#f9a825"},
            "error": {"color": "#c62828"},
        }
    }
    json_rows = []
    for label, metrics, cloud, params_b, memory_gb, quantized_analysis_only, quantization in rows:
        entry = {
            "model": f"(*) {label}" if cloud else label,
            "cloud": cloud,
            "params_b": params_b,
            "memory_gb": memory_gb,
            "quantized_analysis_only": quantized_analysis_only,
            "quantization": quantization,
            **{k: round(normalize_score(k, metrics.get(k)), 4) if normalize_score(k, metrics.get(k)) is not None else None for k in norm_keys},
            "clam_pct": round(clam_score(metrics), 2) if clam_score(metrics) is not None else None,
        }
        json_rows.append(entry)
    json_path = Path(args.json_norm)
    json_path.write_text(
        json.dumps(
            {"text": json_text, "metrics": json_metrics, "data": json_rows},
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Normalized JSON saved to {json_path}")


if __name__ == "__main__":
    main()
