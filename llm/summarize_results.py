"""
Reads all result JSONs and prints a summary table of metrics per model.

Usage:
  python summarize_results.py
  python summarize_results.py --results-dir .        # default
"""

import argparse
import json
import re
from statistics import fmean
from pathlib import Path

from comet_config import COMET_CHECKPOINT
from jinja2 import Environment

from eval_common.model_urls import repo_url

SCRIPT_DIR = Path(__file__).parent


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
        metrics["club_qa_f1"] = club_qa["token_f1"]

    casum = benchmarks.get("casum", {})
    if casum:
        if "rougeL" in casum:
            metrics["casum_rougeL"] = casum["rougeL"]

    flores = benchmarks.get("flores", {})
    if flores:
        directions = {
            "flores_en2ca": "catalan_bench_flores_en-ca",
            "flores_ca2en": "catalan_bench_flores_ca-en",
            "flores_es2ca": "catalan_bench_flores_es-ca",
            "flores_ca2es": "catalan_bench_flores_ca-es",
        }
        for metric, task in directions.items():
            result = flores.get(task, {})
            if result and result.get("comet") is not None:
                metrics[metric] = result["comet"]

        for pair, keys in {
            "flores_en_ca": ("flores_en2ca", "flores_ca2en"),
            "flores_es_ca": ("flores_es2ca", "flores_ca2es"),
        }.items():
            values = [metrics[key] for key in keys if metrics.get(key) is not None]
            # A language-pair score is bidirectional, so do not publish a
            # misleading half-pair when one direction failed or is missing.
            if len(values) == len(keys):
                metrics[pair] = fmean(values)

    ifeval = benchmarks.get("ifeval", {})
    if ifeval and "error" not in ifeval:
        prompt_strict = ifeval.get("prompt_level_strict_acc,none")
        if prompt_strict is not None:
            metrics["ifeval_prompt_strict"] = prompt_strict

    return metrics


def configured_model_id(model_config: dict) -> str | None:
    args = model_config.get("args", [])
    flags = {"--gemini-model", "--openai-model", "--openrouter-model", "--model"}
    return next((args[i + 1] for i, arg in enumerate(args[:-1]) if arg in flags), None)


def load_benchmark_speeds(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    speeds: dict[str, float] = {}
    for run in data.get("runs", []):
        if not isinstance(run, dict):
            continue
        speed = run.get("generation_tokens_per_sec")
        if not isinstance(speed, (int, float)):
            continue
        for key in (run.get("model"), run.get("model_spec"), run.get("server_model")):
            if key:
                speeds[str(key)] = float(speed)
    return speeds


# Baselines per task for normalization (HF Open LLM Leaderboard v2 approach).
# FLORES uses the measured source-copy COMET baseline (mt = src), rather than 0.
COMET_SOURCE_COPY_BASELINES = {
    "flores_en_ca": fmean((0.6808871791511774, 0.7549228701740504)),
    "flores_es_ca": fmean((0.8222366382181644, 0.822775568291545)),
}

RANDOM_BASELINES = {
    "sts_ca":  0.0,   # correlation, ranges -1..1
    "catcola_mcc":     0.0,   # MCC for binary classification: random baseline is 0
    "club_qa_f1":      0.0,   # bounded 0..1, no trivial guesser
    "casum_rougeL":    0.0,   # bounded 0..1
    **COMET_SOURCE_COPY_BASELINES,
    "ifeval_prompt_strict": 0.0,  # prompt-level strict accuracy, bounded 0..1
}

CLAM_TASKS = list(RANDOM_BASELINES.keys())

COLUMN_LABELS = {
    "model": "Model",
    "cloud": "Cloud",
    "params_b": "Paràmetres (B)",
    "memory_gb": "Memòria (GB)",
    "generation_tokens_per_sec": "tok/s",
    "sts_ca": "STS",
    "catcola_mcc": "CatCoLA MCC",
    "club_qa_f1": "CLUB QA",
    "casum_rougeL": "CaSum",
    "flores_en_ca": "EN↔CA",
    "flores_es_ca": "ES↔CA",
    "ifeval_prompt_strict": "IFEval",
    "clam": "CLAM",
}


def normalize_score(key: str, raw) -> float | None:
    """Normalize a raw metric to 0..1 using HF Open LLM Leaderboard v2 formula.

    normalized = (score − baseline) / (1 − baseline), clamped to [0, 1].
    FLORES COMET uses its measured source-copy baseline.
    """
    if raw is None:
        return None
    value = raw
    # Directional FLORES metrics are normalized for JSON diagnostics but are
    # deliberately not CLAM tasks or public table columns.
    baseline = RANDOM_BASELINES.get(key, 0.0)
    if baseline == 1.0:
        return None  # degenerate
    normalized = (value - baseline) / (1.0 - baseline)
    return max(0.0, min(1.0, normalized))


def clam_score(metrics: dict) -> float | None:
    """Compute CLAM composite score (0–100) as mean of normalized task scores."""
    translation_keys = ("flores_en_ca", "flores_es_ca")
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


def fmt_score(value) -> str:
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
    {% for row in rows %}{% set label, metrics, cloud, params_b, memory_gb, quantized_analysis_only, quantization, generation_tokens_per_sec = row %}
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
      <th>CLAM</th>
    </tr>
  </thead>
  <tbody>
    {% for row in rows %}{% set label, metrics, cloud, params_b, memory_gb, quantized_analysis_only, quantization, generation_tokens_per_sec = row %}
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
    env.filters["clam"] = lambda metrics: fmt_score(clam_score(metrics))
    env.filters["fmt_params"] = lambda row: fmt_params_fn(row[3], row[4])
    template = env.from_string(HTML_TEMPLATE_SRC)
    return template.render(rows=rows, raw_cols=all_metric_keys, norm_cols=norm_keys)


def main():
    parser = argparse.ArgumentParser(description="Summarize eval results")
    parser.add_argument("--results-dir", default=SCRIPT_DIR / "evals", help="Directory containing result JSONs")
    parser.add_argument("--html", default=SCRIPT_DIR / "summary.html", help="Output HTML file (default: summary.html)")
    parser.add_argument("--json-norm", default=SCRIPT_DIR / "llms.json", help="Output JSON file for normalized scores (default: llms.json)")
    parser.add_argument("--benchmark-json", default=SCRIPT_DIR / "benchmark.json", help="Optional local generation speed benchmark JSON")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    benchmark_speeds = load_benchmark_speeds(Path(args.benchmark_json))

    # Build lookup from output path -> quantized_analysis_only using run_evals.py as source of truth
    from llm.run_evals import MODELS
    quantized_only_by_output = {
        Path(m["output"]).name: m.get("quantized_analysis_only", False)
        for m in MODELS
    }
    quantization_by_output = {
        Path(m["output"]).name: m.get("quantization", "")
        for m in MODELS
    }
    display_name_by_output = {
        Path(m["output"]).name: m.get("display_name")
        for m in MODELS
    }
    model_id_by_output = {
        Path(m["output"]).name: configured_model_id(m)
        for m in MODELS
    }

    rows = []
    all_metric_keys = []
    repo_url_by_label = {}

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
        model_id = data["model"]
        if "/" not in model_id and not model_id.startswith(
            ("gemini-", "gpt-", "claude-", "global.anthropic.")
        ):
            model_id = model_id_by_output.get(path.name) or model_id
        speed_keys = [
            model_id_by_output.get(path.name),
            model_id,
            data.get("model"),
        ]
        if not quantized_analysis_only:
            speed_keys.extend((display, display_name_by_output.get(path.name)))
        speed_keys = tuple(
            key for key in speed_keys if key is not None
        )
        generation_tokens_per_sec = next(
            (benchmark_speeds[key] for key in speed_keys if key in benchmark_speeds),
            None,
        )
        repo_url_by_label[display] = repo_url(model_id)
        rows.append((
            display,
            metrics,
            cloud,
            params_b,
            memory_gb,
            quantized_analysis_only,
            quantization,
            generation_tokens_per_sec,
        ))
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
    direction_keys = ("flores_en2ca", "flores_ca2en", "flores_es2ca", "flores_ca2es")
    # This is the public schema and table order. Keep it stable even when a
    # result set has no value for a newly introduced task yet.
    display_metric_keys = list(CLAM_TASKS)
    col_width = max(14, max(len(k) for k in display_metric_keys) + 2)
    params_col_w = 16
    label_width = max(12, max(len(label) for label, _, _cloud, *_ in rows) + 2)
    header = f"{'Model':<{label_width}}" + f"{'Params (mem)':>{params_col_w}}" + "".join(f"{k:>{col_width}}" for k in display_metric_keys)
    separator = "-" * len(header)

    print("\nRaw scores")
    print(separator)
    print(header)
    print(separator)
    for label, metrics, _cloud, params_b, memory_gb, *_ in rows:
        row = "".join(f"{fmt(metrics.get(k)):>{col_width}}" for k in display_metric_keys)
        print(f"{label:<{label_width}}{fmt_params(params_b, memory_gb):>{params_col_w}}{row}")
    print(separator)

    # ── Normalized scores + CLAM composite table ──────────────────────────────
    norm_keys = list(CLAM_TASKS)
    norm_col_w = max(14, max(len(k) for k in norm_keys) + 2)
    clam_col_w = 10
    norm_label_w = label_width

    norm_header = (
        f"{'Model':<{norm_label_w}}"
        + f"{'Params (mem)':>{params_col_w}}"
        + "".join(f"{k:>{norm_col_w}}" for k in norm_keys)
        + f"{'CLAM':>{clam_col_w}}"
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
        clam = fmt_score(clam_score(metrics))
        print(f"{label:<{norm_label_w}}{fmt_params(params_b, memory_gb):>{params_col_w}}{norm_row}{clam:>{clam_col_w}}")
    print(norm_sep)

    # ── HTML export ───────────────────────────────────────────────────────────
    html_path = Path(args.html)
    html_path.write_text(render_html(rows, display_metric_keys, norm_keys, fmt_params), encoding="utf-8")
    print(f"\nHTML saved to {html_path}")

    # ── JSON export ───────────────────────────────────────────────────────────
    json_text = {
        "model": COLUMN_LABELS["model"],
        "cloud": COLUMN_LABELS["cloud"],
        "params_b": COLUMN_LABELS["params_b"],
        "memory_gb": COLUMN_LABELS["memory_gb"],
        "generation_tokens_per_sec": COLUMN_LABELS["generation_tokens_per_sec"],
        **{k: COLUMN_LABELS.get(k, k) for k in norm_keys},
        "clam": COLUMN_LABELS["clam"],
    }
    json_rows = []
    for label, metrics, cloud, params_b, memory_gb, quantized_analysis_only, quantization, generation_tokens_per_sec in rows:
        entry = {
            "model": f"(*) {label}" if cloud else label,
            "repo_url": repo_url_by_label[label],
            "cloud": cloud,
            "params_b": params_b,
            "memory_gb": memory_gb,
            "generation_tokens_per_sec": round(generation_tokens_per_sec, 2) if generation_tokens_per_sec is not None else None,
            "quantized_analysis_only": quantized_analysis_only,
            "quantization": quantization,
            **{k: round(normalize_score(k, metrics.get(k)), 4) if normalize_score(k, metrics.get(k)) is not None else None for k in norm_keys},
            **{
                k: round(normalize_score(k, metrics.get(k)), 4)
                if normalize_score(k, metrics.get(k)) is not None else None
                for k in direction_keys
            },
            "clam": round(clam_score(metrics), 2) if clam_score(metrics) is not None else None,
        }
        json_rows.append(entry)
    json_path = Path(args.json_norm)
    json_path.write_text(
        json.dumps(
            {
                "text": json_text,
                "flores_comet_checkpoint": COMET_CHECKPOINT,
                "data": [r for r in json_rows if not r["quantized_analysis_only"]],
            },
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Normalized JSON saved to {json_path}")

    quantized_json_rows = sorted(
        [r for r in json_rows if r["model"].startswith("gemma3")],
        key=lambda r: (-(r["params_b"] or 0), re.sub(r"-q\d+$", "", r["model"].lower()), -(r["clam"] or 0)),
    )
    quantized_json_path = json_path.with_name("llms_quantized.json")
    quantized_json_path.write_text(
        json.dumps(
            {
                "text": {
                    **json_text,
                    "params_b": "Paràmetres (B)",
                    "memory_gb": "Memòria (GB)",
                    "quantization": "Quantització",
                },
                "flores_comet_checkpoint": COMET_CHECKPOINT,
                "data": quantized_json_rows,
            },
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Quantized JSON saved to {quantized_json_path}")


if __name__ == "__main__":
    main()
