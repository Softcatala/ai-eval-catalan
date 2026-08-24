"""
Render HTML table snippets from llms.json and asrs.json.

Usage:
  python render_tables.py
  python render_tables.py --llm-json llm/llms.json --asr-json asr/asrs.json
"""

import argparse
import json
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def fmt_score(value) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


def fmt_speed(value) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def fmt_dec_pct(value) -> str:
    """Format a 0–1 decimal as a percentage with 2 decimal places."""
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def fmt_params(row) -> str:
    params_b = row.get("params_b")
    memory_gb = row.get("memory_gb")
    if params_b is None:
        return "—"
    mem = f"{memory_gb:.1f}GB" if memory_gb is not None else "?"
    if params_b < 1:
        size = f"{params_b * 1000:.0f}M"
    else:
        size = f"{params_b:.0f}B"
    return f"{size} ({mem})"


def render(
    json_path: Path,
    template_path: Path,
    out: Path,
    row_filter=None,
    extra_cols=None,
    sort_key=None,
    show_params=True,
) -> None:
    data = json.loads(json_path.read_text())
    col_labels = dict(data["text"])
    rows = data["data"]
    if row_filter is not None:
        rows = [r for r in rows if row_filter(r)]
    if sort_key is not None:
        rows = sorted(rows, key=sort_key)
    cols = [
        k
        for k in col_labels
        if k != "model" and k not in ("cloud", "params_b", "memory_gb")
    ]
    if extra_cols:
        for k, label in reversed(list(extra_cols.items())):
            if k not in cols:
                cols.insert(0, k)
            col_labels[k] = label

    env = Environment(loader=FileSystemLoader(str(template_path.parent)))
    env.filters["fmt"] = fmt
    env.filters["fmt_score"] = fmt_score
    env.filters["fmt_speed"] = fmt_speed
    env.filters["fmt_dec_pct"] = fmt_dec_pct
    env.filters["fmt_params"] = fmt_params
    template = env.get_template(template_path.name)

    html = template.render(
        rows=rows, cols=cols, col_labels=col_labels, show_params=show_params
    )
    out.write_text(html, encoding="utf-8")
    print(f"Saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Render table HTML from JSON")
    parser.add_argument("--llm-json", default="llm/llms.json")
    parser.add_argument("--llm-quantized-json", default="llm/llms_quantized.json")
    parser.add_argument("--asr-json", default="asr/asrs.json")
    parser.add_argument("--llm-out", default="llm/llms_table.html")
    parser.add_argument("--llm-quantized-out", default="llm/llms_quantized_table.html")
    parser.add_argument("--asr-out", default="asr/asrs_table.html")
    parser.add_argument("--emb-json", default="embeddings/embeddings.json")
    parser.add_argument("--emb-out", default="embeddings/embeddings_table.html")
    args = parser.parse_args()

    render(
        Path(args.llm_json),
        Path("llm/table_template.jinja"),
        Path(args.llm_out),
        row_filter=lambda r: not r.get("quantized_analysis_only", False),
    )
    render(
        Path(args.llm_quantized_json),
        Path("llm/table_template.jinja"),
        Path(args.llm_quantized_out),
        extra_cols={"quantization": "Quantització"},
        sort_key=lambda r: (
            -(r.get("params_b") or 0),
            re.sub(r"-q\d+$", "", r["model"].lower()),
            -(r.get("clam") or 0),
        ),
    )
    render(Path(args.asr_json), Path("asr/table_template.jinja"), Path(args.asr_out))
    render(
        Path(args.emb_json),
        Path("embeddings/table_template.jinja"),
        Path(args.emb_out),
        sort_key=lambda r: -(r.get("composite") or -1),
        show_params=False,
    )


if __name__ == "__main__":
    main()
