"""Console report for theoretical vs observed CLAM component weights."""

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from summarize_results import CLAM_TASKS, extract_metrics, normalize_score


RESULTS_DIR = ROOT / "llm" / "evals"
TRANSLATION_KEYS = ("flores_en_ca", "flores_es_ca")
TRANSLATION = "translation_score"
LABELS = {
    "sts_ca": "STS",
    "catcola_mcc": "CatCoLA MCC",
    "club_qa_f1": "CLUB QA",
    "casum_rougeL": "CaSum",
    "ifeval_prompt_strict": "IFEval",
    TRANSLATION: "Translation",
}


def label(key):
    return LABELS.get(key, key)


def avg(xs):
    return sum(xs) / len(xs)


def cov(xs, ys):
    mx, my = avg(xs), avg(ys)
    return avg([(x - mx) * (y - my) for x, y in zip(xs, ys)])


def load_rows(results_dir):
    rows = []
    for path in sorted(results_dir.glob("results_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = extract_metrics(data)
        components = {}

        for key in CLAM_TASKS:
            if key in TRANSLATION_KEYS:
                continue
            value = normalize_score(key, metrics.get(key))
            if value is not None:
                components[key] = value

        translations = [
            value
            for key in TRANSLATION_KEYS
            if (value := normalize_score(key, metrics.get(key))) is not None
        ]
        if translations:
            components[TRANSLATION] = fmean(translations)
        if not components:
            continue

        weight = 1 / len(components)
        rows.append(
            {
                "model": data.get("display_name") or data.get("model") or path.stem,
                "components": components,
                "weights": {key: weight for key in components},
                "clam": avg(list(components.values())),
            }
        )
    return rows


def component_keys(rows):
    keys = [key for key in CLAM_TASKS if key not in TRANSLATION_KEYS] + [TRANSLATION]
    return [key for key in keys if any(key in row["components"] for row in rows)]


def weight_table(rows):
    clams = [row["clam"] for row in rows]
    clam_var = cov(clams, clams)
    if clam_var == 0:
        raise SystemExit("CLAM has zero variance; real weights are undefined.")

    table = []
    for key in component_keys(rows):
        theoretical = avg([row["weights"].get(key, 0) for row in rows])
        contribution = [
            row["weights"].get(key, 0) * row["components"].get(key, 0)
            for row in rows
        ]
        real = cov(contribution, clams) / clam_var
        factor = theoretical / real if real > 0 else None
        table.append((key, theoretical, real, factor))
    return table


def correction_factors(table):
    return {key: factor for key, _theoretical, _real, factor in table if factor}


def projected_clam(row, factors):
    weights = {
        key: row["weights"][key] * factors.get(key, 1) for key in row["components"]
    }
    total = sum(weights.values())
    return sum(weights[key] * row["components"][key] for key in weights) / total


def print_weights(rows, table):
    width = max(len("Component CLAM"), max(len(label(key)) for key, *_ in table))
    header = (
        f"{'CLAM component':<{width}}  {'Theoretical':>11}  "
        f"{'Observed':>9}  {'Correction factor':>17}"
    )
    print(f"Models: {len(rows)}")
    print("Observed weight = covariance contribution to CLAM variance.")
    print("Correction factor = theoretical weight / observed weight.")
    print()
    print(header)
    print("-" * len(header))
    for key, theoretical, real, factor in table:
        factor_text = "n/a" if factor is None else f"{factor:.3f}"
        print(
            f"{label(key):<{width}}  {theoretical * 100:9.2f}%  "
            f"{real * 100:8.2f}%  {factor_text:>16}"
        )


def print_current(rows):
    keys = component_keys(rows)
    model_w = max(len("Model"), max(len(row["model"]) for row in rows))
    comp_w = max(10, max(len(label(key)) for key in keys))
    header = (
        f"{'Model':<{model_w}}  {'CLAM':>6}  "
        + "".join(f"{label(key):>{comp_w + 2}}" for key in keys)
    )
    print("\nCurrent CLAM and normalized components")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda item: item["clam"], reverse=True):
        values = "".join(
            f"{row['components'].get(key, 0) * 100:>{comp_w + 1}.1f} "
            if key in row["components"]
            else f"{'--':>{comp_w + 2}}"
            for key in keys
        )
        print(f"{row['model']:<{model_w}}  {row['clam'] * 100:5.1f}  {values}")


def print_projected(rows, factors):
    model_w = max(len("Model"), max(len(row["model"]) for row in rows))
    header = (
        f"{'Model':<{model_w}}  {'Current':>8}  "
        f"{'Projected CLAM':>15}  {'Difference':>10}"
    )
    print("\nProjected CLAM with suggested correction factors")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    projected = [(row, projected_clam(row, factors)) for row in rows]
    for row, score in sorted(projected, key=lambda item: item[1], reverse=True):
        print(
            f"{row['model']:<{model_w}}  {row['clam'] * 100:8.2f}  "
            f"{score * 100:15.2f}  {(score - row['clam']) * 100:10.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--current_clam", action="store_true")
    parser.add_argument("--current_projected", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.results_dir)
    if len(rows) < 2:
        raise SystemExit("Need at least two result files with CLAM components.")

    table = weight_table(rows)
    print_weights(rows, table)
    if args.current_clam:
        print_current(rows)
    if args.current_projected:
        print_projected(rows, correction_factors(table))


if __name__ == "__main__":
    main()
