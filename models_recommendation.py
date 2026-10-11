"""Recommend local models using this repository's raw evaluation results."""

import argparse
import json
import math
from pathlib import Path

from embeddings.summarize_results import composite, is_cloud
from llm.models_config import MODELS
from llm.summarize_results import CLAM_TASKS, clam_score, extract_metrics

ROOT = Path(__file__).resolve().parent
MEMORY_FILE = ROOT / "llm" / "embedding_memory.json"
RESERVE_PERCENT = 25
DEFAULT_CAPACITIES = (8, 16, 32)
CATEGORIES = {"llm": "LLM", "embeddings": "Embeddings", "asr": "ASR"}
METRICS = {"llm": "CLAM ↑", "embeddings": "Composta ↑", "asr": "WER combinat ↓"}
SCORE_FORMATS = {"llm": ".1f", "embeddings": ".4f", "asr": ".2%"}
EXTRA_COLUMNS = {
    "llm": ("Δ CLAM", "score_gap", ".1f"),
    "asr": ("RTF (FLEURS)", "rtf", ".4f"),
}


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def evaluation_score(category, data):
    """Only compare models with every benchmark required by their category."""
    benchmarks = data.get("benchmarks", {})
    if category == "llm":
        metrics = extract_metrics(data)
        if all(finite_number(metrics.get(key)) for key in CLAM_TASKS):
            return clam_score(metrics)
    elif category == "embeddings":
        metrics = {
            "xquad_ndcg_at_10": benchmarks.get("xquad_ca_retrieval", {}).get(
                "ndcg_at_10"
            ),
            "sts_ca_spearman": benchmarks.get("sts_ca", {}).get("spearman"),
            "tecla_macro_f1": benchmarks.get("tecla_classification", {}).get(
                "macro_f1"
            ),
        }
        if all(finite_number(value) for value in metrics.values()):
            return composite(metrics)
    else:
        results = [benchmarks.get(key, {}) for key in ("fleurs_ca", "openslr69_ca")]
        if all(
            finite_number(r.get("wer"))
            and r["wer"] >= 0
            and finite_number(r.get("n"))
            and r["n"] > 0
            for r in results
        ):
            return sum(r["wer"] * r["n"] for r in results) / sum(
                r["n"] for r in results
            )
    return None


def load_candidates(root, memory_file=MEMORY_FILE):
    memory_catalog = json.loads(memory_file.read_text(encoding="utf-8"))["models"]
    analysis_only = {
        Path(model["output"]).name
        for model in MODELS
        if model.get("quantized_analysis_only")
    }
    candidates = {category: [] for category in CATEGORIES}
    skipped = []
    for category in CATEGORIES:
        directory = root / category / "evals"
        if not directory.is_dir():
            raise ValueError(f"No existeix el directori d'avaluacions: {directory}")
        for path in sorted(directory.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("quantized_analysis_only") or (
                category == "llm" and path.name in analysis_only
            ):
                continue
            if data.get("cloud") or (category == "embeddings" and is_cloud(data)):
                continue
            model_id = data.get("model", path.stem)
            model = data.get("display_name") or model_id
            source = str(path.relative_to(root))
            score = evaluation_score(category, data)
            memory = data.get("memory_gb")
            precision = data.get("quantization", "")
            if category == "llm" and ":" in model_id:
                precision = model_id.rsplit(":", 1)[1]
            if memory is None and category == "embeddings":
                spec = memory_catalog.get(model_id, {})
                memory = spec.get("memory_gb")
                precision = spec.get("precision", "")
            reason = None
            if score is None:
                reason = "avaluació incompleta"
            elif category != "asr" and (not finite_number(memory) or memory <= 0):
                reason = "memòria desconeguda o invàlida"
            if reason:
                skipped.append({"source": source, "model": model, "reason": reason})
                continue
            candidate = {
                "model": model,
                "model_id": model_id,
                "precision": precision,
                "score": score,
                "memory_gb": memory,
            }
            if category == "asr":
                rtf = data.get("benchmarks", {}).get("fleurs_ca", {}).get("rtf")
                candidate["rtf"] = rtf if finite_number(rtf) and rtf >= 0 else None
            elif category == "embeddings":
                dimension = data.get("embedding_dim")
                candidate["embedding_dim"] = (
                    dimension
                    if isinstance(dimension, int)
                    and finite_number(dimension)
                    and dimension > 0
                    else None
                )
            candidates[category].append(candidate)
    return candidates, skipped


def rank_candidates(candidates, category, budget, min_memory_gb=0):
    return sorted(
        (
            model
            for model in candidates[category]
            if finite_number(model["memory_gb"])
            and min_memory_gb < model["memory_gb"] <= budget
        ),
        key=lambda m: (
            m["score"] if category == "asr" else -m["score"],
            m["memory_gb"],
            m["model_id"],
        ),
    )


def memory_budgets(capacities):
    for capacity in capacities:
        if not finite_number(capacity) or capacity <= 0:
            raise ValueError("Les capacitats han de ser nombres positius i finits.")
        yield capacity, capacity * (1 - RESERVE_PERCENT / 100)


def recommend(candidates, capacities=DEFAULT_CAPACITIES, llm_uncertainty=2):
    if not finite_number(llm_uncertainty) or llm_uncertainty < 0:
        raise ValueError("El marge CLAM ha de ser un nombre finit no negatiu.")
    configurations = []
    for capacity, budget in memory_budgets(capacities):
        models = {}
        llm_alternatives = []
        for category in CATEGORIES:
            ranked = rank_candidates(candidates, category, budget)
            if category == "llm" and ranked:
                best_score = ranked[0]["score"]
                ranked = [
                    {
                        **model,
                        "score_gap": best_score - model["score"],
                    }
                    for model in ranked
                    if model["score"] == best_score
                    or best_score - model["score"] < llm_uncertainty
                ]
                preferred = min(
                    ranked,
                    key=lambda m: (m["memory_gb"], -m["score"], m["model_id"]),
                )
                ranked = [preferred] + [m for m in ranked if m is not preferred]
                llm_alternatives = ranked[1:]
            models[category] = ranked[0] if ranked else None
        configurations.append(
            {
                "capacity_gb": capacity,
                "budget_gb": budget,
                "models": models,
                "llm_alternatives": llm_alternatives,
            }
        )
    return configurations


def format_value(value, spec, suffix=""):
    return "—" if value is None else f"{value:{spec}}{suffix}"


def print_table(report):
    print(
        "Millors models locals segons les avaluacions del repositori (RAM)\n"
        f"Reserva: {RESERVE_PERCENT:g}%. Cada model s'executa individualment.\n"
        "Memòria orientativa; el consum real depèn del context, el lot i el motor.\n"
    )
    threshold = report["llm_uncertainty_points"]
    print(
        f"Diferències CLAM de menys de {threshold:g} punts respecte del millor "
        "LLM es consideren no concloents.\n"
        if threshold > 0
        else "Alternatives CLAM: només empats exactes amb el millor LLM.\n"
    )
    for category, title in CATEGORIES.items():
        print(title)
        header = ["GB", "Útils", "Tipus", "Model", "Precisió", "GB model", "Puntuació"]
        extra = EXTRA_COLUMNS.get(category)
        if extra:
            header.append(extra[0])
        rows = [header]
        group_starts = set()
        for config in report["configurations"]:
            group_starts.add(len(rows))
            entries = [(config["models"][category], title)]
            if category == "llm":
                entries.extend(
                    (model, "LLM (semblant)") for model in config["llm_alternatives"]
                )
            for model, label in entries:
                model = model or {}
                row = [
                    f"{config['capacity_gb']:g}",
                    f"{config['budget_gb']:g}",
                    label,
                    model.get("model", "Cap model compatible amb dades completes"),
                    model.get("precision") or "—",
                    format_value(model.get("memory_gb"), ".2f"),
                    format_value(
                        model.get("score"),
                        SCORE_FORMATS[category],
                        f" {METRICS[category]}",
                    ),
                ]
                if extra:
                    row.append(format_value(model.get(extra[1]), extra[2]))
                rows.append(row)
        widths = [max(map(len, column)) for column in zip(*rows)]
        separator = "  ".join("─" * width for width in widths)
        for index, row in enumerate(rows):
            if index in group_starts:
                print(separator)
            print("  ".join(value.ljust(width) for value, width in zip(row, widths)))
        print()
    if report["skipped"]:
        print("\nAvaluacions locals excloses:")
        for item in report["skipped"]:
            print(f"- {item['model']}: {item['reason']} ({item['source']})")


def recommendations_json(
    candidates, capacities=DEFAULT_CAPACITIES, llm_uncertainty=2, category="llm"
):
    """Export recommendations by LLM memory, ASR speed or embedding dimension."""

    def model_label(model):
        if model is None:
            return None
        precision = model.get("precision")
        name = (
            model["model"].rsplit("/", 1)[-1] if category == "asr" else model["model"]
        )
        return name + (f" · {precision}" if precision else "")

    if category in ("asr", "embeddings"):
        asr = category == "asr"
        metric = "rtf" if asr else "embedding_dim"
        ranked = sorted(
            candidates[category],
            key=lambda m: (m["score"] if asr else -m["score"], m["model_id"]),
        )
        best = ranked[0] if ranked else None
        balanced = None
        if best and finite_number(best.get(metric)) and best[metric] > 0:
            balanced = next(
                (
                    model
                    for model in ranked[1:]
                    if (
                        model["score"] <= best["score"] + 0.02
                        if asr
                        else model["score"] >= best["score"] - 0.02
                    )
                    and finite_number(model.get(metric))
                    and model[metric] > 0
                    and (
                        model[metric] <= best[metric] / 2
                        if asr
                        else model[metric] < best[metric]
                    )
                ),
                None,
            )
        descriptions = (
            (
                f"Model amb la millor qualitat, {best['score']:.1%} WER"
                if asr
                else f"Model amb la millor qualitat, vector de {best.get('embedding_dim') or '—'} dimensions "
                f"i puntuació composta {best['score']:.1f}"
            )
            if best
            else None,
            (
                f"Model amb qualitat {balanced['score']:.1%} WER i "
                f"{best['rtf'] / balanced['rtf'] - 1:.1%} més ràpid que el primer"
                if asr
                else f"Model amb qualitat similar i vector més petit, de {balanced['embedding_dim']} dimensions, "
                f"puntuació composta {balanced['score']:.1f}"
            )
            if balanced
            else None,
        )
        return {
            "text": {
                "priority": "Prioritat",
                "recommended": "Model recomanat",
                "description": "Descripció",
            },
            "data": [
                {
                    "priority": priority,
                    "recommended": model_label(model),
                    "description": description.replace(".", ",")
                    if description
                    else None,
                }
                for model, priority, description in zip(
                    (best, balanced),
                    ("Màxima precisió", "Precisió i velocitat")
                    if asr
                    else ("Màxima qualitat", "Qualitat i vector més petit"),
                    descriptions,
                )
            ],
        }

    columns = {
        "capacity_gb": "Memòria de l’ordinador",
        "recommended": "Model recomanat",
        "alternatives": "Alternativa",
    }
    rows = []
    for capacity, budget in sorted(set(memory_budgets(capacities))):
        config = recommend(
            {**{category: [] for category in CATEGORIES}, **candidates},
            [capacity],
            llm_uncertainty,
        )[0]
        model = config["models"]["llm"]
        ranked = rank_candidates(candidates, "llm", budget)
        alternative = (
            next((m for m in ranked if m["model_id"] != model["model_id"]), None)
            if model
            else None
        )
        rows.append(
            {
                "capacity_gb": f"{capacity:g} GB"
                + (" o més" if capacity == 32 else ""),
                "recommended": model_label(model),
                "alternatives": model_label(alternative),
            }
        )
    return {"text": columns, "data": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--memory",
        type=float,
        nargs="+",
        default=DEFAULT_CAPACITIES,
        help="Capacitats de RAM en GB",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--llm-uncertainty",
        type=float,
        default=2,
        help="Llindar CLAM per preferir el model més petit; 0: només empats",
    )
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument(
        "--category",
        choices=tuple(CATEGORIES),
        default="llm",
        help="Categoria del JSON",
    )
    parser.add_argument("--output", type=Path, help="Fitxer de sortida JSON")
    args = parser.parse_args(argv)
    if args.output and args.format == "table":
        parser.error("--output requereix --format json")
    try:
        candidates, skipped = load_candidates(args.repo_root)
        if args.format == "json":
            report = recommendations_json(
                candidates, args.memory, args.llm_uncertainty, args.category
            )
        else:
            report = {
                "llm_uncertainty_points": args.llm_uncertainty,
                "configurations": recommend(
                    candidates, args.memory, llm_uncertainty=args.llm_uncertainty
                ),
                "skipped": skipped,
            }
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.format == "json":
        output = (
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    else:
        print_table(report)


if __name__ == "__main__":
    main()
