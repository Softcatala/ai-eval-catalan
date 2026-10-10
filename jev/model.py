#!/usr/bin/env python3
"""Evaluate decision models on MASSIVE 1.1 scenarios or TE-ca inference."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import tarfile
import time
import urllib.request
import urllib.error
import urllib.parse

SCRIPT_DIR = Path(__file__).resolve().parent

# Parameter counts from the GGUF tensor shapes, in billions.
MODEL_PARAMS_B = {
    "julia-1": 0.144192769,
    "laya": 0.421029889,
    "kev-4b": 4.207062528,
    "kev-9b": 8.955900928,
    "lev": 4.205751296,
    "openjev": 26.895998464,
    "clef-flash": 9.075566084,
    "clef": 27.024054788,
    "bespoke-nimble-9b-v3": 8.953803264,
}

DATA_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
# Dataset's original scenario order; descriptions deliberately distinguish nearby labels.
LABELS = {
    "social": (
        "Xarxes socials: llegir o publicar missatges",
        "Social media: read or post messages",
    ),
    "transport": (
        "Transport: trànsit, taxis i bitllets",
        "Transport: traffic, taxis and tickets",
    ),
    "calendar": (
        "Calendari: consultar, crear o eliminar esdeveniments",
        "Calendar: query, create or remove events",
    ),
    "play": (
        "Reproducció: iniciar música, ràdio, pòdcasts, audiollibres o jocs",
        "Playback: start music, radio, podcasts, audiobooks or games",
    ),
    "news": ("Notícies: consultar l'actualitat", "News: current news reports"),
    "datetime": (
        "Data i hora: consultar o convertir hores i dates",
        "Date and time: query or convert times and dates",
    ),
    "recommendation": (
        "Recomanacions de llocs, activitats o pel·lícules",
        "Recommendations for places, activities or movies",
    ),
    "email": (
        "Correu electrònic: missatges i contactes",
        "Email: messages and contacts",
    ),
    "iot": (
        "Domòtica: controlar llums i electrodomèstics",
        "Smart home: control lights and appliances",
    ),
    "general": (
        "Conversa general: salutacions, bromes i conversa informal",
        "General conversation: greetings, jokes and small talk",
    ),
    "audio": ("Àudio: ajustar el volum o silenciar", "Audio: adjust volume or mute"),
    "lists": (
        "Llistes: consultar, afegir o eliminar elements",
        "Lists: query, add or remove items",
    ),
    "qa": (
        "Preguntes factuals: definicions, càlculs, divises i borsa",
        "Factual questions: definitions, calculations, currencies and stocks",
    ),
    "cooking": (
        "Cuina: receptes i preparació d'aliments",
        "Cooking: recipes and food preparation",
    ),
    "takeaway": (
        "Menjar a domicili: demanar o consultar una comanda",
        "Takeaway: order food or query an order",
    ),
    "music": (
        "Música: consultar informació, preferències o configuració musical",
        "Music: information, preferences or music settings",
    ),
    "alarm": (
        "Alarmes: crear, consultar o eliminar alarmes",
        "Alarms: set, query or remove alarms",
    ),
    "weather": (
        "Meteorologia: temps i previsions",
        "Weather: conditions and forecasts",
    ),
}

TECA_LABELS = {
    "entailment": (
        "La premissa implica la hipòtesi",
        "The premise entails the hypothesis",
    ),
    "neutral": (
        "La premissa no permet deduir ni refutar la hipòtesi",
        "The premise neither entails nor contradicts the hypothesis",
    ),
    "contradiction": (
        "La premissa contradiu la hipòtesi",
        "The premise contradicts the hypothesis",
    ),
}


def labels_for(args):
    return TECA_LABELS if args.dataset == "teca" else LABELS


def dataset_name(args):
    return "Tornem a TE-ca" if args.dataset == "teca" else "MASSIVE 1.1"


def benchmark_result(result, dataset):
    if dataset in result.get("benchmarks", {}):
        return result["benchmarks"][dataset]
    expected = "Tornem a TE-ca" if dataset == "teca" else "MASSIVE 1.1"
    if "accuracy" in result and result.get("dataset", "MASSIVE 1.1") == expected:
        return {k: v for k, v in result.items() if k != "benchmarks"}
    return {}


def result_document(previous, summary=None):
    metadata = (
        "model",
        "display_name",
        "provider",
        "cloud",
        "params_b",
        "memory_gb",
        "quantization",
        "model_file",
        "model_revision",
        "requested_model",
        "models",
    )
    result = {key: previous[key] for key in metadata if key in previous}
    result["benchmarks"] = {
        key: value
        for key in ("massive", "teca")
        if (value := benchmark_result(previous, key))
    }
    if summary is not None:
        result.update({key: summary[key] for key in metadata if key in summary})
        key = "teca" if summary["dataset"] == "Tornem a TE-ca" else "massive"
        result["benchmarks"][key] = summary
    for key, value in {
        "requested_model": result.get("model"),
        "models": [result.get("model")],
    }.items():
        if result.get(key) == value:
            result.pop(key, None)
    defaults = {"locale": "ca-ES", "labels": "ca"}
    result["benchmarks"] = {
        name: {
            key: value
            for key, value in benchmark.items()
            if key not in metadata and (key not in defaults or value != defaults[key])
        }
        for name, benchmark in result["benchmarks"].items()
    }
    return result


def load_rows(args):
    if args.dataset == "teca":
        if args.data:
            examples = [
                json.loads(line)
                for line in args.data.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            from datasets import load_dataset

            examples = load_dataset("projecte-aina/teca", split="test")
        rows = [
            {
                "id": i,
                "utt": f"Premissa: {item['premise']}\nHipòtesi: {item['hypothesis']}",
                "scenario": list(TECA_LABELS)[int(item["label"])],
            }
            for i, item in enumerate(examples)
        ]
        random.Random(args.seed).shuffle(rows)
        return rows[: args.limit] if args.limit else rows
    if args.data:
        lines = args.data.read_text(encoding="utf-8").splitlines()
    else:
        args.cache.mkdir(parents=True, exist_ok=True)
        archive = args.cache / "massive-1.1.tar.gz"
        if not archive.exists():
            print("Downloading MASSIVE 1.1...", flush=True)
            temporary = archive.with_suffix(".partial")
            urllib.request.urlretrieve(DATA_URL, temporary)
            temporary.replace(archive)
        with tarfile.open(archive, "r:gz") as tar:
            member = next(
                (m for m in tar if Path(m.name).name == args.locale + ".jsonl"), None
            )
            if member is None:
                raise ValueError(f"Locale {args.locale} not found in MASSIVE 1.1")
            with tar.extractfile(member) as source:
                lines = source.read().decode("utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    rows = [r for r in rows if r["partition"] == "test" and r["locale"] == args.locale]
    if not rows:
        raise ValueError("No matching test examples")
    # Select by ID so parallel locales use the same examples.
    rows.sort(key=lambda r: str(r["id"]))
    random.Random(args.seed).shuffle(rows)
    return rows[: args.limit] if args.limit else rows


def request_json(args, url, payload=None):
    headers = {"Content-Type": "application/json"}
    key_name = "OPENAI_API_KEY" if args.provider == "openai" else "SYSTEMONE_API_KEY"
    if os.environ.get(key_name):
        headers["Authorization"] = "Bearer " + os.environ[key_name]
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def model_display_name(model_id, display_name=None, params_b=None):
    model = model_id.rsplit("/", 1)[-1].split(":")[0]
    model = re.sub(r"-gguf$", "", model, flags=re.I)
    name = (
        display_name
        if display_name and display_name != model_id
        else model.replace("-", " ")
    )
    name = re.sub(r"\s+\((?:I?Q\d[^)]*|BF16|F16|F32)\)$", "", name, flags=re.I)
    if name == "Bespoke Nimble 9B v3":
        name = "Nimble 9B"
    if model_id == "gpt-6-luna" and not display_name:
        name = "GPT-6 Luna Decisions"
    if params_b is None:
        params_b = MODEL_PARAMS_B.get(model.lower())
    if params_b is not None and not re.search(r"\b\d+(?:\.\d+)?[MB]$", name, re.I):
        size = f"{params_b * 1000:.0f}M" if params_b < 1 else f"{params_b:.0f}B"
        name = f"{name} {size}"
    return name


def discover_models(args):
    url = urllib.parse.urljoin(args.url, "/v1/models")
    models = request_json(args, url)["data"]
    # Match decision-model families at name boundaries, including quantized variants.
    family = re.compile(
        r"(?:^|[/_-])(?:julia|laya|kev|lev|openjev|jev|clef|nimble|rune|d1)(?=$|[._:/-]|[0-9])",
        re.I,
    )
    return sorted({model["id"] for model in models if family.search(model["id"])})


def predict(args, row, index, model_id):
    language = 0 if args.labels == "ca" else 1
    criteria = [(key, desc[language]) for key, desc in labels_for(args).items()]
    if args.shuffle_options:
        random.Random(args.seed + index).shuffle(criteria)
    payload = {
        "state": row["utt"],
        "questions": {
            "scenario": {
                "type": "choice",
                "instructions": (
                    (
                        "Classifica la relació entre la premissa i la hipòtesi."
                        if language == 0
                        else "Classify the relationship between the premise and the hypothesis."
                    )
                    if args.dataset == "teca"
                    else (
                        "A quina categoria pertany aquesta petició a un assistent?"
                        if language == 0
                        else "Which category does this request to an assistant belong to?"
                    )
                ),
                "criteria": dict(criteria),
            }
        },
    }
    payload["model"] = model_id
    if args.provider == "openai":
        question = payload["questions"]["scenario"]
        payload = {
            "model": model_id,
            "input": row["utt"],
            "questions": [
                {
                    "name": "scenario",
                    "type": "choice",
                    "instructions": question["instructions"],
                    "choices": [
                        {"value": key, "description": desc} for key, desc in criteria
                    ],
                }
            ],
        }
    started = time.perf_counter()
    result = request_json(args, args.url, payload)
    if args.provider == "openai":
        answer = next(a for a in result["answers"] if a["name"] == "scenario")
        if answer["type"] == "refusal":
            answer = {"type": "refusal", "choice": None}
        elif answer["type"] == "choice":
            answer = dict(
                answer,
                probabilities={
                    p["value"]: p["probability"] for p in answer["probabilities"]
                },
            )
        else:
            raise ValueError(f"Unexpected answer type: {answer['type']}")
    else:
        answer = result["answers"]["scenario"]
    if answer["choice"] not in labels_for(args) and answer.get("type") != "refusal":
        raise ValueError(f"Unexpected choice: {answer['choice']}")
    return answer, result.get("model"), (time.perf_counter() - started) * 1000


def print_summary(summary, output):
    rows = [
        ("Dataset", summary["dataset"]),
        (
            "Model",
            ", ".join(summary["models"]) or summary["requested_model"] or "Unspecified",
        ),
        ("Input locale", summary["locale"]),
        ("Label language", {"ca": "Catalan", "en": "English"}[summary["labels"]]),
        ("Examples", str(summary["n"])),
        ("Accuracy", f"{summary['accuracy']:.2%}"),
        (
            "Macro F1",
            f"{next(value for key, value in summary.items() if key.startswith('macro_f1_')):.4f}",
        ),
        ("Mean latency", f"{summary['mean_latency_ms']:.2f} ms"),
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(2)]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print("\nEvaluation results")
    print(border)
    for label, value in rows:
        print(f"| {label:<{widths[0]}} | {value:<{widths[1]}} |")
    print(border)
    suffix = ".teca.jsonl" if summary["dataset"] == "Tornem a TE-ca" else ".jsonl"
    print(f"Predictions: {output.with_suffix(suffix)}")
    print(f"Results:     {output}")


def evaluate_model(args, rows, model_id, output_path):
    labels = labels_for(args)
    records = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows):
            gold = row["scenario"]
            if isinstance(gold, int):
                gold = list(labels)[gold]
            if gold not in labels:
                raise ValueError(f"Unknown scenario: {gold}")
            answer, model, elapsed = predict(args, row, index, model_id)
            record = {
                "id": row["id"],
                "locale": args.locale,
                "labels": args.labels,
                "model": model,
                "text": row["utt"],
                "gold": gold,
                "prediction": answer["choice"],
                "probabilities": answer.get("probabilities"),
                "latency_ms": elapsed,
            }
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if (index + 1) % 25 == 0:
                print(f"{index + 1}/{len(rows)}", flush=True)
    f1s = []
    for label in labels:
        tp = sum(r["gold"] == label and r["prediction"] == label for r in records)
        fp = sum(r["gold"] != label and r["prediction"] == label for r in records)
        fn = sum(r["gold"] == label and r["prediction"] != label for r in records)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0)
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_name(args),
        "locale": args.locale,
        "labels": args.labels,
        "requested_model": model_id,
        "models": sorted({r["model"] for r in records if r["model"]}),
        "seed": args.seed,
        "shuffled_options": args.shuffle_options,
        "n": len(records),
        "accuracy": sum(r["gold"] == r["prediction"] for r in records) / len(records),
        f"macro_f1_{len(labels)}_labels": sum(f1s) / len(f1s),
        "mean_latency_ms": sum(r["latency_ms"] for r in records) / len(records),
    }


def add_evaluation_arguments(parser):
    parser.add_argument("--dataset", choices=["massive", "teca"], default="massive")
    parser.add_argument(
        "--provider", choices=["systemone", "openai"], default="systemone"
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("LLAMA_SERVER_URL", "http://localhost:9090/v1"),
    )
    parser.add_argument(
        "--locale", default="ca-ES", choices=["ca-ES", "en-US", "es-ES"]
    )
    parser.add_argument("--labels", choices=["ca", "en"], default="ca")
    parser.add_argument(
        "--n-samples",
        "--limit",
        dest="limit",
        type=int,
        default=0,
        help="Number of examples; 0 = full test split (default)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-options", action="store_true")
    parser.add_argument(
        "--data", type=Path, help="Local JSONL file for the selected dataset"
    )
    parser.add_argument("--cache", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("--timeout", type=float, default=120)


def validate_arguments(parser, args):
    if args.dataset == "teca" and args.locale != "ca-ES":
        parser.error("TE-ca requires --locale ca-ES")
    if args.limit < 0:
        parser.error("--n-samples must be nonnegative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            parser.error("--provider openai requires OPENAI_API_KEY")
        args.url = "https://api.openai.com/v1/decisions"
    else:
        args.url = args.server_url.rstrip("/") + "/systemone"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser)
    parser.add_argument(
        "--model", required=True, help="Exact model ID exposed by /v1/models"
    )
    parser.add_argument("--server-model", help="Override the request model ID")
    parser.add_argument("--display-name", help="User-facing model name")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validate_arguments(parser, args)
    if args.output is None:
        args.output = (
            SCRIPT_DIR
            / "evals"
            / ("teca/teca.json" if args.dataset == "teca" else "massive.json")
        )
    try:
        previous = (
            json.loads(args.output.read_text(encoding="utf-8"))
            if args.output.exists()
            else {}
        )
    except json.JSONDecodeError:
        previous = {}
    if previous.get("model", args.model) != args.model:
        parser.error("Output belongs to a different model")
    if args.dataset == "teca" and benchmark_result(previous, "massive") and args.limit:
        parser.error("Use a separate output directory for partial TE-ca evaluations")
    previous = result_document(previous)
    previous["benchmarks"].pop(args.dataset, None)
    if previous["benchmarks"]:
        args.output.write_text(
            json.dumps(previous, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    else:
        # Remove only the result being replaced; retain other benchmarks below.
        args.output.unlink(missing_ok=True)
    predictions = args.output.with_suffix(
        ".teca.jsonl" if args.dataset == "teca" else ".jsonl"
    )
    try:
        rows = load_rows(args)
        summary = evaluate_model(
            args, rows, args.server_model or args.model, predictions
        )
        summary["requested_n_samples"] = args.limit
        summary["data_source"] = (
            str(args.data.resolve())
            if args.data
            else (
                "projecte-aina/teca:test" if args.dataset == "teca" else "MASSIVE 1.1"
            )
        )
        summary["model"] = args.model
        summary["provider"] = args.provider
        summary["cloud"] = args.provider == "openai"
        summary["display_name"] = model_display_name(args.model, args.display_name)
        stored = result_document(previous, summary)
        args.output.write_text(
            json.dumps(stored, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print_summary(summary, args.output)


if __name__ == "__main__":
    main()
