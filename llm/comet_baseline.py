"""Calculate the FLORES source-copy COMET baseline in four directions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from comet_config import COMET_CHECKPOINT

TASKS = {
    "catalan_bench_flores_en-ca": ("sentence_eng_Latn", "sentence_cat_Latn"),
    "catalan_bench_flores_ca-en": ("sentence_cat_Latn", "sentence_eng_Latn"),
    "catalan_bench_flores_es-ca": ("sentence_spa_Latn", "sentence_cat_Latn"),
    "catalan_bench_flores_ca-es": ("sentence_cat_Latn", "sentence_spa_Latn"),
}
CHUNK_SIZE = 8


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def result(scores: dict, progress: dict, n_samples: int) -> dict:
    value = {
        "baseline": "source-copy (mt = src)",
        "flores_comet_checkpoint": COMET_CHECKPOINT,
        "benchmarks": {"flores": scores},
    }
    if progress:
        value["progress"] = progress
        value["n_samples"] = n_samples
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/comet_source_copy_baseline.json"),
    )
    args = parser.parse_args()
    if args.n_samples < 1:
        parser.error("--n-samples must be positive")

    # FLORES is a legacy dataset script, so datasets requires this opt-in.
    from datasets import load_dataset
    from comet import download_model, load_from_checkpoint

    dataset = load_dataset(
        "facebook/flores", "all", split="devtest", trust_remote_code=True
    )
    if args.n_samples > len(dataset):
        parser.error(f"--n-samples exceeds the {len(dataset)} FLORES devtest examples")

    comet = load_from_checkpoint(download_model(COMET_CHECKPOINT))
    samples = dataset.select(range(args.n_samples))
    existing = json.loads(args.output.read_text()) if args.output.exists() else {}
    scores = existing.get("benchmarks", {}).get("flores", {})
    progress = existing.get("progress", {})
    for task, (source_key, reference_key) in TASKS.items():
        if task in scores:
            continue
        examples = [
            {"src": row[source_key], "ref": row[reference_key], "mt": row[source_key]}
            for row in samples
        ]
        state = progress.setdefault(task, {"done": 0, "score_sum": 0.0})
        for start in range(state["done"], args.n_samples, CHUNK_SIZE):
            chunk = examples[start : start + CHUNK_SIZE]
            prediction = comet.predict(chunk, batch_size=CHUNK_SIZE, gpus=0)
            state["done"] += len(chunk)
            state["score_sum"] += sum(float(score) for score in prediction.scores)
            save(args.output, result(scores, progress, args.n_samples))
            print(f"{task}: {state['done']}/{args.n_samples}", flush=True)
        scores[task] = {"comet,none": state["score_sum"] / args.n_samples, "n": args.n_samples}
        del progress[task]
        save(args.output, result(scores, progress, args.n_samples))
        print(f"{task}: source-copy COMET={scores[task]['comet,none']:.4f}", flush=True)

    save(args.output, result(scores, progress, args.n_samples))
    print(f"Saved baseline to {args.output}", flush=True)


if __name__ == "__main__":
    main()
