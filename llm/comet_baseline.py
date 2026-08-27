"""Calculate the FLORES random-mismatch COMET baseline in four directions."""

from __future__ import annotations

import argparse
import json
import os
import random
from statistics import fmean, stdev
from pathlib import Path

from comet_config import COMET_CHECKPOINT

TASKS = {
    "catalan_bench_flores_en-ca": ("sentence_eng_Latn", "sentence_cat_Latn"),
    "catalan_bench_flores_ca-en": ("sentence_cat_Latn", "sentence_eng_Latn"),
    "catalan_bench_flores_es-ca": ("sentence_spa_Latn", "sentence_cat_Latn"),
    "catalan_bench_flores_ca-es": ("sentence_cat_Latn", "sentence_spa_Latn"),
}
CHUNK_SIZE = 32
DEFAULT_SEED = 1714
DEFAULT_PERMUTATIONS = 5


def deranged_permutation(size: int, seed: int, index: int) -> list[int]:
    """Return a deterministic random derangement for mismatched references.

    FLORES devtest is ordered by document. A simple `(i + 1) % n` shift keeps most
    hypotheses near the original topic, which gives COMET an unrealistically easy
    mismatch baseline. A random cycle is still deterministic but breaks document
    adjacency and guarantees that `mt` is never the row's own reference.
    """
    if size < 2:
        raise ValueError("at least two samples are required for a mismatched baseline")

    order = list(range(size))
    random.Random(f"{seed}:{index}").shuffle(order)
    permutation = [0] * size
    for current, next_item in zip(order, order[1:] + order[:1], strict=True):
        permutation[current] = next_item
    return permutation


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def result(
    scores: dict, progress: dict, n_samples: int, n_permutations: int, seed: int
) -> dict:
    value = {
        "baseline": "random-mismatched reference (fixed-seed derangements)",
        "flores_comet_checkpoint": COMET_CHECKPOINT,
        "seed": seed,
        "n_samples": n_samples,
        "n_permutations": n_permutations,
        "benchmarks": {"flores": scores},
    }
    if progress:
        value["progress"] = progress
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument(
        "--n-permutations",
        "-k",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help="Number of fixed-seed random mismatches to average.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/comet_random_mismatch_baseline.json"),
    )
    args = parser.parse_args()
    if args.n_samples < 2:
        parser.error("--n-samples must be at least 2")
    if args.n_permutations < 1:
        parser.error("--n-permutations must be positive")

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
        references = [row[reference_key] for row in samples]
        state = progress.setdefault(
            task,
            {"permutation": 0, "done": 0, "score_sum": 0.0, "permutation_scores": []},
        )
        while state["permutation"] < args.n_permutations:
            permutation_index = state["permutation"]
            permutation = deranged_permutation(
                args.n_samples, args.seed, permutation_index
            )
            examples = [
                {
                    "src": row[source_key],
                    "ref": row[reference_key],
                    "mt": references[permutation[index]],
                }
                for index, row in enumerate(samples)
            ]
            for start in range(state["done"], args.n_samples, CHUNK_SIZE):
                chunk = examples[start : start + CHUNK_SIZE]
                prediction = comet.predict(chunk, batch_size=CHUNK_SIZE, gpus=0)
                state["done"] += len(chunk)
                state["score_sum"] += sum(float(score) for score in prediction.scores)
                save(
                    args.output,
                    result(
                        scores, progress, args.n_samples, args.n_permutations, args.seed
                    ),
                )
                print(
                    f"{task}: permutation {permutation_index + 1}/{args.n_permutations}, "
                    f"{state['done']}/{args.n_samples}",
                    flush=True,
                )

            state["permutation_scores"].append(state["score_sum"] / args.n_samples)
            state["permutation"] += 1
            state["done"] = 0
            state["score_sum"] = 0.0
            save(
                args.output,
                result(
                    scores, progress, args.n_samples, args.n_permutations, args.seed
                ),
            )

        permutation_scores = state["permutation_scores"]
        scores[task] = {
            "comet": fmean(permutation_scores),
            "comet_stddev": (
                stdev(permutation_scores) if len(permutation_scores) > 1 else 0.0
            ),
            "n": args.n_samples,
            "n_permutations": args.n_permutations,
            "seed": args.seed,
        }
        del progress[task]
        save(
            args.output,
            result(scores, progress, args.n_samples, args.n_permutations, args.seed),
        )
        print(
            f"{task}: random-mismatch COMET={scores[task]['comet']:.4f} "
            f"± {scores[task]['comet_stddev']:.4f}",
            flush=True,
        )

    save(
        args.output,
        result(scores, progress, args.n_samples, args.n_permutations, args.seed),
    )
    print(f"Saved baseline to {args.output}", flush=True)


if __name__ == "__main__":
    main()
