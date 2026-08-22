"""Calculate the FLORES source-copy COMET baseline in four directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from comet_config import COMET_CHECKPOINT
TASKS = {
    "catalan_bench_flores_en-ca": ("sentence_eng_Latn", "sentence_cat_Latn"),
    "catalan_bench_flores_ca-en": ("sentence_cat_Latn", "sentence_eng_Latn"),
    "catalan_bench_flores_es-ca": ("sentence_spa_Latn", "sentence_cat_Latn"),
    "catalan_bench_flores_ca-es": ("sentence_cat_Latn", "sentence_spa_Latn"),
}


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
    scores = {}
    for task, (source_key, reference_key) in TASKS.items():
        examples = [
            {"src": row[source_key], "ref": row[reference_key], "mt": row[source_key]}
            for row in samples
        ]
        score = float(comet.predict(examples, batch_size=8, gpus=0).system_score)
        scores[task] = {"comet,none": score, "n": args.n_samples}
        print(f"{task}: source-copy COMET={score:.4f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "baseline": "source-copy (mt = src)",
                "flores_comet_checkpoint": COMET_CHECKPOINT,
                "benchmarks": {"flores": scores},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved baseline to {args.output}", flush=True)


if __name__ == "__main__":
    main()
