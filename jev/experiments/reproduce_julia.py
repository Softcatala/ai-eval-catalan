# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "torch==2.11.0", "transformers==5.18.0", "tokenizers==0.23.2",
#   "huggingface-hub==1.33.0", "safetensors==0.8.0",
# ]
# ///
"""Reproduce the native Julia Catalan/English prompt comparison on fixed IDs."""

import argparse
from collections import Counter
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import tarfile
import types
import urllib.request

REVISION = "a85b127321d580d65176c89ced8273f305745d85"
WEIGHTS_SHA = "df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72"
DATA_SHA = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
DATA_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
HERE = Path(__file__).resolve().parent


def verify(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".julia-catalan-cache"))
    parser.add_argument("--output", type=Path, default=Path("julia-reproduction.json"))
    parser.add_argument("--head-length", type=int, default=256)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    import torch
    from transformers import AutoTokenizer

    checkpoint = args.checkpoint or Path(
        snapshot_download(
            "SupersonicLabs/Julia-1",
            revision=REVISION,
            local_dir=args.cache / "checkpoint",
            allow_patterns=[
                "julia/*.py",
                "encoder/config.json",
                "tokenizer/*",
                "julia_config.json",
                "model.safetensors",
            ],
        )
    )
    verify(checkpoint / "model.safetensors", WEIGHTS_SHA)
    archive = args.archive or args.cache / "massive-1.1.tar.gz"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, archive)
    verify(archive, DATA_SHA)
    with tarfile.open(archive, "r:gz") as source:
        member = next(m for m in source if Path(m.name).name == "ca-ES.jsonl")
        rows = [json.loads(line) for line in source.extractfile(member)]
    rows = {
        str(r["id"]): r
        for r in rows
        if r["partition"] == "test" and r["locale"] == "ca-ES"
    }
    ids = json.loads((HERE / "sample-ids.json").read_text())
    rows = [rows[sample_id] for sample_id in ids]

    # Load the publisher's low-level model and collator without its router API.
    package = types.ModuleType("julia")
    package.__path__ = [str(checkpoint / "julia")]
    sys.modules["julia"] = package
    data = importlib.import_module("julia.data")
    native = importlib.import_module("julia.model")
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
    network = native.JuliaDecisionModel.from_pretrained(checkpoint).eval()
    collator = data.Collator(tokenizer, max_length=1024, head_length=args.head_length)
    results = []
    for case in json.loads((HERE / "prompts.json").read_text()):
        if case["name"] not in ("question_0_options_0", "question_1_options_1"):
            continue
        keys = list(case["criteria"])
        records = [
            dict(
                state=r["utt"],
                question=case["question"],
                options=list(case["criteria"].values()),
                type="choice",
            )
            for r in rows
        ]
        # Audit every example; inference still uses the original standard collator.
        for record in records:
            data.sequence(
                tokenizer,
                record,
                max_length=1024,
                head_length=args.head_length,
                strict=True,
            )
        predictions = []
        with torch.inference_mode():
            for offset in range(0, len(rows), 8):
                scores = network(
                    **collator(records[offset : offset + 8], include_targets=False)
                )
                for row, index in zip(
                    rows[offset : offset + 8], scores.argmax(-1).tolist()
                ):
                    predictions.append(
                        dict(
                            id=str(row["id"]),
                            text=row["utt"],
                            gold=row["scenario"],
                            prediction=keys[index],
                        )
                    )
        result = dict(
            case=case["name"],
            n=len(rows),
            correct=sum(p["gold"] == p["prediction"] for p in predictions),
            counts=dict(Counter(p["prediction"] for p in predictions)),
            predictions=predictions,
        )
        print(f"{case['name']}: {result['correct']}/{len(rows)}", flush=True)
        results.append(result)
    output = dict(
        revision=REVISION,
        weights_sha256=WEIGHTS_SHA,
        archive_sha256=DATA_SHA,
        python=platform.python_version(),
        head_length=args.head_length,
        packages={
            p: importlib.metadata.version(p)
            for p in [
                "torch",
                "transformers",
                "tokenizers",
                "huggingface-hub",
                "safetensors",
            ]
        },
        results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
