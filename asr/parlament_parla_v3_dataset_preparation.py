#!/usr/bin/env python3
"""Download a fixed, local ParlamentParla v3 Catalan evaluation set."""

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_url
import requests


DATASET = "projecte-aina/parlament_parla_v3"
REVISION = "5fabb9fb9cf76ecf8a6cf915dd41d276d48fd0ff"
SPLIT = "clean_test_short"
REPO_TYPE = "dataset"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/parlament_parla_v3_test_400"),
    )
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--max-duration", type=float, default=30)
    args = parser.parse_args()

    output, audio_dir = args.output_dir, args.output_dir / "audio"
    if output.exists() and (
        (output / "manifest.json").exists()
        or any(path != audio_dir for path in output.iterdir())
        or (audio_dir.exists() and any(audio_dir.iterdir()))
    ):
        parser.error(f"{output} already exists")
    audio_dir.mkdir(parents=True, exist_ok=True)

    metadata_url = hf_hub_url(
        DATASET,
        "corpus/files/clean_test_parlament_short.csv",
        repo_type=REPO_TYPE,
        revision=REVISION,
    )
    with requests.get(metadata_url, stream=True) as response:
        response.raise_for_status()
        rows = list(
            csv.DictReader((line.decode("utf-8") for line in response.iter_lines()))
        )
    wanted = {row["identifier"]: row for row in rows[: args.num_samples]}
    records = []
    for shard in range(1, 3):
        archive_url = hf_hub_url(
            DATASET,
            f"corpus/speech/{SPLIT}/{SPLIT}_{shard}.tar.gz",
            repo_type=REPO_TYPE,
            revision=REVISION,
        )
        with requests.get(archive_url, stream=True) as response:
            response.raise_for_status()
            with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
                for member in archive:
                    identifier = Path(member.name).stem
                    if identifier not in wanted or not member.isfile():
                        continue
                    filename = f"audio/{identifier}.wav"
                    with (
                        archive.extractfile(member) as source,
                        (output / filename).open("wb") as target,
                    ):
                        shutil.copyfileobj(source, target)
                    duration = sf.info(output / filename).duration
                    if duration <= args.max_duration:
                        row = wanted.pop(identifier)
                        records.append(
                            {
                                "id": identifier,
                                "audio": filename,
                                "duration_s": round(duration, 6),
                                "reference": row["text"],
                            }
                        )
                    if len(records) == args.num_samples:
                        break
        if len(records) == args.num_samples:
            break
    if len(records) != args.num_samples:
        raise ValueError(f"could only find {len(records)} of {args.num_samples} clips")

    payload = {
        "dataset": {"path": DATASET, "split": SPLIT, "revision": REVISION},
        "benchmark": {"key": "parlament_parla_v3", "label": "ParlamentParla v3"},
        "records": records,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} clips to {output}")


if __name__ == "__main__":
    main()
