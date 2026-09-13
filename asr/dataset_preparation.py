#!/usr/bin/env python3
"""Download a fixed, local FLEURS Catalan evaluation set."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset


REVISION = "a3c817cbf7c08863e0c472861c7c39e27ce7f38e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/fleurs_ca_test_400")
    )
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--max-duration", type=float, default=30)
    args = parser.parse_args()

    output = args.output_dir
    audio_dir = output / "audio"
    if output.exists() and (
        (output / "manifest.json").exists()
        or any(path != audio_dir for path in output.iterdir())
        or (audio_dir.exists() and any(audio_dir.iterdir()))
    ):
        parser.error(f"{output} already exists")
    audio_dir.mkdir(parents=True, exist_ok=True)

    records = []
    dataset = load_dataset(
        "google/fleurs",
        "ca_es",
        split="test",
        streaming=True,
        revision=REVISION,
        trust_remote_code=True,
    )
    for sample in dataset:
        duration = sample["num_samples"] / 16000
        if duration > args.max_duration:
            continue
        audio = sample["audio"]
        if audio["sampling_rate"] != 16000:
            raise ValueError("expected 16 kHz FLEURS audio")
        filename = f"audio/{sample['id']}.wav"
        sf.write(output / filename, np.asarray(audio["array"]), 16000, subtype="PCM_16")
        records.append(
            {
                "id": sample["id"],
                "audio": filename,
                "duration_s": round(duration, 6),
                "reference": sample["transcription"],
            }
        )
        if len(records) == args.num_samples:
            break

    payload = {
        "dataset": {
            "path": "google/fleurs",
            "config": "ca_es",
            "split": "test",
            "revision": REVISION,
        },
        "records": records,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Wrote {len(records)} clips to {output}")


if __name__ == "__main__":
    main()
