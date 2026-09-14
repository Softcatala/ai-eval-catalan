#!/usr/bin/env python3
"""Download a fixed, local OpenSLR-69 Catalan evaluation set."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset


DATASET = "openslr/openslr"
CONFIG = "SLR69"
REVISION = "6ecf590be4a156bbc909fb20ced53332e95b4a33"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/openslr69_ca_eval_400")
    )
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--max-duration", type=float, default=30)
    args = parser.parse_args()

    output, audio_dir = args.output_dir, args.output_dir / "audio"
    partial_path = output / "records.partial.json"
    if (output / "manifest.json").exists():
        parser.error(f"{output} already exists")
    audio_dir.mkdir(parents=True, exist_ok=True)

    records = (
        json.loads(partial_path.read_text(encoding="utf-8"))
        if partial_path.exists()
        else []
    )
    seen_ids = {record["id"] for record in records}
    dataset = load_dataset(
        DATASET,
        CONFIG,
        split="train",
        streaming=True,
        revision=REVISION,
        trust_remote_code=True,
    )
    for sample in dataset:
        sample_id = Path(sample["path"].split("::", 1)[0]).stem
        if sample_id in seen_ids:
            continue
        audio = sample["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        if duration > args.max_duration:
            continue
        filename = f"audio/{sample_id}.wav"
        sf.write(
            output / filename,
            np.asarray(audio["array"]),
            audio["sampling_rate"],
            subtype="PCM_16",
        )
        records.append(
            {
                "id": sample_id,
                "audio": filename,
                "duration_s": round(duration, 6),
                "reference": sample["sentence"],
            }
        )
        seen_ids.add(sample_id)
        partial_path.write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8"
        )
        if len(records) == args.num_samples:
            break

    if len(records) != args.num_samples:
        raise ValueError(f"could only find {len(records)} samples")

    payload = {
        "dataset": {
            "path": DATASET,
            "config": CONFIG,
            "split": "train",
            "revision": REVISION,
        },
        "benchmark": {"key": "openslr69_ca", "label": "OpenSLR-69 Catalan"},
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
    partial_path.unlink()
    print(f"Wrote {len(records)} clips to {output}")


if __name__ == "__main__":
    main()
