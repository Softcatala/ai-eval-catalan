#!/usr/bin/env python3
"""Create a 396-clip Catalan Common Voice 26.0 evaluation set from its archive."""

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import soundfile as sf

from benchmark_preparation import copy_audio, prepare_output, write_manifest


GROUPS = [(accent, gender) for accent in ("balearic", "central", "valencian") for gender in ("female", "male")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/commonvoice_ca_balearic_central_valencian_396"))
    args = parser.parse_args()
    tsv = next(args.source_dir.rglob("validated.tsv"), None)
    if not tsv or not (clips := tsv.parent / "clips").is_dir():
        parser.error("--source-dir must contain validated.tsv and clips/")
    try:
        prepare_output(args.output_dir)
    except ValueError as error:
        parser.error(str(error))

    candidates = {group: [] for group in GROUPS}
    with tsv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gender = row.get("gender", "").lower().split("_", 1)[0]
            group = (row.get("accent", "").lower(), gender)
            path = clips / row["path"]
            if group in candidates and row.get("sentence", "").strip() and path.is_file() and sf.info(path).duration <= 30:
                candidates[group].append((hashlib.sha256(f"20260913:{path.name}".encode()).hexdigest(), row, path))

    output, records = args.output_dir, []
    for accent, gender in GROUPS:
        chosen, speakers = [], Counter()
        for _, row, path in sorted(candidates[accent, gender]):
            if speakers[row["client_id"]] < 2:
                speakers[row["client_id"]] += 1
                chosen.append((row, path))
            if len(chosen) == 66:
                break
        if len(chosen) != 66:
            raise ValueError(f"Not enough {accent}/{gender} clips")
        for index, (row, path) in enumerate(chosen):
            audio = f"audio/{accent}_{gender}_{index:03d}.mp3"
            copy_audio(path, output, audio)
            records.append({"id": f"{accent}_{gender}_{index:03d}", "audio": audio, "reference": row["sentence"],
                            "duration_s": round(sf.info(path).duration, 6), "accent": accent, "gender_group": gender})

    manifest = {"dataset": {"name": "Mozilla Common Voice Scripted Speech 26.0 - Catalan", "url": "https://mozilladatacollective.com/datasets/cmqim2ln300tcnq070ylazhfe"}, "records": records}
    write_manifest(output, manifest)


if __name__ == "__main__":
    main()
