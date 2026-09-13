"""Small shared helpers for creating local ASR benchmarks."""

import hashlib
import json
import shutil
from pathlib import Path


def prepare_output(output: Path) -> Path:
    audio_dir = output / "audio"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} already exists")
    audio_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir


def copy_audio(source: Path, output: Path, filename: str) -> None:
    shutil.copy2(source, output / filename)


def write_manifest(output: Path, manifest: dict) -> None:
    manifest["sha256"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
