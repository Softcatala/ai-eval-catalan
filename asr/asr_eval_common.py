"""Shared ASR evaluation helpers."""

import hashlib
import json
import platform
import re
import subprocess
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.lower())
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = data.pop("sha256", None)
    actual = hashlib.sha256(
        json.dumps(
            data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if not expected or expected != actual:
        raise ValueError(f"manifest hash mismatch: {path}")
    if not data.get("records"):
        raise ValueError(f"manifest contains no records: {path}")
    if data.get("dataset", {}).get("path") == "projecte-aina/parlament_parla_v3":
        data.setdefault("benchmark", {"key": "parlament_parla_v3", "label": "ParlamentParla v3"})
    return {**data, "sha256": expected}


def _run_optional(command: list[str], timeout: float = 3.0) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _gpu_name() -> str | None:
    output = _run_optional(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return None
    parts = [part.strip() for part in output.splitlines()[0].split(",")]
    if len(parts) == 2:
        return f"{parts[0]} {parts[1]} MiB"
    return parts[0] if parts else None


def hardware_string(device: str) -> str | None:
    if device == "cuda":
        return _gpu_name()
    if device == "cpu":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        return platform.processor() or platform.machine()
    return None
