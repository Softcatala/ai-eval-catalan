import json
import math
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from jev.summarize_results import model_memory_gb, model_params_b, model_repo_url


EXPECTED = {
    "dataset": "MASSIVE 1.1",
    "data_source": "MASSIVE 1.1",
    "locale": "ca-ES",
    "labels": "ca",
    "n": 2974,
    "requested_n_samples": 0,
    "seed": 42,
    "shuffled_options": False,
}


def test_evaluation_results():
    paths = [
        path
        for path in sorted(
            (Path(__file__).resolve().parents[2] / "evals").glob("*.json")
        )
        if not path.name.endswith((".summary.json", ".comparison.json"))
    ]
    assert paths, "No JEV evaluation results found"
    seen = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{path.name}: expected a JSON object"
        for field, value in EXPECTED.items():
            assert type(data.get(field)) is type(value) and data[field] == value, (
                f"{path.name}: {field} must be {value!r}"
            )
        for field in ("model", "display_name", "requested_model", "evaluated_at"):
            assert isinstance(data.get(field), str) and data[field].strip(), (
                f"{path.name}: missing or invalid {field}"
            )
        assert datetime.fromisoformat(data["evaluated_at"]).utcoffset() is not None, (
            f"{path.name}: evaluated_at must include a timezone"
        )
        assert data["model"] not in seen, (
            f"{path.name}: duplicate model {data['model']}"
        )
        seen.add(data["model"])
        models = data.get("models")
        assert (
            isinstance(models, list)
            and models
            and all(isinstance(model, str) and model.strip() for model in models)
        ), f"{path.name}: models must contain nonempty model IDs"
        for field, value in {
            "accuracy": data.get("accuracy"),
            "macro_f1_18_labels": data.get("macro_f1_18_labels"),
            "mean_latency_ms": data.get("mean_latency_ms"),
            "params_b": model_params_b(data),
            "memory_gb": model_memory_gb(data),
        }.items():
            assert type(value) in (int, float) and math.isfinite(value), (
                f"{path.name}: {field} must be a finite number"
            )
            valid = (
                0 <= value <= 1
                if field in ("accuracy", "macro_f1_18_labels")
                else value > 0
            )
            assert valid, f"{path.name}: {field} out of range: {value}"
        url = urlsplit(model_repo_url(data) or "")
        assert url.scheme == "https" and url.netloc and url.path.strip("/"), (
            f"{path.name}: model repository URL could not be resolved"
        )
