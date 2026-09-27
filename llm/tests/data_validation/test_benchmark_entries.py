import json
import unittest
from pathlib import Path

from benchmark import _local_models


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_JSON = ROOT / "llm" / "benchmark.json"


def benchmark_runs():
    with BENCHMARK_JSON.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("runs", [])


class BenchmarkEntriesTest(unittest.TestCase):
    def test_all_local_models_have_speed_benchmark_entry(self):
        expected = {model["model_spec"] for model in _local_models()}
        speeds = {}
        for run in benchmark_runs():
            if not isinstance(run, dict):
                continue
            model_spec = run.get("model_spec")
            speed = run.get("generation_tokens_per_sec")
            if model_spec and isinstance(speed, (int, float)) and speed > 0:
                speeds[model_spec] = speed

        self.assertFalse(expected - set(speeds))
