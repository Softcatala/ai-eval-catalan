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
    def test_all_local_models_have_benchmark_entry(self):
        expected = {model["model_spec"] for model in _local_models()}
        actual = {run.get("model_spec") for run in benchmark_runs() if isinstance(run, dict)}
        self.assertFalse(expected - actual)
