import json
import subprocess
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NON_METRICS = {"n", "n_valid", "n_invalid", "alias", "invalid_rate"}
REQUIRED_FIELDS = {"model", "display_name", "params_b", "memory_gb", "benchmarks"}


def result_files():
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "llm/evals/results_*.json"],
        text=True,
    ).splitlines()


def leaves(prefix, value):
    if not isinstance(value, dict) or "error" in value:
        return
    children = [(k, v) for k, v in value.items() if isinstance(v, dict)]
    if children:
        for key, child in children:
            yield from leaves(f"{prefix}.{key}", child)
    elif any("," in key or key in {"pearson", "mcc", "rougeL", "exact_match_approx"} for key in value):
        yield prefix, value


class EvalResultSampleCountsTest(unittest.TestCase):
    def test_result_files_have_required_fields(self):
        missing = []
        for file in result_files():
            data = json.loads((ROOT / file).read_text(encoding="utf-8"))
            missing += [f"{Path(file).name}: {key}" for key in REQUIRED_FIELDS - data.keys()]
        self.assertFalse(missing, "Missing required fields:\n" + "\n".join(missing))

    def test_metric_sample_counts_are_stored_and_consistent(self):
        counts = defaultdict(set)
        missing = []

        for file in result_files():
            data = json.loads((ROOT / file).read_text(encoding="utf-8"))
            for metric, result in leaves("benchmarks", data.get("benchmarks", {})):
                missing += [] if "n" in result else [f"{Path(file).name}: {metric}"]
                counts[metric].add(result.get("n"))

        self.assertFalse(missing, "Missing sample counts:\n" + "\n".join(missing))
        self.assertFalse(
            bad := {k: sorted(v) for k, v in counts.items() if len(v) > 1},
            f"Inconsistent sample counts: {bad}",
        )

    def test_metric_values_are_positive(self):
        bad = []
        for file in result_files():
            data = json.loads((ROOT / file).read_text(encoding="utf-8"))
            for metric, result in leaves("benchmarks", data.get("benchmarks", {})):
                for key, value in result.items():
                    if key in NON_METRICS or "stderr" in key or not isinstance(value, (int, float)):
                        continue
                    bad += [] if value > 0 else [f"{Path(file).name}: {metric}.{key}={value}"]

        self.assertFalse(bad, "Non-positive metric values:\n" + "\n".join(bad))
