import json
import subprocess
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REQUIRED_FIELDS = {"model", "display_name", "params_b", "memory_gb", "benchmarks"}
EXPECTED_SAMPLE_COUNTS = {
    "benchmarks.sts_ca": 400,
    "benchmarks.casum": 400,
    "benchmarks.catcola": 400,
    "benchmarks.club_qa": 400,
    "benchmarks.iberbench.catcola": 400,
    "benchmarks.iberbench.wnli_ca": 71,
    "benchmarks.iberbench.teca": 400,
    "benchmarks.flores.catalan_bench_flores_en-ca": 400,
    "benchmarks.flores.catalan_bench_flores_ca-en": 400,
    "benchmarks.flores.catalan_bench_flores_es-ca": 400,
    "benchmarks.flores.catalan_bench_flores_ca-es": 400,
    "benchmarks.ifeval": 400,
}


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

    def test_metric_sample_counts_match_expected_dataset_sizes(self):
        bad = []
        for file in result_files():
            data = json.loads((ROOT / file).read_text(encoding="utf-8"))
            for metric, result in leaves("benchmarks", data.get("benchmarks", {})):
                expected = EXPECTED_SAMPLE_COUNTS.get(metric)
                if expected is not None and result.get("n") != expected:
                    bad.append(f"{Path(file).name}: {metric} n={result.get('n')} expected {expected}")

        self.assertFalse(bad, "Unexpected sample counts:\n" + "\n".join(bad))
