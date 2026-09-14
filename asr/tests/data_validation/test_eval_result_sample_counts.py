import json
import unittest
from pathlib import Path


ASR_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ASR_ROOT / "evals"
REQUIRED_FIELDS = {
    "model",
    "params_b",
    "memory_gb",
    "hardware",
    "evaluated_at",
    "benchmarks",
}
EXPECTED_SAMPLE_COUNT = 400
OPENSLR_BENCHMARK = "openslr69_ca"


class EvalResultSampleCountsTest(unittest.TestCase):
    def test_result_files_have_required_fields(self):
        missing = []
        for path in sorted(RESULTS_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            missing.extend(
                f"{path.name}: {field}" for field in REQUIRED_FIELDS - data.keys()
            )

        self.assertFalse(missing, "Missing required fields:\n" + "\n".join(missing))

    def test_fleurs_results_use_all_samples(self):
        incomplete = []
        for path in sorted(RESULTS_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            n = data.get("benchmarks", {}).get("fleurs_ca", {}).get("n")
            if n != EXPECTED_SAMPLE_COUNT:
                incomplete.append(
                    f"{path.name}: fleurs_ca n={n}, expected {EXPECTED_SAMPLE_COUNT}"
                )

        self.assertFalse(
            incomplete, "Incomplete FLEURS evaluations:\n" + "\n".join(incomplete)
        )

    def test_openslr_results_use_all_samples(self):
        incomplete = []
        for path in sorted(RESULTS_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            n = data.get("benchmarks", {}).get(OPENSLR_BENCHMARK, {}).get("n")
            if n != EXPECTED_SAMPLE_COUNT:
                incomplete.append(
                    f"{path.name}: {OPENSLR_BENCHMARK} n={n}, "
                    f"expected {EXPECTED_SAMPLE_COUNT}"
                )

        self.assertFalse(
            incomplete,
            "Incomplete OpenSLR-69 evaluations:\n" + "\n".join(incomplete),
        )
