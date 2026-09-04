import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "llm"))

from summarize_results import CLAM_TASKS, extract_metrics


RESULTS_DIR = ROOT / "llm" / "evals"


class ClamComponentsTest(unittest.TestCase):
    def test_all_result_files_have_every_clam_component(self):
        missing = []

        for path in sorted(RESULTS_DIR.glob("results_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            metrics = extract_metrics(data)
            missing.extend(
                f"{path.name}: {component}"
                for component in CLAM_TASKS
                if metrics.get(component) is None
            )

        self.assertFalse(
            missing,
            "Missing CLAM components:\n" + "\n".join(missing),
        )
