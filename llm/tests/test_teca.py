from unittest.mock import Mock, patch

import model
from summarize_results import CLAM_TASKS, extract_metrics, normalize_score


def test_teca_scores_all_labels_and_counts_invalid_answers_as_incorrect():
    dataset = [
        {"premise": "Premissa", "hypothesis": "Hipòtesi", "label": label}
        for label in [0, "1", 2, 0, 1, 2]
    ]
    evaluator = Mock()
    evaluator.generate.side_effect = ["0", " 1\n", "2", "2", "", "Resposta: 2"]
    with patch.object(model, "load_dataset", return_value=dataset) as load:
        result = model.run_teca(evaluator, n_samples=10)
    load.assert_called_once_with("projecte-aina/teca", split="test")
    assert result == {
        "accuracy": 0.5,
        "invalid_rate": 0.3333,
        "n": 6,
        "n_valid": 4,
        "n_invalid": 2,
    }


def test_teca_respects_sample_limit():
    dataset = [{"premise": "P", "hypothesis": "H", "label": 0}] * 3
    evaluator = Mock()
    evaluator.generate.return_value = "0"
    with patch.object(model, "load_dataset", return_value=dataset):
        result = model.run_teca(evaluator, n_samples=2)
    assert result["n"] == evaluator.generate.call_count == 2
    assert result["accuracy"] == 1.0


def test_teca_is_published_without_changing_clam_components():
    metrics = extract_metrics({"benchmarks": {"teca": {"accuracy": 1.0}}})
    assert metrics["teca_accuracy"] == 1.0
    assert "teca_accuracy" not in CLAM_TASKS
    assert normalize_score("teca_accuracy", 1 / 3) == 0.0
    assert normalize_score("teca_accuracy", 1.0) == 1.0
