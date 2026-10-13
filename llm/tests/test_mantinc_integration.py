from unittest.mock import patch

import pytest
import model
from mantinc import dataset_path


def test_mantinc_dependency_bundles_dataset():
    path = dataset_path()

    assert path.is_file()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 300


def test_run_catalan_drift_uses_packaged_task(tmp_path, monkeypatch):
    language_model = tmp_path / "lid.176.ftz"
    language_model.touch()
    monkeypatch.setenv("LANGUAGE_ID_MODEL", str(language_model))
    with (
        patch.object(
            model, "mantinc_task_path", return_value=tmp_path / "lm_eval_tasks"
        ),
        patch.object(model, "run_ifeval", return_value={}) as run_ifeval,
    ):
        model.run_catalan_drift("model", n_samples=100)

    assert run_ifeval.call_args.kwargs["task"] == model.MANTINC_TASK_NAME
    assert run_ifeval.call_args.kwargs["include_path"] == tmp_path / "lm_eval_tasks"


def test_run_catalan_drift_rejects_missing_language_model(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGUAGE_ID_MODEL", str(tmp_path / "missing.ftz"))
    with pytest.raises(RuntimeError, match="Language identification model not found"):
        model.run_catalan_drift("model")
