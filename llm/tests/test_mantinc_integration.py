from unittest.mock import patch

import model


def test_run_catalan_drift_uses_packaged_task(tmp_path):
    with (
        patch.object(
            model, "mantinc_task_path", return_value=tmp_path / "lm_eval_tasks"
        ),
        patch.object(model, "run_ifeval", return_value={}) as run_ifeval,
    ):
        model.run_catalan_drift("model", n_samples=100)

    assert run_ifeval.call_args.kwargs["task"] == model.MANTINC_TASK_NAME
    assert run_ifeval.call_args.kwargs["include_path"] == tmp_path / "lm_eval_tasks"


def test_run_catalan_drift_supports_pinned_dataset_revision(tmp_path, monkeypatch):
    task_config_path = (
        tmp_path / "lm_eval_tasks" / "catalan_drift" / "catalan_drift.yaml"
    )
    task_config_path.parent.mkdir(parents=True)
    task_config_path.write_text("task: catalan_drift\n", encoding="utf-8")
    monkeypatch.setenv("MANTINC_DATASET_REVISION", "dataset-commit")

    with (
        patch.object(
            model, "mantinc_task_path", return_value=tmp_path / "lm_eval_tasks"
        ),
        patch(
            "lm_eval.utils.load_yaml_config",
            return_value={
                "task": "catalan_drift",
                "dataset_path": "softcatala/mantinc-catalan-drift",
            },
        ),
        patch.object(model, "run_ifeval", return_value={}) as run_ifeval,
    ):
        model.run_catalan_drift("model")

    assert run_ifeval.call_args.kwargs["task"]["dataset_kwargs"] == {
        "revision": "dataset-commit"
    }
