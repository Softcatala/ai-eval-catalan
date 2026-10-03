import argparse
from datetime import datetime, timezone
import json
from unittest.mock import patch

import pytest

from jev import model, run_evals, summarize_results


def arguments(tmp_path):
    return argparse.Namespace(
        data=tmp_path / "ca.jsonl",
        cache=tmp_path,
        locale="ca-ES",
        labels="ca",
        seed=42,
        limit=2,
        shuffle_options=False,
    )


def test_default_evaluates_full_test_split(tmp_path):
    parser = argparse.ArgumentParser()
    model.add_evaluation_arguments(parser)
    args = parser.parse_args([])
    assert args.limit == 0
    args.data = tmp_path / "ca.jsonl"
    rows = [{"id": i, "partition": "test", "locale": "ca-ES"} for i in range(5)]
    args.data.write_text("\n".join(json.dumps(row) for row in rows))
    assert len(model.load_rows(args)) == 5
    args.limit = 2
    assert len(model.load_rows(args)) == 2


def test_only_selected_test_partition_and_parallel_ids(tmp_path):
    args = arguments(tmp_path)
    rows = [
        {
            "id": i,
            "partition": partition,
            "locale": locale,
            "utt": "hola",
            "scenario": "general",
        }
        for i in range(5)
        for partition in ("train", "test")
        for locale in ("ca-ES", "en-US")
    ]
    args.data.write_text("\n".join(json.dumps(row) for row in rows))
    catalan = model.load_rows(args)
    assert len(catalan) == 2
    assert all(
        row["partition"] == "test" and row["locale"] == "ca-ES" for row in catalan
    )
    args.locale = "en-US"
    assert [row["id"] for row in catalan] == [
        row["id"] for row in model.load_rows(args)
    ]


def test_metrics_and_summary_aggregation(tmp_path):
    args = arguments(tmp_path)
    rows = [
        {"id": 1, "utt": "hola", "scenario": "general"},
        {"id": 2, "utt": "plou?", "scenario": "weather"},
    ]
    answers = [({"choice": "general"}, "jev", 10), ({"choice": "general"}, "jev", 30)]
    with patch.object(model, "predict", side_effect=answers):
        result = model.evaluate_model(args, rows, "jev", tmp_path / "predictions.jsonl")
    assert result["accuracy"] == 0.5
    assert result["macro_f1_18_labels"] == pytest.approx((2 / 3) / 18)
    assert result["mean_latency_ms"] == 20
    assert datetime.fromisoformat(result["evaluated_at"]).tzinfo == timezone.utc
    result["model"] = "jev"
    (tmp_path / "jev.json").write_text(json.dumps(result))
    rows = summarize_results.load_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["massive_accuracy"] == 0.5
    assert rows[0]["massive_macro_f1"] == round(result["macro_f1_18_labels"], 4)
    assert rows[0]["massive_decisions_per_sec"] == 50
    assert rows[0]["evaluated_at"] == result["evaluated_at"]
    assert (
        not {"dataset", "locale", "labels", "n", "seed", "shuffled_options"}
        & rows[0].keys()
    )


def test_skip_requires_matching_completed_configuration(tmp_path):
    args = arguments(tmp_path)
    output = run_evals.output_path(tmp_path, "org/jev:q4")
    result = {
        "model": "org/jev:q4",
        "locale": "ca-ES",
        "labels": "ca",
        "seed": 42,
        "shuffled_options": False,
        "requested_n_samples": 2,
        "data_source": str(args.data.resolve()),
        "accuracy": 0.5,
    }
    output.write_text(json.dumps(result))
    assert run_evals.completed(output, args, "org/jev:q4")
    args.limit = 0
    assert not run_evals.completed(output, args, "org/jev:q4")
    assert output != run_evals.output_path(tmp_path, "org_jev:q4")


def test_discovery_excludes_unrelated_models():
    with patch.object(
        model,
        "request_json",
        return_value={
            "data": [
                {"id": "ggml-org/OpenJev-GGUF:Q4_K_M"},
                {"id": "Julia-1"},
                {"id": "unrelated"},
                {"id": "clever-model"},
            ]
        },
    ):
        assert model.discover_models(
            argparse.Namespace(url="http://localhost:9090/v1/systemone")
        ) == [
            "Julia-1",
            "ggml-org/OpenJev-GGUF:Q4_K_M",
        ]


def test_failed_rerun_removes_completed_result(tmp_path):
    output = tmp_path / "jev.json"
    output.write_text('{"accuracy": 1}')
    with (
        patch("sys.argv", ["model.py", "--model", "jev", "--output", str(output)]),
        patch.object(model, "load_rows", side_effect=ValueError("invalid dataset")),
    ):
        with pytest.raises(SystemExit) as error:
            model.main()
    assert error.value.code == 1
    assert not output.exists()


def test_runner_continues_after_failed_model(tmp_path):
    with (
        patch(
            "sys.argv",
            ["run_evals.py", "--models", "jev", "julia", "--output-dir", str(tmp_path)],
        ),
        patch.object(
            run_evals.subprocess,
            "run",
            side_effect=[
                argparse.Namespace(returncode=1),
                argparse.Namespace(returncode=0),
            ],
        ) as execute,
    ):
        with pytest.raises(SystemExit) as error:
            run_evals.main()
    assert error.value.code == 1
    assert execute.call_count == 2
