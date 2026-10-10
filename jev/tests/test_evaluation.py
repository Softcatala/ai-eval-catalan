import argparse
from datetime import datetime, timezone
import json
from unittest.mock import patch

import pytest

from jev import model, run_evals, summarize_results


@pytest.fixture
def args(tmp_path):
    parser = argparse.ArgumentParser()
    model.add_evaluation_arguments(parser)
    args = parser.parse_args([])
    args.data = tmp_path / "ca.jsonl"
    return args


def test_test_split_sampling_and_parallel_ids(args):
    rows = [
        {"id": i, "partition": partition, "locale": locale}
        for i in range(5)
        for partition in ("train", "test")
        for locale in ("ca-ES", "en-US")
    ]
    args.data.write_text("\n".join(json.dumps(row) for row in rows))
    assert args.limit == 0
    assert len(model.load_rows(args)) == 5
    args.limit = 2
    catalan = model.load_rows(args)
    assert len(catalan) == 2
    assert all(
        row["partition"] == "test" and row["locale"] == "ca-ES" for row in catalan
    )
    args.locale = "en-US"
    assert [row["id"] for row in catalan] == [
        row["id"] for row in model.load_rows(args)
    ]


@pytest.mark.parametrize(
    "model_id, display_name, expected",
    [
        ("ggml-org/Clef-Flash-GGUF:Q8_0", None, "Clef Flash 9B"),
        ("Clef-Flash-GGUF", "Clef-Flash-GGUF", "Clef Flash 9B"),
        ("Clef-Flash-GGUF", "Clef Flash 9B", "Clef Flash 9B"),
    ],
)
def test_metrics_and_summary_aggregation(
    args, tmp_path, model_id, display_name, expected
):
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
    result.update(model=model_id, display_name=display_name)
    (tmp_path / "jev.json").write_text(json.dumps(result))
    rows = summarize_results.load_rows(tmp_path)
    assert rows == [
        {
            "model": expected,
            "repo_url": "https://huggingface.co/ggml-org/Clef-Flash-GGUF",
            "cloud": False,
            "evaluated_at": result["evaluated_at"],
            "params_b": 9.075566084,
            "memory_gb": 9.7,
            "n": 2,
            "massive_accuracy": 0.5,
            "massive_macro_f1": 0.037,
            "massive_decisions_per_sec": 50.0,
        }
    ]
    output = tmp_path / "jevs.json"
    with patch(
        "sys.argv",
        [
            "summarize_results.py",
            "--results-dir",
            str(tmp_path),
            "--json-out",
            str(output),
        ],
    ):
        summarize_results.main()
    published = json.loads(output.read_text())
    assert published["data"] == rows


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("Julia-1-GGUF", "Julia 1 144M"),
        ("Laya-GGUF", "Laya 421M"),
        ("Kev-4B-GGUF", "Kev 4B"),
        ("Kev-9B-GGUF", "Kev 9B"),
        ("lev-GGUF", "lev 4B"),
        ("OpenJev-GGUF", "OpenJev 27B"),
        ("Clef-GGUF", "Clef 27B"),
        ("Bespoke-Nimble-9B-v3-GGUF", "Nimble 9B"),
    ],
)
def test_model_labels_include_parameter_count(model_id, expected):
    assert model.model_display_name(model_id) == expected


def test_skip_requires_matching_completed_configuration(args, tmp_path):
    args.limit = 2
    output = run_evals.output_path(tmp_path, "org/jev:q4")
    # Per-run cache metadata; these fields are not published in jevs.json.
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


def test_configured_output_filenames(tmp_path):
    assert run_evals.output_path(tmp_path, "Julia-1-GGUF") == tmp_path / "julia_1.json"
    assert run_evals.output_path(tmp_path, "org/jev:q4") == tmp_path / "org_jev_q4.json"


def test_runner_rejects_colliding_output_filenames(tmp_path):
    with (
        patch("sys.argv", ["run_evals.py", "--models", "org/jev:q4", "org_jev:q4"]),
        patch.object(run_evals.subprocess, "run") as execute,
        pytest.raises(SystemExit) as error,
    ):
        run_evals.main()
    assert error.value.code == 2
    execute.assert_not_called()


def test_discovery_excludes_unrelated_models():
    expected = [
        "Bespoke-Nimble-9B-v3-GGUF",
        "Clef-Flash-GGUF",
        "Clef-GGUF",
        "Julia-1",
        "LiquidAI/d1-3B-GGUF:Q8_0",
        "LiquidAI/d1-omni-600M-GGUF:Q8_0",
        "ggml-org/Clef-Flash-GGUF:Q8_0",
        "ggml-org/Kev-0.8B-GGUF:Q8_0",
        "ggml-org/Kev-9B-GGUF:Q8_0",
        "ggml-org/OpenJev-GGUF:Q8_0",
        "owao/surogate-rune-26b-a4b-GGUF:Q8_0",
    ]
    ids = list(reversed(expected)) + ["unrelated", "clever-model"]
    with patch.object(
        model,
        "request_json",
        return_value={"data": [{"id": name} for name in ids]},
    ):
        assert (
            model.discover_models(
                argparse.Namespace(url="http://localhost:9090/v1/systemone")
            )
            == expected
        )


@pytest.mark.parametrize("refused", [False, True])
def test_openai_decisions_translation(args, refused):
    args.provider = "openai"
    args.url = "https://api.openai.com/v1/decisions"
    answer = (
        {"name": "scenario", "type": "refusal"}
        if refused
        else {
            "name": "scenario",
            "type": "choice",
            "choice": "general",
            "probabilities": [{"value": "general", "probability": 1.0}],
        }
    )
    with patch.object(
        model,
        "request_json",
        return_value={
            "model": "gpt-6-luna",
            "answers": [answer],
        },
    ) as request:
        result, model_id, elapsed = model.predict(
            args, {"utt": "hola"}, 0, "gpt-6-luna"
        )
    payload = request.call_args.args[2]
    assert payload["input"] == "hola"
    assert payload["questions"][0]["name"] == "scenario"
    assert payload["questions"][0]["choices"] == [
        {"value": key, "description": desc[0]} for key, desc in model.LABELS.items()
    ]
    assert result["choice"] == (None if refused else "general")
    assert result.get("probabilities") == (None if refused else {"general": 1.0})
    assert model_id == "gpt-6-luna"
    assert elapsed >= 0


def test_failed_rerun_removes_completed_result(tmp_path):
    output = tmp_path / "jev.json"
    output.write_text('{"accuracy": 1}')
    with (
        patch("sys.argv", ["model.py", "--model", "jev", "--output", str(output)]),
        patch.object(model, "load_rows", side_effect=ValueError("invalid dataset")),
        pytest.raises(SystemExit) as error,
    ):
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
        pytest.raises(SystemExit) as error,
    ):
        run_evals.main()
    assert error.value.code == 1
    assert execute.call_count == 2
