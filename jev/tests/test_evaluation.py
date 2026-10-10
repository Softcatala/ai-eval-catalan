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


def test_teca_labels_sampling_and_scores(args, tmp_path):
    args.dataset = "teca"
    examples = [
        {"premise": "Premissa", "hypothesis": "Hipòtesi", "label": label}
        for label in range(3)
    ]
    args.data.write_text("\n".join(json.dumps(row) for row in examples))
    rows = model.load_rows(args)
    assert {r["scenario"] for r in rows} == set(model.TECA_LABELS)
    assert all("Premissa: Premissa\nHipòtesi: Hipòtesi" == r["utt"] for r in rows)
    assert rows == model.load_rows(args)
    answers = [({"choice": r["scenario"]}, "jev", 10) for r in rows]
    with patch.object(model, "predict", side_effect=answers):
        result = model.evaluate_model(args, rows, "jev", tmp_path / "teca.jsonl")
    assert result["dataset"] == "Tornem a TE-ca"
    assert result["accuracy"] == result["macro_f1_3_labels"] == 1.0
    assert "macro_f1_18_labels" not in result
    args.limit = 2
    assert model.load_rows(args) == rows[:2]


@pytest.mark.parametrize("provider", ["systemone", "openai"])
def test_teca_request_uses_entailment_choices(args, provider):
    args.dataset = "teca"
    args.provider = provider
    args.url = "https://example.test"
    args.timeout = 1
    answer = {"type": "choice", "choice": "neutral", "probabilities": []}
    response = (
        {"answers": [{"name": "scenario", **answer}]}
        if provider == "openai"
        else {"answers": {"scenario": answer}}
    )
    with patch.object(model, "request_json", return_value=response) as request:
        prediction, _, _ = model.predict(
            args, {"utt": "Premissa: P\nHipòtesi: H"}, 0, "jev"
        )
    assert prediction["choice"] == "neutral"
    payload = request.call_args.args[2]
    question = (
        payload["questions"][0]
        if provider == "openai"
        else payload["questions"]["scenario"]
    )
    assert "premissa" in question["instructions"]
    choices = (
        {c["value"] for c in question["choices"]}
        if provider == "openai"
        else set(question["criteria"])
    )
    assert choices == set(model.TECA_LABELS)


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
            "teca_n": None,
            "teca_accuracy": None,
            "teca_macro_f1": None,
            "average_accuracy": None,
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


@pytest.mark.parametrize("embedded", [False, True])
def test_average_ranking_requires_both_full_datasets(tmp_path, embedded):
    teca_dir = tmp_path / "teca_full"
    teca_dir.mkdir()
    common = {
        "locale": "ca-ES",
        "labels": "ca",
        "seed": 42,
        "shuffled_options": False,
        "requested_n_samples": 0,
        "mean_latency_ms": 10,
    }
    for name, massive_accuracy, teca_accuracy in [
        ("Kev-4B-GGUF", 0.9, 0.5),
        ("Kev-9B-GGUF", 0.8, 0.8),
        ("lev-GGUF", 0.99, None),
    ]:
        massive = {
            **common,
            "model": name,
            "dataset": "MASSIVE 1.1",
            "n": 2974,
            "accuracy": massive_accuracy,
            "macro_f1_18_labels": massive_accuracy,
        }
        (tmp_path / f"{name}.json").write_text(json.dumps(massive))
        if teca_accuracy is not None:
            teca = {
                **common,
                "model": name,
                "dataset": "Tornem a TE-ca",
                "data_source": "projecte-aina/teca:test",
                "n": 2117,
                "accuracy": teca_accuracy,
                "macro_f1_3_labels": teca_accuracy,
            }
            if embedded:
                massive["benchmarks"] = {"teca": teca}
                (tmp_path / f"{name}.json").write_text(json.dumps(massive))
            else:
                (teca_dir / f"{name}.json").write_text(json.dumps(teca))
    rows = summarize_results.load_rows(tmp_path)
    assert [r["model"] for r in rows] == ["Kev 9B", "Kev 4B", "lev 4B"]
    assert [r["average_accuracy"] for r in rows] == [0.8, 0.7, None]
    assert rows[0]["teca_n"] == 2117
    path = (tmp_path if embedded else teca_dir) / "Kev-4B-GGUF.json"
    result = json.loads(path.read_text())
    (result["benchmarks"]["teca"] if embedded else result)["n"] = 200
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="n must be 2117"):
        summarize_results.load_rows(tmp_path)


@pytest.mark.parametrize("dataset", ["massive", "teca"])
def test_rerun_preserves_other_benchmark(tmp_path, dataset):
    output = tmp_path / "jev.json"
    original = {
        "dataset": "MASSIVE 1.1",
        "model": "jev",
        "accuracy": 0.8,
        "n": 2974,
        "benchmarks": {
            "teca": {"dataset": "Tornem a TE-ca", "accuracy": 0.7, "n": 2117}
        },
    }
    output.write_text(json.dumps(original))
    summary = {
        "dataset": "MASSIVE 1.1" if dataset == "massive" else "Tornem a TE-ca",
        "models": ["jev"],
        "accuracy": 0.9,
        "n": 2974 if dataset == "massive" else 2117,
        "macro_f1_18_labels" if dataset == "massive" else "macro_f1_3_labels": 0.9,
        "mean_latency_ms": 10,
        "locale": "ca-ES",
        "labels": "ca",
        "seed": 42,
        "shuffled_options": False,
    }
    with (
        patch(
            "sys.argv",
            [
                "model.py",
                "--model",
                "jev",
                "--dataset",
                dataset,
                "--output",
                str(output),
            ],
        ),
        patch.object(model, "load_rows", return_value=[]),
        patch.object(model, "evaluate_model", return_value=summary),
    ):
        model.main()
    stored = json.loads(output.read_text())
    if dataset == "teca":
        assert stored["benchmarks"]["massive"] == {
            k: v for k, v in original.items() if k != "benchmarks"
        }
        assert stored["benchmarks"]["teca"]["accuracy"] == 0.9
        parser = argparse.ArgumentParser()
        model.add_evaluation_arguments(parser)
        args = parser.parse_args(["--dataset", "teca"])
        assert run_evals.completed(output, args, "jev")
    else:
        assert stored["benchmarks"]["teca"] == original["benchmarks"]["teca"]
        assert stored["benchmarks"]["massive"]["accuracy"] == 0.9


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
        "seed": 42,
        "shuffled_options": False,
        "requested_n_samples": 2,
        "data_source": str(args.data.resolve()),
        "accuracy": 0.5,
    }
    output.write_text(json.dumps(result))
    assert run_evals.completed(output, args, "org/jev:q4")
    args.labels = "en"
    assert not run_evals.completed(output, args, "org/jev:q4")
    args.labels = "ca"
    args.locale = "en-US"
    assert not run_evals.completed(output, args, "org/jev:q4")
    args.locale = "ca-ES"
    args.limit = 0
    assert not run_evals.completed(output, args, "org/jev:q4")


@pytest.mark.parametrize(
    "locale, labels", [("ca-ES", "ca"), ("en-US", "ca"), ("ca-ES", "en")]
)
def test_result_omits_only_default_language_fields(locale, labels):
    summary = {
        "model": "jev",
        "dataset": "MASSIVE 1.1",
        "accuracy": 0.8,
        "locale": locale,
        "labels": labels,
    }
    stored = model.result_document({}, summary)["benchmarks"]["massive"]
    assert ("locale" in stored) == (locale != "ca-ES")
    assert ("labels" in stored) == (labels != "ca")
    assert stored.get("locale", "ca-ES") == locale
    assert stored.get("labels", "ca") == labels
    assert summary["locale"] == locale and summary["labels"] == labels


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


def test_failed_teca_rerun_does_not_publish_archived_score(tmp_path):
    output = tmp_path / "jev.json"
    massive = {
        "model": "jev",
        "dataset": "MASSIVE 1.1",
        "accuracy": 0.8,
        "macro_f1_18_labels": 0.8,
        "n": 2974,
        "requested_n_samples": 0,
        "locale": "ca-ES",
        "labels": "ca",
        "seed": 42,
        "shuffled_options": False,
        "mean_latency_ms": 10,
    }
    teca = {
        **massive,
        "dataset": "Tornem a TE-ca",
        "data_source": "projecte-aina/teca:test",
        "n": 2117,
        "accuracy": 0.7,
        "macro_f1_3_labels": 0.7,
    }
    output.write_text(
        json.dumps({"model": "jev", "benchmarks": {"massive": massive, "teca": teca}})
    )
    archive = tmp_path / "teca_full"
    archive.mkdir()
    (archive / output.name).write_text(json.dumps(teca))
    with (
        patch(
            "sys.argv",
            [
                "model.py",
                "--model",
                "jev",
                "--dataset",
                "teca",
                "--output",
                str(output),
            ],
        ),
        patch.object(model, "load_rows", side_effect=ValueError("dataset unavailable")),
        pytest.raises(SystemExit) as error,
    ):
        model.main()
    assert error.value.code == 1
    assert json.loads(output.read_text())["benchmarks"] == {
        "massive": {k: v for k, v in massive.items() if k not in ("locale", "labels")}
    }
    row = summarize_results.load_rows(tmp_path)[0]
    assert row["massive_accuracy"] == 0.8
    assert row["teca_accuracy"] is None
    assert row["average_accuracy"] is None


@pytest.mark.parametrize("contents", ["", "{"])
def test_corrupt_result_can_be_rerun(tmp_path, contents):
    output = tmp_path / "jev.json"
    output.write_text(contents)
    summary = {"dataset": "MASSIVE 1.1", "accuracy": 1.0, "n": 1}
    with (
        patch("sys.argv", ["model.py", "--model", "jev", "--output", str(output)]),
        patch.object(model, "load_rows", return_value=[]) as load,
        patch.object(model, "evaluate_model", return_value=summary),
        patch.object(model, "print_summary"),
    ):
        model.main()
    load.assert_called_once()
    stored = json.loads(output.read_text())
    assert stored["model"] == "jev"
    assert stored["benchmarks"]["massive"]["accuracy"] == 1.0


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
