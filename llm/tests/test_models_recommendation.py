import json
import subprocess
import sys

import pytest

from models_recommendation import (
    CATEGORIES,
    ROOT,
    evaluation_score,
    load_candidates,
    main,
    recommend,
    web_table,
)


def candidate(name, memory, score):
    return {"model_id": name, "model": name, "memory_gb": memory, "score": score}


def test_memory_boundary_reserve_and_score_direction():
    candidates = {
        "llm": [candidate("large", 6.01, 90), candidate("fits", 6, 60)],
        "embeddings": [candidate("good", 2, 0.9), candidate("bad", 1, 0.8)],
        "asr": [candidate("good", 3, 0.05), candidate("bad", 1, 0.2)],
    }
    result = recommend(candidates, [8])[0]
    assert result["budget_gb"] == 6
    assert result["models"]["llm"]["model"] == "fits"
    assert result["models"]["embeddings"]["model"] == "good"
    assert result["models"]["asr"]["model"] == "good"
    assert recommend(candidates, [8], 0)[0]["models"]["llm"]["model"] == "large"


def test_ties_prefer_smaller_model_and_empty_categories_are_explicit():
    candidates = {category: [] for category in CATEGORIES}
    candidates["llm"] = [candidate("large", 5, 60), candidate("small", 3, 60)]
    result = recommend(candidates, [8])[0]["models"]
    assert result["llm"]["model"] == "small"
    assert result["embeddings"] is None
    assert result["asr"] is None
    assert recommend(candidates, [1])[0]["models"]["llm"] is None
    assert recommend(candidates, [1])[0]["llm_alternatives"] == []


def test_llm_alternatives_use_strict_gap_from_best_and_fit_memory():
    candidates = {category: [] for category in CATEGORIES}
    candidates["llm"] = [
        candidate("touches_boundary", 4, 58),
        candidate("outside", 2, 57.99),
        candidate("best", 6, 60),
        candidate("over_budget", 6.01, 61),
        candidate("close", 5, 58.01),
        candidate("chained_overlap", 2, 57),
    ]
    config = recommend(candidates, [8])[0]
    assert config["models"]["llm"]["model"] == "best"
    alternatives = config["llm_alternatives"]
    assert [model["model"] for model in alternatives] == ["close"]
    assert [model["score_gap"] for model in alternatives] == pytest.approx([1.99])
    assert "score_gap" not in candidates["llm"][0]


def test_llm_uncertainty_is_configurable_and_zero_preserves_exact_ties():
    candidates = {category: [] for category in CATEGORIES}
    candidates["llm"] = [
        candidate("best", 3, 60),
        candidate("tied", 4, 60),
        candidate("near", 2, 58),
    ]
    strict = recommend(candidates, [8], llm_uncertainty=0)[0]
    assert [m["model"] for m in strict["llm_alternatives"]] == ["tied"]
    relaxed = recommend(candidates, [8], llm_uncertainty=3)[0]
    assert [m["model"] for m in relaxed["llm_alternatives"]] == ["tied", "near"]


@pytest.mark.parametrize("margin", [-1, float("nan"), float("inf")])
def test_invalid_llm_uncertainty(margin):
    with pytest.raises(ValueError, match="marge CLAM"):
        recommend({}, [8], llm_uncertainty=margin)


def test_table_shows_llm_alternatives_and_gap(capsys):
    main(["--memory", "16"])
    output = capsys.readouterr().out
    assert "menys de 2 punts" in output
    assert "no concloents" in output
    assert "LLM (semblant)" in output
    assert "Δ CLAM" in output
    assert "gemma3-12b" in output


@pytest.mark.parametrize("capacity", [0, -8, float("nan"), float("inf")])
@pytest.mark.parametrize("generate", [recommend, web_table])
def test_invalid_memory(capacity, generate):
    with pytest.raises(ValueError, match="capacitats"):
        generate({}, [capacity])


@pytest.mark.parametrize("reserve", [-1, 100, float("nan"), float("inf")])
@pytest.mark.parametrize("generate", [recommend, web_table])
def test_invalid_reserve(reserve, generate):
    with pytest.raises(ValueError, match="reserva"):
        generate({}, [8], reserve)


def test_asr_weights_samples_and_requires_both_benchmarks():
    data = {
        "benchmarks": {
            "fleurs_ca": {"wer": 0.1, "n": 100},
            "openslr69_ca": {"wer": 0.3, "n": 300},
        }
    }
    assert evaluation_score("asr", data) == pytest.approx(0.25)
    data["benchmarks"]["openslr69_ca"]["n"] = 0
    assert evaluation_score("asr", data) is None
    del data["benchmarks"]["openslr69_ca"]
    assert evaluation_score("asr", data) is None


@pytest.mark.parametrize("category", CATEGORIES)
def test_missing_benchmarks_are_not_ranked(category):
    assert evaluation_score(category, {"benchmarks": {}}) is None


def test_loader_uses_raw_evals_filters_cloud_and_reports_unknown_memory(tmp_path):
    for category in CATEGORIES:
        (tmp_path / category / "evals").mkdir(parents=True)
    data = {
        "model": "intfloat/multilingual-e5-large",
        "display_name": "local",
        "benchmarks": {
            "xquad_ca_retrieval": {"ndcg_at_10": 0.9},
            "sts_ca": {"spearman": 0.8},
            "tecla_classification": {"macro_f1": 0.7},
        },
    }
    directory = tmp_path / "embeddings" / "evals"
    (directory / "local.json").write_text(json.dumps(data))
    (directory / "cloud.json").write_text(json.dumps({**data, "cloud": True}))
    (directory / "unknown.json").write_text(
        json.dumps({**data, "model": "org/unknown", "display_name": "unknown"})
    )
    (tmp_path / "embeddings" / "embeddings.json").write_text("not used")
    candidates, skipped = load_candidates(tmp_path)
    assert len(candidates["embeddings"]) == 1
    model = candidates["embeddings"][0]
    assert model["score"] == pytest.approx(0.8)
    assert model["memory_gb"] == 2.24
    assert model["eval_source"] == "embeddings/evals/local.json"
    assert [row["model"] for row in skipped] == ["unknown"]


def test_cli_runs_from_another_directory_and_emits_json(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "models_recommendation.py"), "--format", "json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["llm_uncertainty_points"] == 2
    assert [r["capacity_gb"] for r in report["configurations"]] == [4, 8, 16, 32]
    for config in report["configurations"]:
        for model in config["models"].values():
            assert model is not None
            assert model["memory_gb"] <= config["budget_gb"]
            assert (ROOT / model["eval_source"]).is_file()
        for alternative in config["llm_alternatives"]:
            assert alternative["memory_gb"] <= config["budget_gb"]
            assert 0 <= alternative["score_gap"] < 2


@pytest.mark.parametrize(
    "filename, flags",
    [
        ("gemma3_12b_q2.json", {}),
        ("gemma3_12b_q2.json", {"quantized_analysis_only": False}),
        ("custom_analysis.json", {"quantized_analysis_only": True}),
    ],
)
def test_loader_excludes_analysis_only_models(tmp_path, filename, flags):
    for category in CATEGORIES:
        (tmp_path / category / "evals").mkdir(parents=True)
    data = json.loads((ROOT / "llm/evals/gemma3_12b_q4.json").read_text())
    directory = tmp_path / "llm/evals"
    (directory / "gemma3_12b_q4.json").write_text(json.dumps(data))
    (directory / filename).write_text(
        json.dumps({**data, "display_name": "analysis-only", **flags})
    )
    candidates, _ = load_candidates(tmp_path)
    assert [m["model"] for m in candidates["llm"]] == [data["display_name"]]


def test_missing_evaluation_directory_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="directori"):
        load_candidates(tmp_path)


@pytest.mark.parametrize("format", ["json", "web-json"])
def test_json_output(tmp_path, capsys, format):
    args = ["--memory", "8", "16", "32", "--format", format]
    main(args)
    expected = json.loads(capsys.readouterr().out)
    output = tmp_path / "llms_recommendations.json"
    main([*args, "--output", str(output)])
    assert capsys.readouterr().out == ""
    table = json.loads(output.read_text())
    assert table == expected
    if format == "json":
        return
    assert set(table) == {"text", "data"}
    assert list(table["text"].items()) == [
        ("capacity_gb", "Memòria de l’ordinador"),
        ("recommended", "Model recomanat"),
        ("alternatives", "Alternativa"),
    ]
    assert [row["capacity_gb"] for row in table["data"]] == [8, 16, 32]
    for row in table["data"]:
        assert row.keys() == table["text"].keys()
        assert " · " in row["recommended"]
        assert " · " in row["alternatives"]
        assert ";" not in row["alternatives"]


def test_web_alternative_is_second_best_eligible_llm_regardless_of_gap():
    models = [
        candidate("over_budget", 6.01, 100),
        candidate("third", 2, 49),
        candidate("second", 3, 50),
        candidate("best", 6, 60),
    ]
    rows = web_table({"llm": models}, [8, 4, 3, 1])["data"]
    assert [(row["recommended"], row["alternatives"]) for row in rows] == [
        ("best", "second"),
        ("second", "third"),
        ("third", None),
        (None, None),
    ]


def test_web_alternative_breaks_score_ties_by_memory_then_model_id():
    models = [
        candidate("larger", 4, 60),
        candidate("b", 3, 60),
        candidate("a", 3, 60),
    ]
    assert web_table({"llm": models}, [8])["data"] == [
        {"capacity_gb": 8, "recommended": "a", "alternatives": "b"}
    ]
