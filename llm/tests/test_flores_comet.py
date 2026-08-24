import model


def test_comet_example_extracts_logged_flores_sample():
    sample = {
        "doc": {
            "sentence_eng_Latn": "The source sentence.",
            "sentence_cat_Latn": "La frase de referència.",
        },
        "target": "La frase de referència.",
        "filtered_resps": [["La traducció generada."]],
    }

    assert model._comet_example("catalan_bench_flores_en-ca", sample) == {
        "src": "The source sentence.",
        "ref": "La frase de referència.",
        "mt": "La traducció generada.",
    }


def test_comet_example_rejects_incomplete_sample():
    assert (
        model._comet_example(
            "catalan_bench_flores_ca-es",
            {"doc": {"sentence_cat_Latn": "Origen"}, "target": "Referència"},
        )
        is None
    )


def test_comet_uses_only_usable_samples_for_count():
    task = "catalan_bench_flores_en-ca"
    results = {
        "results": {task: {"bleu,none": 42}},
        "samples": {
            task: [
                {
                    "doc": {"sentence_eng_Latn": "Source"},
                    "target": "Referència",
                    "filtered_resps": [["Traducció"]],
                },
                {"doc": {"sentence_eng_Latn": "Incomplete"}},
            ]
        },
    }

    class Comet:
        def predict(self, examples, **kwargs):
            assert len(examples) == 1
            return type("Prediction", (), {"system_score": 0.75})()

    score = model._score_flores_comet(results, [task], Comet())[task]

    assert score["n"] == 1
    assert score["comet"] == 0.75
    assert "bleu,none" not in score


def test_comet_error_is_limited_to_its_flores_direction():
    good_task = "catalan_bench_flores_en-ca"
    empty_task = "catalan_bench_flores_ca-en"
    results = {
        "results": {good_task: {}, empty_task: {}},
        "samples": {
            good_task: [
                {
                    "doc": {"sentence_eng_Latn": "Source"},
                    "target": "Referència",
                    "filtered_resps": [["Traducció"]],
                }
            ],
            empty_task: [],
        },
    }

    class Comet:
        def predict(self, examples, **kwargs):
            return type("Prediction", (), {"system_score": 0.75})()

    scores = model._score_flores_comet(results, [good_task, empty_task], Comet())

    assert scores[good_task]["comet"] == 0.75
    assert (
        scores[empty_task]["error"]
        == "lm-eval returned no usable translations for COMET"
    )
