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
