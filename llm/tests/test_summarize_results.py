from summarize_results import extract_metrics, normalize_score


def test_extract_metrics_uses_comet_for_flores():
    data = {
        "benchmarks": {
            "flores": {
                "catalan_bench_flores_en-ca": {"comet,none": 0.8},
                "catalan_bench_flores_ca-en": {"comet,none": 0.6},
            }
        }
    }

    metrics = extract_metrics(data)

    assert metrics["flores_en2ca"] == 0.8
    assert metrics["flores_ca2en"] == 0.6
    assert metrics["flores_en_ca"] == 0.7


def test_comet_is_not_divided_by_100_during_normalization():
    assert normalize_score("flores_en_ca", 0.75) == 0.75
