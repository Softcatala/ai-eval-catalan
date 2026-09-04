from summarize_results import (
    CLAM_TASKS,
    COMET_SOURCE_COPY_BASELINES,
    clam_score,
    extract_metrics,
    normalize_score,
)


def test_extract_metrics_uses_comet_for_flores():
    data = {
        "benchmarks": {
            "flores": {
                "catalan_bench_flores_en-ca": {"comet": 0.8},
                "catalan_bench_flores_ca-en": {"comet": 0.6},
            }
        }
    }

    metrics = extract_metrics(data)

    assert metrics["flores_en2ca"] == 0.8
    assert metrics["flores_ca2en"] == 0.6
    assert metrics["flores_en_ca"] == 0.7


def test_comet_uses_the_measured_source_copy_baseline():
    baseline = COMET_SOURCE_COPY_BASELINES["flores_en_ca"]
    assert normalize_score("flores_en_ca", baseline) == 0.0
    assert normalize_score("flores_en_ca", 0.75) == (0.75 - baseline) / (1 - baseline)


def test_extract_metrics_adds_lm_eval_catalan_drift_pass_rate():
    metrics = extract_metrics(
        {"benchmarks": {"catalan_drift": {"drift_pass,none": 0.7133}}}
    )

    assert metrics["catalan_drift_pass_rate"] == 0.7133
    assert "catalan_drift_pass_rate" in CLAM_TASKS
    assert clam_score(metrics) == 71.33
