from comet_baseline import TASKS


def test_baseline_covers_the_four_flores_directions():
    assert set(TASKS) == {
        "catalan_bench_flores_en-ca",
        "catalan_bench_flores_ca-en",
        "catalan_bench_flores_es-ca",
        "catalan_bench_flores_ca-es",
    }
