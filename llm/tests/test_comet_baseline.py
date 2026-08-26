from comet_baseline import deranged_permutation


def test_deranged_permutation_is_fixed_seed_random_mismatch():
    first = deranged_permutation(20, seed=1714, index=0)
    second = deranged_permutation(20, seed=1714, index=0)
    adjacent_shift = [(index + 1) % 20 for index in range(20)]

    assert first == second
    assert sorted(first) == list(range(20))
    assert all(source != target for source, target in enumerate(first))
    assert first != adjacent_shift


def test_deranged_permutation_changes_with_permutation_index():
    assert deranged_permutation(20, seed=1714, index=0) != deranged_permutation(
        20, seed=1714, index=1
    )
