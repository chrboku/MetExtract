from src.metaboliteGrouping import split_group_by_relative_abundance


def _normalize_groups(groups):
    return sorted([sorted(group) for group in groups])


def test_split_group_by_relative_abundance_splits_dissimilar_profiles():
    groups = split_group_by_relative_abundance(
        [1, 2, 3, 4],
        {
            1: [100, 200, 300],
            2: [50, 100, 150],
            3: [300, 200, 100],
            4: [150, 100, 50],
        },
        min_peak_correlation=0.8,
        min_connection_rate=0.6,
    )

    assert _normalize_groups(groups) == [[1, 2], [3, 4]]


def test_split_group_by_relative_abundance_keeps_group_for_missing_vectors():
    groups = split_group_by_relative_abundance(
        [10, 11, 12],
        {
            10: [100, 200, 300],
            11: [120, 220, 320],
        },
        min_peak_correlation=0.8,
        min_connection_rate=0.6,
    )

    assert groups == [[10, 11, 12]]


def test_split_group_by_relative_abundance_respects_similarity_threshold():
    abundance_vectors = {
        1: [100, 200, 300, 400],
        2: [90, 210, 280, 420],
        3: [400, 300, 200, 100],
    }

    relaxed_groups = split_group_by_relative_abundance(
        [1, 2, 3],
        abundance_vectors,
        min_peak_correlation=-1.0,
        min_connection_rate=0.6,
    )
    strict_groups = split_group_by_relative_abundance(
        [1, 2, 3],
        abundance_vectors,
        min_peak_correlation=0.8,
        min_connection_rate=0.6,
    )

    assert _normalize_groups(relaxed_groups) == [[1, 2, 3]]
    assert _normalize_groups(strict_groups) == [[1, 2], [3]]


def test_split_group_by_relative_abundance_ignores_mismatched_zero_presence():
    groups = split_group_by_relative_abundance(
        [1, 2, 3],
        {
            1: [10, 0, 11, 0],
            2: [9, 0, 12, 0],
            3: [0, 20, 0, 18],
        },
        min_peak_correlation=0.7,
        min_connection_rate=0.6,
    )

    assert _normalize_groups(groups) == [[1, 2], [3]]


def test_split_group_by_relative_abundance_keeps_constant_proportional_profiles_together():
    groups = split_group_by_relative_abundance(
        [1, 2, 3],
        {
            1: [10, 10, 0, 0],
            2: [20, 20, 0, 0],
            3: [0, 0, 30, 30],
        },
        min_peak_correlation=0.8,
        min_connection_rate=0.6,
    )

    assert _normalize_groups(groups) == [[1, 2], [3]]
