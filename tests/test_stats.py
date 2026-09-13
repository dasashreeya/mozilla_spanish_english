from lit_sp_en.stats import bucket_counts, describe


def test_describe_basic() -> None:
    summary = describe([1, 2, 3, 4])
    assert summary["count"] == 4
    assert summary["sum"] == 10
    assert summary["min"] == 1 and summary["max"] == 4
    assert summary["p50"] == 2.5


def test_describe_ignores_none_and_nan() -> None:
    assert describe([None, float("nan"), 5])["count"] == 1


def test_describe_empty() -> None:
    summary = describe([])
    assert summary["count"] == 0 and summary["p50"] is None


def test_bucket_counts_edges_are_half_open() -> None:
    counts = bucket_counts([0.5, 1, 2.9, 3, 100], [1, 3])
    assert counts == {"<1": 1, "1-3": 2, ">=3": 2}
