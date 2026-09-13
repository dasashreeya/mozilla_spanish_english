import pytest

from lit_sp_en import evaluation
from lit_sp_en.transcripts import official_wer

REFS = ["Yo fui a la store. Then I left", "I don... I don't know", "Okay so Maria said"]
HYPS = ["yo fui a la tienda then I left", "I don I don't know", ""]


def test_official_wer_matches_vendored_scorer() -> None:
    report = evaluation.evaluate(REFS, HYPS)
    assert report["official"]["wer"] == round(official_wer(REFS, HYPS), 4)


def test_error_breakdown_sums() -> None:
    official = evaluation.evaluate(REFS, HYPS)["official"]
    errors = official["substitutions"] + official["deletions"] + official["insertions"]
    assert official["wer"] == round(errors / official["reference_words"], 4)
    assert official["clips"] == 3


def test_groups_partition_the_corpus() -> None:
    report = evaluation.evaluate(REFS, HYPS, {"language": ["enspa", "enspa", "spa"]})
    groups = report["groups"]["language"]
    assert groups["spa"]["wer"] == 1.0
    assert groups["enspa"]["reference_words"] + groups["spa"]["reference_words"] == (
        report["official"]["reference_words"])


def test_bootstrap_is_deterministic_and_brackets_point() -> None:
    counts = evaluation.per_clip_counts(REFS * 10, HYPS * 10)
    first = evaluation.bootstrap_interval(counts, reps=500, seed=3)
    assert first == evaluation.bootstrap_interval(counts, reps=500, seed=3)
    point = evaluation.evaluate(REFS * 10, HYPS * 10)["official"]["wer"]
    assert first["low"] <= point <= first["high"]


def test_formatting_blind_diagnostic_ignores_case_and_ellipsis() -> None:
    report = evaluation.evaluate(["I don... Know Maria"], ["i don know maria"])
    assert report["official"]["wer"] > 0
    assert report["diagnostics"]["wer_lowercased_no_apostrophe_no_ellipsis"] == 0.0


@pytest.mark.parametrize(("hyp", "loops"), [("a b c a b c a b c a b c", 1), ("a b c d", 0)])
def test_loop_detector(hyp: str, loops: int) -> None:
    report = evaluation.evaluate(["a b c d e"], [hyp])
    assert report["diagnostics"]["hypotheses_with_repeated_trigram_x4"] == loops


def test_paired_bootstrap_detects_consistent_gap() -> None:
    refs = ["a b c d"] * 30
    better = evaluation.per_clip_counts(refs, ["a b c d"] * 30)
    worse = evaluation.per_clip_counts(refs, ["a b c x"] * 30)
    result = evaluation.paired_bootstrap(better, worse, reps=300)
    assert result["difference_b_minus_a"] == 0.25
    assert result["low"] == result["high"] == 0.25
    assert result["share_of_resamples_b_worse"] == 1.0


def test_paired_bootstrap_requires_same_clips() -> None:
    with pytest.raises(ValueError):
        evaluation.paired_bootstrap(evaluation.per_clip_counts(["a"], ["a"]), [])


def test_oracle_selection_takes_per_clip_minimum() -> None:
    refs = ["a b", "c d"]
    runs = {
        "x": evaluation.per_clip_counts(refs, ["a b", "z z"]),
        "y": evaluation.per_clip_counts(refs, ["z z", "c d"]),
    }
    result = evaluation.oracle_selection(runs)
    assert result == {"oracle_wer": 0.0, "clips_won": {"x": 1, "y": 1}}


def test_empty_hypotheses_counted() -> None:
    assert evaluation.evaluate(REFS, HYPS)["diagnostics"]["empty_hypotheses"] == 1
