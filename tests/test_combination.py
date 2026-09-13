import pytest

from lit_sp_en.combination import edit_distance, mbr_select


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [("a b c", "a b c", 0), ("a b c", "a x c", 1), ("a b", "a b c d", 2), ("", "a b", 2),
     ("a b", "", 2), ("", "", 0)],
)
def test_edit_distance(a: str, b: str, distance: int) -> None:
    assert edit_distance(a, b) == distance
    assert edit_distance(b, a) == distance


def test_mbr_picks_the_consensus_hypothesis() -> None:
    candidates = {
        "x": ["the cat sat", "hola amigo"],
        "y": ["the cat sat", "hola amigos que tal"],
        "z": ["a dog ran", "hola amigo"],
    }
    selected, chosen = mbr_select(candidates)
    assert selected == ["the cat sat", "hola amigo"]
    assert chosen == ["x", "x"]


def test_mbr_ties_go_to_declared_order() -> None:
    candidates = {"late": ["a"], "early": ["b"], "third": ["c"]}
    assert mbr_select(candidates)[1] == ["late"]
    reordered = {"early": ["b"], "late": ["a"], "third": ["c"]}
    assert mbr_select(reordered)[1] == ["early"]


def test_mbr_uses_official_normalization() -> None:
    # Punctuation and sentence-initial case are normalized away before comparing.
    candidates = {"x": ["Hola, amigo."], "y": ["hola amigo"], "z": ["adios"]}
    selected, chosen = mbr_select(candidates)
    assert chosen == ["x"] and selected == ["Hola, amigo."]


def test_mbr_requires_three_systems_and_equal_lengths() -> None:
    with pytest.raises(ValueError, match="three"):
        mbr_select({"x": ["a"], "y": ["b"]})
    with pytest.raises(ValueError, match="same clips"):
        mbr_select({"x": ["a"], "y": ["b"], "z": ["c", "d"]})
