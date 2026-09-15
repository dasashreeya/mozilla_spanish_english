import ast
import random
from pathlib import Path

import pytest

from lit_sp_en import combination
from lit_sp_en.combination import edit_distance
from lit_sp_en.transcripts import normalize_text as vendored_normalize
from submission_src import selection

ROOT = Path(__file__).resolve().parents[1]


def _function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False)


@pytest.mark.parametrize("name", ["normalize_text", "_lowercase_sentence_initial"])
def test_normalizer_source_is_identical_to_official_scorer(name: str) -> None:
    assert _function_ast(ROOT / "submission_src" / "selection.py", name) == _function_ast(
        ROOT / "vendor" / "runtime_score.py", name)


@pytest.mark.parametrize("text", [
    "Hola, ¿cómo estás? I don... I don't know [laugh] (???)", "NASA said — well.",
    "  Él dijo \"sí\"; bueno: ok!", "a ... b ... c", "(word) #x27;s ~tilde",
])
def test_normalizer_behaves_like_official_scorer(text: str) -> None:
    assert selection.normalize_text(text) == vendored_normalize(text)


def test_word_edit_distance_matches_jiwer_on_random_pairs() -> None:
    rng = random.Random(0)
    vocab = ["a", "b", "c", "hola", "the", "y", "que", "so"]
    for _ in range(500):
        a = " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 9)))
        b = " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 9)))
        assert selection.word_edit_distance(a, b) == edit_distance(a, b)


def test_mbr_matches_the_evaluated_selector() -> None:
    rng = random.Random(1)
    vocab = ["Yo", "fui", "a", "la", "store.", "Then", "I", "left", "¿sabes?", "okay"]
    candidates = {name: [" ".join(rng.choice(vocab) for _ in range(rng.randint(0, 8)))
                         for _ in range(60)] for name in ("t", "q", "r", "s")}
    assert selection.mbr_select(candidates) == combination.mbr_select(candidates)


@pytest.mark.parametrize("text", [
    "Hola, ¿cómo estás? I don... I don't know [laugh] (???)", "NASA said — well.",
    "Word—Other thing. A B", "don... OK then", "  (word) #x27;s ~tilde  ", "a ... b ... c",
])
def test_normalizer_is_idempotent(text: str) -> None:
    once = selection.normalize_text(text)
    assert selection.normalize_text(once) == once


def test_rover_identical_hypotheses_pass_through() -> None:
    assert selection.rover_combine(["Hola amigo, qué tal."] * 4) == "hola amigo qué tal"


def test_rover_majority_substitution_and_insertion() -> None:
    hyps = ["a b c", "a x b c", "a b c", "a b d"]
    assert selection.rover_combine(hyps) == "a b c"


def test_rover_can_beat_every_single_hypothesis() -> None:
    # Each hypothesis has one error in a different place; voting removes all of them.
    hyps = ["x b c d", "a y c d", "a b z d", "a b c d"]
    assert selection.rover_combine(hyps[:3] + ["a b c w"]) == "a b c d"


def test_rover_drops_words_only_one_system_hallucinated() -> None:
    assert selection.rover_combine(["a b", "a b", "a extra b", "a b"]) == "a b"


def test_rover_ties_go_to_the_first_hypothesis() -> None:
    assert selection.rover_combine(["a b", "a c", "a b", "a c"]) == "a b"
    assert selection.rover_combine(["a c", "a b", "a b", "a c"]) == "a c"


def test_rover_handles_empty_hypotheses() -> None:
    assert selection.rover_combine(["", "", "", ""]) == ""
    assert selection.rover_combine(["a b", "", "a b", "a b"]) == "a b"
    assert selection.rover_combine(["", "", "", "a b"]) == ""


def test_rover_output_words_come_from_inputs() -> None:
    rng = random.Random(3)
    vocab = ["Yo", "fui", "a", "la", "store", "then", "I", "left", "okay", "sí"]
    for _ in range(200):
        hyps = [" ".join(rng.choice(vocab) for _ in range(rng.randint(0, 7))) for _ in range(4)]
        combined = selection.rover_combine(hyps).split()
        pool = {w for h in hyps for w in selection.normalize_text(h).split()}
        assert set(combined) <= pool


def test_rover_backbone_variants() -> None:
    candidates = {
        "t": ["a b c", "hola que tal"],
        "q": ["a b c", "hola que tal amigo"],
        "r": ["a x c", "hola que tal amigo"],
        "s": ["a b c", "hola amigo"],
    }
    outputs, stats = selection.rover(candidates, backbone="mbr")
    assert outputs[0] == "a b c"
    assert stats["clips_changed_from_backbone"] >= 0
    ordered, _ = selection.rover(candidates, backbone="order")
    assert ordered[0] == "a b c"
    with pytest.raises(ValueError):
        selection.rover(candidates, backbone="nope")


@pytest.mark.parametrize(("output", "leaked"), [
    ("Hola cómo estás I'm good", True),        # 4 consecutive prompt words
    ("hola, ¿CÓMO estás? fine", False),        # only 3 consecutive (case-insensitive)
    ("vamos a la tienda", True),
    ("vamos a la playa tienda later", False),
    ("", False),
])
def test_prompt_leaked(output: str, leaked: bool) -> None:
    prompt = "Hola, ¿cómo estás? I'm good, gracias. Vamos a la tienda later, okay?"
    assert selection.prompt_leaked(output, prompt, 4) is leaked


def test_mbr_rejects_fewer_than_three_systems() -> None:
    with pytest.raises(ValueError):
        selection.mbr_select({"a": ["x"], "b": ["y"]})


def test_case_vote_takes_unanimous_voter_casing_only() -> None:
    from submission_src.selection import case_vote

    primary = "we went to miami, and then Pues nada."
    voters = ["We went to Miami and then pues nada", "we went to Miami, and then pues nada"]
    # "Miami" is unanimous; "We"/"we" disagree; "pues" is unanimous lowercase.
    assert case_vote(primary, voters) == "we went to Miami, and then pues nada."
    assert case_vote(primary, ["we went to Miami"]) == "we went to Miami, and then Pues nada."
    assert case_vote("hola", []) == "hola"


def test_collapse_repeats_drops_immediate_duplicates_only() -> None:
    from submission_src.selection import collapse_repeats

    assert collapse_repeats("I I think, think that that's it it.") == "I think, that that's it"
    assert collapse_repeats("no no no") == "no"
    assert collapse_repeats("sí, Sí claro") == "sí, claro"
    assert collapse_repeats("") == ""


def test_remove_sentence_breaks_keeps_i_forms_and_acronyms() -> None:
    from submission_src.selection import remove_sentence_breaks

    text = "Yo fui. I'm tired. Pero bueno? NASA called! Okay."
    assert remove_sentence_breaks(text) == "Yo fui I'm tired pero bueno NASA called okay."
    assert remove_sentence_breaks("Hola.") == "Hola."
    assert remove_sentence_breaks("") == ""
    assert remove_sentence_breaks("Well... Maybe") == "Well maybe"


def test_remove_sentence_breaks_can_keep_ellipses() -> None:
    from submission_src.selection import remove_sentence_breaks

    assert remove_sentence_breaks("So... Maybe. Yes", keep_ellipsis=True) == "So... Maybe yes"
