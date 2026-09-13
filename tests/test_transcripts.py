from lit_sp_en import transcripts

REFS = [
    "Yo fui a la store. Then I left [laugh]",
    "I don... I don't know, ¿sabes?",
    "Okay. NASA said so",
]


def test_convention_counts() -> None:
    counts = transcripts.convention_counts(REFS)
    assert counts["square_bracket_tag"] == {"occurrences": 1, "rows": 1}
    assert counts["ellipsis_..."] == {"occurrences": 1, "rows": 1}
    assert counts["inverted_question_¿"]["rows"] == 1
    assert "tilde" not in counts


def test_unbalanced_rows() -> None:
    assert transcripts.unbalanced_rows(["ok [laugh", "fine (x)", "a ) b"]) == {
        "square": [0],
        "paren": [2],
    }


def test_token_profile_counts_shapes() -> None:
    profile = transcripts.token_profile(["I don... NASA don't", "hola ñandú 3"])
    assert profile["tokens"] == 7
    assert profile["ending_in_ellipsis"] == 1
    assert profile["all_caps_2plus_letters"] == 1
    assert profile["with_apostrophe"] == 1
    assert profile["with_digit"] == 1
    assert profile["with_non_ascii_letter"] == 1


def test_sensitivity_oracle_zero_empty_one() -> None:
    result = transcripts.sensitivity(REFS)
    assert result["oracle"] == 0.0
    assert result["empty"] == 1.0
    assert result["tags_removed"] == 0.0


def test_sensitivity_penalises_casing_and_ellipsis() -> None:
    result = transcripts.sensitivity(REFS)
    # "Then" after a period is lowercased by the scorer only if the period survives.
    assert result["no_sentence_punctuation"] > 0
    assert result["no_ellipsis"] > 0
    assert result["lowercased"] > 0


def test_variants_never_return_the_reference_for_empty() -> None:
    assert transcripts.VARIANTS["empty"]("anything") == ""
