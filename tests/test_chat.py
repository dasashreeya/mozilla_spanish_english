from pathlib import Path

from lit_sp_en import chat

B = "\x15"


def test_clean_words_default_language_and_switches() -> None:
    body = f"so I went to la@s:spa tienda@s yesterday . {B}100_900{B}"
    words = chat.clean_words(body, ["eng", "spa"])
    assert words == [
        ("so", "eng"), ("I", "eng"), ("went", "eng"), ("to", "eng"),
        ("la", "spa"), ("tienda", "spa"), ("yesterday", "eng"),
    ]


def test_clean_words_precode_switches_the_utterance_default() -> None:
    words = chat.clean_words("[- spa] pero OK@s:eng .", ["eng", "spa"])
    assert words == [("pero", "spa"), ("OK", "eng")]


def test_clean_words_drops_codes_and_placeholders() -> None:
    body = "<I want> [/] I want xxx &=laughs (.) goin(g) so: [?] www +..."
    words = [w for w, _ in chat.clean_words(body, ["eng", "spa"])]
    assert words == ["I", "want", "I", "want", "going", "so"]


def test_annotation_counts_ignore_bullet_digits() -> None:
    counts = chat.annotation_counts(f"yeah@s:eng &=laughs <a b> [/] a b +... {B}12_34{B}")
    assert counts["word_switch_@s"] == 1
    assert counts["event_&="] == 1
    assert counts["retrace_[/]"] == 1
    assert counts["scope_<...>"] == 1
    assert counts["trail_off_+..."] == 1
    assert counts["compound_underscore"] == 0


def test_parse_file(tmp_path: Path) -> None:
    path = tmp_path / "x.cha"
    path.write_text(
        "@UTF8\n@Begin\n@Languages:\tspa, eng\n@Warning:\tsome note\n"
        f"*AAA:\thola . {B}0_1200{B}\n%eng:\thello\n"
        f"*BBB:\tyes@s:eng .\n@End\n",
        encoding="utf-8",
    )
    parsed = chat.parse(path)
    assert parsed.languages == ["spa", "eng"]
    assert parsed.header_tags["@Languages"] == 1
    assert parsed.dependent_tiers["%eng"] == 1
    assert parsed.warnings == ["some note"]
    first, second = parsed.utterances
    assert (first.speaker, first.start_ms, first.end_ms) == ("AAA", 0, 1200)
    assert first.words == [("hola", "spa")]
    assert second.start_ms is None and second.words == [("yes", "eng")]
