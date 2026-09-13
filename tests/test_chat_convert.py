import pytest

from lit_sp_en.chat_convert import convert
from lit_sp_en.transcripts import normalize_text

B = "\x15"
LANGS = ["eng", "spa"]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (f"so I went to la@s:spa tienda@s . {B}100_900{B}", "So I went to la tienda."),
        ("[- spa] pero OK@s:eng ?", "Pero OK?"),
        ("<I want> [/] I want ice_cream .", "I want I want ice cream."),
        ("goin(g) so: (.) there !", "Going so there!"),
        ("&um I &don I don't know +...", "Um I don... I don't know."),
        # Terminator stays on the last word; the scorer drops both tag and period.
        ("yeah &=laughs .", "Yeah. [laughs]"),
        ("+< mhm .", "Mhm."),
        ("no [=! laughs] way [?] +//.", "No way."),
        ("well , you know +/.", "Well, you know."),
        ("I said +\"/.", "I said."),
        ("0is it ?", "It?"),
        ("&-uh right .", "Uh right."),
        ("I &don &=laughs .", "I don... [laughs]"),
        ("pues, bueno , ok .", "Pues, bueno, ok."),
    ],
)
def test_convert_rules(body: str, expected: str) -> None:
    assert convert(body, LANGS).text == expected


def test_word_languages_follow_default_precode_and_switches() -> None:
    assert convert("I like la@s:spa casa@s .", LANGS).words == [
        ("I", "eng"), ("like", "eng"), ("la", "spa"), ("casa", "spa")]
    assert convert("[- spa] sí OK@s:eng .", LANGS).words == [("sí", "spa"), ("OK", "eng")]
    assert convert("hola .", ["spa", "eng"]).words == [("hola", "spa")]


@pytest.mark.parametrize(
    ("body", "flag"),
    [("xxx and then .", "unintelligible"), ("www .", "untranscribed"), ("yyy .", "phonological")],
)
def test_placeholders_flag_exclusion(body: str, flag: str) -> None:
    converted = convert(body, LANGS)
    assert flag in converted.flags
    assert converted.excluded


def test_tag_only_utterance_has_no_words_but_is_not_excluded() -> None:
    converted = convert("&=laughs .", LANGS)
    assert converted.text == "[laughs]"
    assert converted.flags == {"no_words"}
    assert not converted.excluded


def test_no_chat_symbols_survive_official_normalization() -> None:
    body = ("[- spa] <pero&+qu> [//] &um pues@s:eng xxx goin(g) ice_cream so: &=laughs "
            "[=! laughs] (.) ‡ „ [?] [\"] [*] OK@s:eng , +...")
    normalized = normalize_text(convert(body, LANGS).text)
    assert not set(normalized) & set("@&<>[]+=_:^()*/\"‡„")
