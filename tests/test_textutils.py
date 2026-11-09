from app.rag.textutils import clean_text


def test_removes_lone_surrogates():
    assert clean_text("a\ud835b") == "ab"


def test_replaces_control_characters_but_keeps_newlines_and_tabs():
    assert clean_text("a\x00b\nc\td") == "a b\nc\td"


def test_keeps_accents_and_math_symbols():
    text = "précision ≥ 95,4 % — α"
    assert clean_text(text) == text