from app.agents.generate import Citation, verify_citations

CHUNKS = [
    "The model reaches an accuracy of 97.0% for the motor,\n95.4% for the gearbox, and 99.8% for the left axle box.",
    "Table 1 Working conditions\nWC3 60 Hz / 0 kN Test\nWC9 60 Hz / -10 kN Test",
]


def kept(*pairs):
    return verify_citations([Citation(chunk=c, quote=q) for c, q in pairs], CHUNKS)


def test_verbatim_quote_is_kept():
    assert len(kept((1, "accuracy of 97.0% for the motor"))) == 1


def test_line_breaks_in_the_source_do_not_matter():
    assert len(kept((1, "motor, 95.4% for the gearbox"))) == 1


def test_translated_or_invented_quote_is_rejected():
    assert kept((1, "la précision du gearbox est de 95,4 %")) == []


def test_quote_must_come_from_the_chunk_it_names():
    assert kept((2, "accuracy of 97.0% for the motor")) == []


def test_unknown_chunk_number_is_rejected():
    assert kept((9, "accuracy of 97.0% for the motor")) == []


def test_short_words_are_rejected_but_short_numbers_are_allowed():
    assert kept((1, "PU")) == []
    assert len(kept((1, "97.0"))) == 1


def test_ellipsis_joins_segments_that_each_exist():
    assert len(kept((2, "WC3 60 Hz / 0 kN Test ... WC9 60 Hz"))) == 1
    assert kept((2, "WC3 60 Hz / 0 kN Test ... invented segment here")) == []