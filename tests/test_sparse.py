from app.rag.sparse import sparse_document, sparse_query, tokenize


def test_tokenize_strips_accents_and_lowercases():
    assert tokenize("Précision du GEARBOX") == ["precision", "du", "gearbox"]


def test_decimal_comma_equals_decimal_point():
    assert tokenize("95,4 %") == tokenize("95.4 %") == ["95.4"]


def test_identifiers_stay_whole_and_underscores_split():
    assert tokenize("WC9 and G4") == ["wc9", "and", "g4"]
    assert tokenize("Std_RMS_phases") == ["std", "rms", "phases"]


def test_query_and_document_share_indices_for_the_same_words():
    doc = sparse_document("The gearbox accuracy is 95.4")
    query = sparse_query("gearbox accuracy")
    assert set(query.indices) <= set(doc.indices)


def test_indices_are_stable_across_calls():
    assert sparse_query("gearbox").indices == sparse_query("gearbox").indices


def test_repeated_words_weigh_more_but_less_than_linearly():
    one = sparse_document("gearbox")
    three = sparse_document("gearbox gearbox gearbox")
    assert one.values[0] < three.values[0] < 3 * one.values[0]


def test_text_without_words_gives_an_empty_vector():
    assert sparse_query("%%% ---").indices == []