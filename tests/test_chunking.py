from app.rag.chunking import CHUNK_SIZE, chunk_page_with_context, chunk_text


def test_chunks_respect_the_size_limit():
    assert all(len(chunk) <= CHUNK_SIZE for chunk in chunk_text("mot " * 1000))


def test_later_chunks_get_the_caption_above_them():
    page = "Table 2 Results of feature-based single-dataset experiments\n" + "valeur " * 400
    chunks = chunk_page_with_context(page)
    assert len(chunks) > 1
    assert chunks[0].startswith("Table 2")            # the caption is already inside the first chunk
    assert chunks[1].startswith("[Table 2 Results")   # prefix added to the following chunks
    assert all("Table 2 Results" in chunk for chunk in chunks)


def test_page_without_caption_is_left_alone():
    chunks = chunk_page_with_context("simple paragraph " * 100)
    assert all(not chunk.startswith("[") for chunk in chunks)