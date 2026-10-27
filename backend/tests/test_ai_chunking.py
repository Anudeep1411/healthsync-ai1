from app.services.ai import _split_document_text


def test_long_document_text_is_split_without_losing_content():
    source = "Page A line one\nPage A line two\n\n" + ("Later page medicine/result text\n" * 900)
    chunks = _split_document_text(source, max_chars=14000)
    assert len(chunks) > 1
    assert all(len(chunk) <= 14000 for chunk in chunks)
    joined = "\n".join(chunks)
    assert "Page A line one" in joined
    assert "Later page medicine/result text" in joined
    assert sum(len(chunk) for chunk in chunks) >= len(source.strip()) - 10
