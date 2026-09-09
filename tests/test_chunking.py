import pytest

from app.chunking import chunk_text


def test_chunk_text_creates_overlapping_chunks():
    chunks = chunk_text("a" * 120, size=50, overlap=10)
    assert len(chunks) == 3
    assert chunks[0].text[-10:] == chunks[1].text[:10]


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", size=10, overlap=10)
