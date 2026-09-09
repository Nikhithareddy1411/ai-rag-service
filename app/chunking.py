from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[Chunk]:
    text = " ".join(text.split())
    if not text:
        return []
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be between 0 and size-1")

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(Chunk(text=text[start:end], index=index))
        if end == len(text):
            break
        start = end - overlap
        index += 1
    return chunks
