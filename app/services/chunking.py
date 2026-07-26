from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    text: str
    page_number: int
    chunk_index: int


def chunk_pages(
    pages: list[dict[str, object]],
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[TextChunk] = []
    for page in pages:
        text = " ".join(str(page["text"]).split())
        page_number = int(page["page_number"])
        start = 0
        index = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(TextChunk(chunk_text, page_number, index))
            if end == len(text):
                break
            start = end - overlap
            index += 1
    return chunks
