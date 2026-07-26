from io import BytesIO

from pypdf import PdfReader


class PdfExtractionError(RuntimeError):
    pass


def extract_pdf_pages(content: bytes) -> list[dict[str, object]]:
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:
        raise PdfExtractionError(f"Could not read PDF: {exc}") from exc

    pages: list[dict[str, object]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page_number": page_number, "text": text})

    if not pages:
        raise PdfExtractionError(
            "No extractable text was found. The PDF may be scanned images and need OCR."
        )
    return pages
