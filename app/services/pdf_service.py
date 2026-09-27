from __future__ import annotations

import re
from io import BytesIO

from pypdf import PdfReader

class PdfExtractionError(RuntimeError):
    pass


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_GLUED_LOWER_UPPER = re.compile(r"([a-z])([A-Z])")
_GLUED_ACRONYM_WORD = re.compile(r"([A-Z]{2,})([A-Z][a-z])")
# GE-style headers: "49-4000214 Rev 3 21" or "21 49-4000214 Rev 3"
_PRINTED_PAGE_NEAR_DOC = re.compile(
    r"(?:49-\d+\s+Rev\s+\d+\s+(\d{1,3})\b)|(?:\b(\d{1,3})\s+49-\d+\s+Rev\s+\d+)",
    re.IGNORECASE,
)


def normalize_extracted_text(text: str) -> str:
    """Clean PDF extract noise without changing meaning."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub(" ", text)
    text = _GLUED_LOWER_UPPER.sub(r"\1 \2", text)
    text = _GLUED_ACRONYM_WORD.sub(r"\1 \2", text)
    # Collapse spaces/tabs but keep paragraph breaks.
    lines = []
    for line in text.split("\n"):
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned)
    # Drop empty runs to a single blank line max.
    collapsed: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if not blank and collapsed:
                collapsed.append("")
            blank = True
            continue
        collapsed.append(line)
        blank = False
    return "\n".join(collapsed).strip()


def _meaningful_len(text: str) -> int:
    """Count letters/digits so layout space-padding does not look 'longer'."""
    return sum(1 for ch in text if ch.isalnum())


def _extract_raw_page_text(page) -> tuple[str, str]:
    """Return (best_text, mode) using layout when it clearly recovers more content."""
    plain = page.extract_text() or ""
    try:
        layout = page.extract_text(extraction_mode="layout") or ""
    except TypeError:
        layout = ""
    except Exception:
        layout = ""

    plain_len = _meaningful_len(plain)
    layout_len = _meaningful_len(layout)
    # Layout helps multi-column pages; require a clear gain to avoid scrambled pages.
    if layout_len >= max(int(plain_len * 1.15), plain_len + 80):
        return layout, "layout"
    return plain, "plain"


_FR_MARKERS = (
    "lave-vaisselle",
    "utilisation du lave",
    "conseils de dépannage",
    "dépannage",
    "numéros de modèle",
    "nous vous remercions",
    "électroménagers",
    "informations de sécurité",
    "entretien et nettoyage",
    "distributeur de détergent",
    "manuel d'utilisation",
    "inscrivez ci-dessous",
    "bienvenue",
)
_ES_MARKERS = (
    "lavavajillas",
    "uso del lavavajillas",
    "soporte para el consumidor",
    "electrodoméstico",
    "información de seguridad",
    "solución de problemas",
    "manual del propietario",
    "bienvenido",
)
_EN_MARKERS = (
    "dishwasher",
    "troubleshooting",
    "owner's manual",
    "using the dishwasher",
    "care and cleaning",
    "limited warranty",
    "consumer support",
    "start dishwasher",
)
_ACCENTS = set("àâäáéèêëíìîïóòôöúùûüçœæñ¿¡")


def page_language_hint(text: str) -> str:
    """Best-effort language label for a page: en, fr, es, or other."""
    lower = text.lower()
    fr_hits = sum(1 for marker in _FR_MARKERS if marker in lower)
    es_hits = sum(1 for marker in _ES_MARKERS if marker in lower)
    en_hits = sum(1 for marker in _EN_MARKERS if marker in lower)
    accent_count = sum(1 for ch in lower if ch in _ACCENTS)

    if "lavavajillas" in lower or es_hits >= 1:
        return "es"
    if fr_hits >= 1 or (accent_count >= 6 and en_hits == 0):
        return "fr"
    if en_hits >= 1:
        return "en"
    if accent_count >= 12:
        return "other"
    return "en"


def is_english_page(text: str) -> bool:
    return page_language_hint(text) == "en"


def filter_english_pages(
    pages: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[int]]:
    """Keep the leading English section; drop FR/ES pages in bilingual manuals.

    After the first non-English page, later pages stay skipped unless they look
    strongly English again (avoids short welcome/notes pages flipping back).
    """
    kept: list[dict[str, object]] = []
    skipped: list[int] = []
    in_non_english_section = False
    for page in pages:
        text = str(page.get("text", ""))
        lang = page_language_hint(text)
        lower = text.lower()
        en_hits = sum(1 for marker in _EN_MARKERS if marker in lower)
        accent_count = sum(1 for ch in lower if ch in _ACCENTS)
        enriched = {**page, "language_hint": lang}
        page_number = int(page["page_number"])

        if lang != "en":
            in_non_english_section = True
            skipped.append(page_number)
            continue

        if in_non_english_section and not (en_hits >= 2 and accent_count < 5):
            skipped.append(page_number)
            continue

        in_non_english_section = False
        kept.append(enriched)
    return kept, skipped


def parse_printed_page_number(text: str) -> int | None:
    """Parse printed page only when it sits next to a document code (avoids false hits)."""
    head = "\n".join(text.splitlines()[:6])
    tail = "\n".join(text.splitlines()[-6:])
    for block in (head, tail, text[:300]):
        match = _PRINTED_PAGE_NEAR_DOC.search(block)
        if not match:
            continue
        value = int(match.group(1) or match.group(2))
        if 1 <= value <= 200:
            return value
    return None


def extract_pdf_pages(content: bytes) -> list[dict[str, object]]:
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:
        raise PdfExtractionError(f"Could not read PDF: {exc}") from exc

    pages: list[dict[str, object]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw, mode = _extract_raw_page_text(page)
        text = normalize_extracted_text(raw)
        if not text:
            continue
        printed = parse_printed_page_number(text)
        pages.append(
            {
                "page_number": page_number,
                "printed_page": printed,
                "extraction_mode": mode,
                "text": text,
            }
        )

    if not pages:
        raise PdfExtractionError(
            "No extractable text was found. The PDF may be scanned images and need OCR."
        )
    return pages
