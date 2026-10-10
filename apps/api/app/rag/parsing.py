"""Layout-aware parsing into Unstructured elements (PRD §10.2).

DOCX / TXT / MD use Unstructured's partitioners. PDF uses Unstructured's hi_res strategy when the layout
stack is installed (`uv sync --extra layout` plus poppler and tesseract; the worker image ships it): a
layout model finds titles, text, tables (structure inferred to HTML) and pictures (charts, diagrams,
photos, extracted as base64 for captioning). Without it, PDFs fall back to pdfminer: titles from font
size, a table heuristic, no images. Page numbers are kept and headers/footers dropped either way.
"""

from __future__ import annotations

import functools
import io
import shutil
import statistics
from dataclasses import dataclass

import structlog
from unstructured.documents.elements import (
    Element,
    ElementMetadata,
    Footer,
    Header,
    Image,
    NarrativeText,
    PageBreak,
    Table,
    Text,
    Title,
)

DROP_TYPES = (Header, Footer, PageBreak)
log = structlog.get_logger()


@dataclass
class ParsedDocument:
    elements: list[Element]
    page_count: int | None


def _clean(elements: list[Element]) -> list[Element]:
    return [
        e
        for e in elements
        if not isinstance(e, DROP_TYPES)
        and ((e.text or "").strip() or (isinstance(e, Image) and e.metadata.image_base64))
    ]


@functools.cache
def hi_res_available() -> bool:
    try:
        import unstructured_inference  # noqa: F401
    except ImportError:
        return False
    return bool(shutil.which("pdftoppm") and shutil.which("tesseract"))


def _pdf_page_count(data: bytes) -> int | None:
    from pdfminer.pdfpage import PDFPage

    return sum(1 for _ in PDFPage.get_pages(io.BytesIO(data))) or None


def parse_pdf_hi_res(data: bytes, strategy: str = "hi_res") -> ParsedDocument:
    from unstructured.partition.pdf import partition_pdf

    els = partition_pdf(
        file=io.BytesIO(data),
        strategy=strategy,
        infer_table_structure=True,  # Table.metadata.text_as_html
        extract_image_block_types=["Image"],  # pictures, charts, diagrams -> metadata.image_base64
        extract_image_block_to_payload=True,
    )
    return ParsedDocument(elements=_clean(list(els)), page_count=_pdf_page_count(data))


def parse_pdf(data: bytes, strategy: str = "auto") -> ParsedDocument:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTChar, LTTextBox, LTTextLine

    laparams = LAParams(line_margin=0.3, char_margin=2.0, boxes_flow=0.5 if strategy != "ocr_only" else None)
    elements: list[Element] = []
    page_count = 0
    for page_no, page in enumerate(extract_pages(io.BytesIO(data), laparams=laparams), start=1):
        page_count = page_no
        page_h = float(page.height or 0)
        boxes: list[tuple[str, float, float]] = []
        for obj in page:
            if not isinstance(obj, LTTextBox):
                continue
            text = obj.get_text().strip()
            if not text:
                continue
            sizes = [
                c.size for line in obj if isinstance(line, LTTextLine) for c in line if isinstance(c, LTChar)
            ]
            size = statistics.median(sizes) if sizes else 10.0
            boxes.append((text, size, float(obj.y1)))
        if not boxes:
            continue
        body = statistics.median(s for _, s, _ in boxes)
        for text, size, y1 in boxes:
            # header/footer: single short line at the extreme top or bottom of the page
            if page_h and (y1 > page_h * 0.94 or y1 < page_h * 0.07) and len(text) < 90 and "\n" not in text:
                continue
            meta = ElementMetadata(page_number=page_no)
            lines = text.split("\n")
            if size >= body * 1.15 and len(text) < 120 and len(lines) <= 2:
                elements.append(Title(text=text.replace("\n", " "), metadata=meta))
            elif _looks_like_table(lines):
                elements.append(Table(text=text, metadata=meta))
            else:
                elements.append(NarrativeText(text=" ".join(line.strip() for line in lines), metadata=meta))
    return ParsedDocument(elements=_clean(elements), page_count=page_count or None)


def _looks_like_table(lines: list[str]) -> bool:
    # ponytail: heuristic (>=3 lines with 3+ multi-space separated cells); replace with a layout model for scans
    if len(lines) < 3:
        return False
    tabular = sum(1 for line in lines if len([c for c in line.split("  ") if c.strip()]) >= 3)
    return tabular >= max(3, int(len(lines) * 0.6))


def parse_docx(data: bytes) -> ParsedDocument:
    from unstructured.partition.docx import partition_docx

    els = partition_docx(file=io.BytesIO(data), infer_table_structure=True)
    pages = {e.metadata.page_number for e in els if e.metadata.page_number}
    return ParsedDocument(elements=_clean(list(els)), page_count=max(pages) if pages else None)


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        from charset_normalizer import from_bytes

        best = from_bytes(data).best()
        return str(best) if best else data.decode("latin-1", errors="replace")


def parse_text(data: bytes) -> ParsedDocument:
    from unstructured.partition.text import partition_text

    return ParsedDocument(elements=_clean(list(partition_text(text=_decode(data)))), page_count=None)


def parse_markdown(data: bytes) -> ParsedDocument:
    from unstructured.partition.md import partition_md

    return ParsedDocument(elements=_clean(list(partition_md(text=_decode(data)))), page_count=None)


def parse_bytes(data: bytes, mime_type: str, strategy: str = "auto") -> ParsedDocument:
    if mime_type == "application/pdf":
        if strategy != "fast" and hi_res_available():
            return parse_pdf_hi_res(data, "ocr_only" if strategy == "ocr_only" else "hi_res")
        if strategy != "fast":
            log.warning(
                "layout_parser_unavailable", hint="uv sync --extra layout; install poppler and tesseract"
            )
        return parse_pdf(data, strategy)
    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return parse_docx(data)
    if mime_type == "text/markdown":
        return parse_markdown(data)
    if mime_type in ("text/plain", "text/csv"):
        return parse_text(data)
    raise ValueError(f"Unsupported file type: {mime_type}")


__all__ = ["Element", "Image", "ParsedDocument", "Table", "Text", "Title", "hi_res_available", "parse_bytes"]
