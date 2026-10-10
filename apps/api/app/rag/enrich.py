"""Ingest-time LLM enrichment: image/chart captions and page-group summaries for hierarchical indexing.

Large documents (more than HIERARCHICAL_MIN_PAGES pages) get one structured index entry per group of
pages: a title, a summary, the questions those pages answer, key topics, and a summary of every table,
picture and chart on them. Those entries go to the vector DB; the full-text chunks go to Postgres
(app.rag.chunk_store) and are fetched by page range at query time.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from app.llm import gateway
from app.rag.chunking import Chunk
from app.rag.parsing import Element, Image

log = structlog.get_logger()

HIERARCHICAL_MIN_PAGES = 31  # 31+ pages -> hierarchical index; 30 or fewer -> flat chunks in the vector DB
TARGET_UNITS = 40  # aim for ~40 index entries per document when grouping pages
MAX_PAGES_PER_UNIT = 4
MAX_UNIT_CHARS = 24_000  # content sent to the summariser per page group
MIN_IMAGE_B64 = 6_000  # ~4.5 KB: smaller pictures are icons, bullets and rules - skipped
MAX_IMAGES = 60  # captioned per document (cost bound); later pictures are dropped
LLM_CONCURRENCY = 8


def pages_per_unit(page_count: int) -> int:
    """31-40 pages: 1 page per entry, 41-80: 2, 81-120: 3, 121+: 4."""
    return max(1, min(MAX_PAGES_PER_UNIT, math.ceil(page_count / TARGET_UNITS)))


def is_hierarchical(page_count: int | None) -> bool:
    return (page_count or 0) >= HIERARCHICAL_MIN_PAGES


@dataclass
class Spend:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, r: gateway.ChatResult) -> None:
        self.input_tokens += r.usage.input_tokens
        self.output_tokens += r.usage.output_tokens
        self.cost_usd += r.est_cost_usd
        self.calls += 1


# ---------------------------------------------------------------- pictures and charts
class Caption(BaseModel):
    kind: Literal["chart", "diagram", "photo", "screenshot", "logo", "decorative", "other"]
    title: str = Field(max_length=300)
    summary: str = Field(max_length=1500)


CAPTION_PROMPT = """This picture comes from page {page} of the document "{filename}".{near}
Return ONLY a JSON object: {{"kind": "chart" | "diagram" | "photo" | "screenshot" | "logo" | "decorative" | "other", \
"title": "...", "summary": "..."}}.
- title: what the picture shows, specific (e.g. "Training loss vs processed tokens for Llama 2 7B-70B").
- summary: 1-3 sentences a reader could search for. For charts and plots give the axes, the series and the \
key values or trend; for diagrams the components and how they connect. Do not invent values you cannot read."""


async def caption_images(elements: list[Element], filename: str, spend: Spend) -> list[Element]:
    """Replaces each extracted picture with its caption text (kept as an Image element so chunks record it);
    drops icons, logos and pictures that could not be captioned."""
    rm = gateway.platform_model(utility=True)
    sem = asyncio.Semaphore(LLM_CONCURRENCY)
    todo = [
        i
        for i, e in enumerate(elements)
        if isinstance(e, Image) and len(e.metadata.image_base64 or "") >= MIN_IMAGE_B64
    ][:MAX_IMAGES]

    async def one(i: int) -> tuple[int, Caption | None]:
        e = elements[i]
        nxt = elements[i + 1].text if i + 1 < len(elements) else ""
        near = f'\nText next to it: "{nxt[:300]}"' if nxt else ""
        prompt = CAPTION_PROMPT.format(page=e.metadata.page_number or "?", filename=filename, near=near)
        mime = e.metadata.image_mime_type or "image/png"
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{e.metadata.image_base64}"}},
            ],
        }
        async with sem:
            try:
                r = await gateway.chat(
                    rm, [msg], temperature=0, max_tokens=400, response_format={"type": "json_object"}
                )
                spend.add(r)
                return i, Caption.model_validate_json(gateway.extract_json(r.content))
            except Exception as ex:  # noqa: BLE001 - a picture we can't caption is dropped, not fatal
                log.warning("image_caption_failed", page=e.metadata.page_number, error=str(ex)[:200])
                return i, None

    captions = dict(await asyncio.gather(*(one(i) for i in todo)))
    out: list[Element] = []
    for i, e in enumerate(elements):
        if not isinstance(e, Image):
            out.append(e)
            continue
        cap = captions.get(i)
        e.metadata.image_base64 = None  # never carry the bytes into chunks or payloads
        if cap and cap.kind not in ("logo", "decorative"):
            label = {"chart": "Chart", "diagram": "Diagram"}.get(cap.kind, "Figure")
            e.text = f"[{label}: {cap.title}] {cap.summary}"
            out.append(e)
    return out


# ---------------------------------------------------------------- page-group summaries
class Visual(BaseModel):
    title: str
    summary: str


class PageSummary(BaseModel):
    title: str
    summary: str
    questions: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    tables: list[Visual] = Field(default_factory=list)
    images: list[Visual] = Field(default_factory=list)
    graphs: list[Visual] = Field(default_factory=list)


SUMMARY_PROMPT = """You are building the index of a retrieval system. Below is the content of pages {ps}-{pe} of \
"{filename}" ({pages} pages in total). Tables are HTML; pictures appear as [Figure: ...], [Chart: ...] or \
[Diagram: ...] descriptions.

Write the index entry for these pages:
- title: a specific title naming the sections or subjects on these pages (never just "Page {ps}").
- summary: 3-6 sentences covering every distinct topic with facts on these pages (rules, procedures, \
definitions, methods, results). Keep key numbers, names and terms verbatim.
- questions: 4-8 specific questions a reader could answer from these pages alone, in the document's terms.
- topics: 5-12 key topics, entities, terms and acronyms on these pages.
- tables: one entry per table (title + 1-2 sentences with its key values). Empty if none.
- images: one entry per photo, figure or diagram (title + 1-2 sentences). Empty if none.
- graphs: one entry per chart or plot (title + 1-2 sentences: axes, series, key values or trend). Empty if none.
Only use facts present in the content.

CONTENT:
{content}"""


@dataclass
class Unit:
    index: int
    page_start: int
    page_end: int
    summary: PageSummary
    chunk_indexes: list[int] = field(default_factory=list)

    def render(self, filename: str) -> str:
        s = self.summary
        pages = (
            f"page {self.page_start}"
            if self.page_start == self.page_end
            else f"pages {self.page_start}-{self.page_end}"
        )
        lines = [f"{filename}, {pages}: {s.title}", s.summary]
        if s.topics:
            lines.append("Topics: " + "; ".join(s.topics))
        if s.questions:
            lines.append("Answers: " + " | ".join(s.questions))
        for label, items in (("Table", s.tables), ("Figure", s.images), ("Chart", s.graphs)):
            lines += [f"{label}: {v.title} - {v.summary}" for v in items]
        return "\n".join(lines)


def group_chunks(chunks: list[Chunk], page_count: int) -> list[tuple[int, int, list[int]]]:
    """(page_start, page_end, chunk indexes) per page group; a chunk belongs to the group of its first page."""
    g = pages_per_unit(page_count)
    groups: list[tuple[int, int, list[int]]] = []
    for start in range(1, page_count + 1, g):
        end = min(start + g - 1, page_count)
        idx = [i for i, c in enumerate(chunks) if c.page_start is not None and start <= c.page_start <= end]
        if idx:  # blank / image-only pages without captions have nothing to index
            groups.append((start, end, idx))
    return groups


async def summarize_units(chunks: list[Chunk], page_count: int, filename: str, spend: Spend) -> list[Unit]:
    rm = gateway.platform_model(utility=True)
    sem = asyncio.Semaphore(LLM_CONCURRENCY)

    async def one(n: int, ps: int, pe: int, idx: list[int]) -> Unit:
        content = "\n\n".join(chunks[i].text for i in idx)[:MAX_UNIT_CHARS]
        prompt = SUMMARY_PROMPT.format(ps=ps, pe=pe, filename=filename, pages=page_count, content=content)
        async with sem:
            try:
                s, r = await gateway.structured(
                    rm, [{"role": "user", "content": prompt}], PageSummary, max_tokens=1500
                )
                spend.add(r)
            except Exception as e:  # noqa: BLE001 - keep the pages findable with a plain-text entry
                log.warning("page_summary_failed", pages=f"{ps}-{pe}", error=str(e)[:200])
                title = next((chunks[i].section_title for i in idx if chunks[i].section_title), None)
                s = PageSummary(title=title or f"{filename} pages {ps}-{pe}", summary=content[:800])
        return Unit(index=n, page_start=ps, page_end=pe, summary=s, chunk_indexes=idx)

    groups = group_chunks(chunks, page_count)
    return list(await asyncio.gather(*(one(n, ps, pe, idx) for n, (ps, pe, idx) in enumerate(groups))))
