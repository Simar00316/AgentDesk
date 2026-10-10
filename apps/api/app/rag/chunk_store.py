"""Full-text chunks of hierarchically indexed documents, in Postgres (see app.rag.enrich)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import and_, delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DocumentChunk
from app.db.session import session_scope
from app.rag.chunking import Chunk


async def replace_chunks(
    org_id: str, kb_id: str, document_id: str, version: int, chunks: list[Chunk], vectors: list[list[float]]
) -> None:
    doc = uuid.UUID(document_id)
    async with session_scope() as s:
        await s.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc))
        s.add_all(
            DocumentChunk(
                org_id=uuid.UUID(org_id),
                kb_id=uuid.UUID(kb_id),
                document_id=doc,
                document_version=version,
                chunk_index=c.chunk_index,
                page_start=c.page_start,
                page_end=c.page_end,
                section_title=c.section_title,
                element_types=c.element_types,
                text=c.text,
                embedding=v,
            )
            for c, v in zip(chunks, vectors, strict=True)
        )


async def delete_chunks(document_id: str) -> None:
    async with session_scope() as s:
        await s.execute(delete(DocumentChunk).where(DocumentChunk.document_id == uuid.UUID(document_id)))


async def page_chunks(
    session: AsyncSession, org_id: uuid.UUID, ranges: Sequence[tuple[str, int, int, int]]
) -> list[DocumentChunk]:
    """Chunks overlapping any (document_id, version, page_start, page_end) range, tenant-scoped."""
    if not ranges:
        return []
    conds = [
        and_(
            DocumentChunk.document_id == uuid.UUID(d),
            DocumentChunk.document_version == v,
            DocumentChunk.page_start <= pe,
            DocumentChunk.page_end >= ps,
        )
        for d, v, ps, pe in ranges
    ]
    q = (
        select(DocumentChunk)
        .where(DocumentChunk.org_id == org_id, or_(*conds))
        .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
    )
    return list((await session.execute(q)).scalars().all())
