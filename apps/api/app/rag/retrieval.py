"""Two-stage retrieval across attached KBs (PRD §10.4).

Stage 1  Hybrid (dense + BM25, RRF) search in Qdrant over flat chunks (small documents) and page-group
         summaries (large documents). Candidates = the relevance top-n plus MMR picks from a larger pool;
         all three sizes scale with the total pages in the attached KBs (select.budget). JEV scores the
         candidates and the top_k are kept.
Stage 2  Chosen summaries expand to the full-text chunks of their pages (Postgres). Those chunks plus any
         chosen flat chunks are ranked by dense similarity + MMR, JEV scores them again, and the top_k with
         a score of at least JEV_MIN_SCORE go to the LLM.
Without JEV the configured reranker, or else retrieval order, does the selecting.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, KnowledgeBase
from app.rag import chunk_store, qdrant
from app.rag.embeddings import embed_query
from app.rag.select import JEV_MIN_SCORE, STAGE2_MMR, STAGE2_TOP_N, budget, cosine_sims, jev_scores, mmr

log = structlog.get_logger()


@dataclass
class RetrievedChunk:
    chunk_id: str
    kb_id: str
    document_id: str
    filename: str
    page_start: int | None
    page_end: int | None
    section_title: str | None
    text: str
    score: float
    fused_score: float
    rerank_score: float | None = None
    jev_score: float | None = None
    level: str = "chunk"  # chunk | summary (a page-group index entry of a large document)
    document_version: int = 0
    vector: list[float] | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("vector")
        d["text"] = self.text[:2000]
        return d


def _from_hit(h: qdrant.Hit) -> RetrievedChunk:
    p = h.payload
    return RetrievedChunk(
        chunk_id=h.id,
        kb_id=p.get("kb_id", ""),
        document_id=p.get("document_id", ""),
        filename=p.get("filename", ""),
        page_start=p.get("page_start"),
        page_end=p.get("page_end"),
        section_title=p.get("section_title"),
        text=p.get("text", ""),
        score=h.score,
        fused_score=h.fused_score,
        level=p.get("level", "chunk"),
        document_version=int(p.get("document_version") or 0),
        vector=h.vector,
    )


async def _total_pages(session: AsyncSession, org_id: uuid.UUID, kb_ids: list[uuid.UUID]) -> int:
    """Pages across ready documents; a document without page numbers counts one page per chunk."""
    q = select(func.coalesce(func.sum(func.coalesce(Document.page_count, Document.chunk_count)), 0)).where(
        Document.org_id == org_id,
        Document.kb_id.in_(kb_ids),
        Document.deleted_at.is_(None),
        Document.status == "ready",
    )
    return int((await session.execute(q)).scalar_one())


def _diverse(
    query_vec: list[float], cands: list[RetrievedChunk], top_n: int, mmr_n: int
) -> list[RetrievedChunk]:
    """The first top_n (already relevance-ordered) plus mmr_n MMR picks from the rest."""
    keep = list(range(min(top_n, len(cands))))
    if all(c.vector for c in cands):
        keep += mmr(query_vec, [c.vector for c in cands], mmr_n, selected=keep)  # type: ignore[misc]
    else:
        keep = list(range(min(top_n + mmr_n, len(cands))))
    return [cands[i] for i in keep]


async def retrieve(
    session: AsyncSession,
    org_id: uuid.UUID,
    kb_ids: list[uuid.UUID],
    query: str,
    *,
    top_k: int = 5,
    threshold: float = 0.3,
    mode: str = "hybrid",
    rerank: dict[str, Any] | None = None,
    rerank_api_key: str | None = None,
    usage: list[dict[str, Any]] | None = None,
) -> list[RetrievedChunk]:
    if not kb_ids or not query.strip():
        return []
    kbs = (
        (
            await session.execute(
                select(KnowledgeBase).where(
                    KnowledgeBase.org_id == org_id,
                    KnowledgeBase.id.in_(kb_ids),
                    KnowledgeBase.deleted_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    if not kbs:
        return []
    rerank = rerank or {}
    top_n, mmr_n, pool = budget(await _total_pages(session, org_id, [kb.id for kb in kbs]))
    by_collection: dict[tuple[str, str], list[str]] = {}
    for kb in kbs:
        by_collection.setdefault((kb.qdrant_collection, kb.embedding_model), []).append(str(kb.id))
    model_of = {str(kb.id): kb.embedding_model for kb in kbs}
    qvec: dict[str, list[float]] = {}

    # ---- stage 1: flat chunks and page-group summaries
    cands: list[RetrievedChunk] = []
    for (collection, model), ids in by_collection.items():
        dense, sparse = await embed_query(query, model)
        qvec[model] = dense
        hits = await qdrant.search(
            collection,
            str(org_id),
            ids,
            dense,
            sparse,
            limit=pool,
            mode=mode,
            threshold=threshold,
            candidates=pool,
        )
        cands += _diverse(dense, [_from_hit(h) for h in hits], top_n, mmr_n)
    chosen = await _select(query, cands, top_k, rerank, rerank_api_key, usage, min_score=None)
    summaries = [c for c in chosen if c.level == "summary"]
    if not summaries:
        return [c for c in chosen if c.jev_score is None or c.jev_score >= JEV_MIN_SCORE]

    # ---- stage 2: the chosen pages' full-text chunks
    rows = await chunk_store.page_chunks(
        session,
        org_id,
        [(c.document_id, c.document_version, c.page_start or 0, c.page_end or 0) for c in summaries],
    )
    parent = {c.document_id: c for c in summaries}
    found = {str(r.document_id) for r in rows}
    pool2 = [
        RetrievedChunk(
            chunk_id=str(r.id),
            kb_id=str(r.kb_id),
            document_id=str(r.document_id),
            filename=parent[str(r.document_id)].filename,
            page_start=r.page_start,
            page_end=r.page_end,
            section_title=r.section_title,
            text=r.text,
            score=0.0,
            fused_score=0.0,
            document_version=r.document_version,
            vector=list(r.embedding),
        )
        for r in rows
    ]
    pool2 += [c for c in chosen if c.level != "summary" or c.document_id not in found]
    picked: list[RetrievedChunk] = []
    for model in {model_of.get(c.kb_id, "") for c in pool2}:
        group = [c for c in pool2 if model_of.get(c.kb_id, "") == model]
        if model not in qvec or not all(c.vector for c in group):
            picked += group
            continue
        for c, s in zip(group, cosine_sims(qvec[model], [c.vector for c in group]), strict=True):  # type: ignore[misc]
            c.score = round(s, 4)
        group.sort(key=lambda c: c.score, reverse=True)
        picked += _diverse(qvec[model], group, STAGE2_TOP_N, STAGE2_MMR)
    return await _select(query, picked, top_k, rerank, rerank_api_key, usage, min_score=JEV_MIN_SCORE)


async def _select(
    query: str,
    cands: list[RetrievedChunk],
    k: int,
    rerank: dict[str, Any],
    rerank_api_key: str | None,
    usage: list[dict[str, Any]] | None,
    *,
    min_score: float | None,
) -> list[RetrievedChunk]:
    """JEV picks the k most useful candidates; without JEV, the reranker or retrieval order does."""
    if not cands:
        return []
    judged = await jev_scores(query, [c.text for c in cands])
    if judged is not None:
        scores, jev_usage = judged
        if usage is not None:
            usage.append(jev_usage)
        for c, s in zip(cands, scores, strict=True):
            c.jev_score = round(s, 3)
        ranked = sorted(cands, key=lambda c: c.jev_score or 0.0, reverse=True)
        return [c for c in ranked if min_score is None or (c.jev_score or 0.0) >= min_score][:k]
    if rerank.get("enabled"):
        return (await _rerank(query, cands, rerank.get("provider") or "local", rerank_api_key))[:k]
    return cands[:k]


async def _rerank(
    query: str, chunks: list[RetrievedChunk], provider: str, api_key: str | None
) -> list[RetrievedChunk]:
    try:
        if provider == "cohere" and api_key:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(
                    "https://api.cohere.com/v2/rerank",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "rerank-v3.5",
                        "query": query,
                        "documents": [c.text for c in chunks],
                        "top_n": len(chunks),
                    },
                )
                r.raise_for_status()
            for item in r.json().get("results", []):
                chunks[item["index"]].rerank_score = float(item["relevance_score"])
        else:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            def _run() -> list[float]:
                enc = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
                return [float(s) for s in enc.rerank(query, [c.text for c in chunks])]

            for c, s in zip(chunks, await asyncio.to_thread(_run), strict=True):
                c.rerank_score = s
        chunks.sort(key=lambda c: c.rerank_score if c.rerank_score is not None else -1e9, reverse=True)
    except Exception as e:  # noqa: BLE001 - reranking is best-effort
        log.warning("rerank_failed", provider=provider, error=str(e)[:200])
    return chunks
