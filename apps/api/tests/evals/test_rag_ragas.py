"""RAGAS eval of the RAG pipeline over two large real PDFs (tests/evals/datasets/rag_pdfs.yaml).

Ingests the PDFs through the real pipeline (S3 -> parse -> chunk/index -> Qdrant) into a dedicated KB in the
demo Northwind org, answers each question with a RAG-only agent through the real LangGraph runtime, and
scores with RAGAS:
  faithfulness        answer claims supported by the retrieved contexts
  answer_relevancy    answer addresses the question
  context_precision   relevant contexts are ranked above irrelevant ones (vs the reference answer)
  context_recall      reference answer claims are covered by the retrieved contexts
  factual_correctness answer vs reference (F1 over claims)
plus page_hit (a retrieved context covers the reference page) and no-answer accuracy.

Run (needs `make seed` and a real model; the PDFs are downloaded on first run):
  AGENTDESK_LIVE=1 RAG_EVAL_TAG=baseline uv run --group evals pytest tests/evals/test_rag_ragas.py -s
Env: RAG_EVAL_TAG (report name, default "run"), RAG_EVAL_REINGEST=1 (re-ingest the PDFs, e.g. after a
pipeline change), EVAL_JUDGE_MODEL (any platform model string, default DEFAULT_CHAT_MODEL), EVAL_CONCURRENCY,
RAG_EVAL_LIMIT (first N cases) or RAG_EVAL_CASES (comma-separated case indexes), for smoke runs.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
import types
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from app.agents.schemas import AgentConfig
from app.core.config import settings
from app.core.ids import uuid7
from app.core.s3 import ensure_bucket, put_bytes
from app.db.models import Agent, Conversation, Document, KnowledgeBase, Membership, Organization, User
from app.db.session import get_sessionmaker
from app.llm import gateway
from app.rag import qdrant
from app.rag.embeddings import dimension
from app.rag.ingestion import delete_document_fully, ingest_document
from app.runtime.runner import prepare_run, run_once
from sqlalchemy import select

from .test_template_evals import judge_declined, new_spend

pytestmark = pytest.mark.asyncio
if os.environ.get("AGENTDESK_LIVE") != "1":
    pytest.skip("live evals: AGENTDESK_LIVE=1", allow_module_level=True)

HERE = Path(__file__).parent
DOCS = HERE / "rag_docs"  # gitignored: downloaded from the URLs in the dataset
REPORTS = HERE / "reports"
TAG = os.environ.get("RAG_EVAL_TAG", "run")
KB_NAME = "RAG eval PDFs"
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "4"))
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL") or settings.default_chat_model
SYSTEM_PROMPT = (
    "You answer questions using only the provided documents (a research paper and an employee handbook). "
    "Be precise with numbers and cite the document. If the documents do not contain the answer, say so."
)


def _ragas() -> Any:
    """ragas 0.4 imports Vertex classes that langchain-community 0.4 removed; they're only isinstance targets."""
    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
        sys.modules[name] = stub
    import langchain_community.llms as lc_llms

    if not hasattr(lc_llms, "VertexAI"):
        lc_llms.VertexAI = type("VertexAI", (), {})  # type: ignore[attr-defined]
    import ragas

    return ragas


# ---------------------------------------------------------------- setup
async def demo_org_agent() -> tuple[Organization, Agent, User] | None:
    async with get_sessionmaker()() as s:
        user = (await s.execute(select(User).where(User.email == "demo@example.com"))).scalar_one_or_none()
        if user is None:
            return None
        agent = (
            (
                await s.execute(
                    select(Agent)
                    .join(Membership, Membership.org_id == Agent.org_id)
                    .where(
                        Membership.user_id == user.id,
                        Agent.template_key == "hr_onboarding",
                        Agent.deleted_at.is_(None),
                    )
                    .order_by(Agent.created_at.desc())
                )
            )
            .scalars()
            .first()
        )
        org = await s.get(Organization, agent.org_id) if agent else None
        return (org, agent, user) if org and agent else None


def download(docs: list[dict[str, str]]) -> None:
    DOCS.mkdir(exist_ok=True)
    for d in docs:
        path = DOCS / d["file"]
        if not path.exists():
            req = urllib.request.Request(d["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 - fixed https URLs from the dataset
                path.write_bytes(r.read())


async def ensure_kb(org: Organization, docs: list[dict[str, str]]) -> tuple[KnowledgeBase, dict[str, Any]]:
    """Returns the eval KB and per-document ingestion time and metrics (empty when reused)."""
    reingest = os.environ.get("RAG_EVAL_REINGEST") == "1"
    async with get_sessionmaker()() as s:
        kb = (
            await s.execute(
                select(KnowledgeBase).where(
                    KnowledgeBase.org_id == org.id,
                    KnowledgeBase.name == KB_NAME,
                    KnowledgeBase.deleted_at.is_(None),
                )
            )
        ).scalar_one_or_none()
        if kb is None:
            emb = settings.default_embedding_model
            dim = dimension(emb)
            kb = KnowledgeBase(
                org_id=org.id,
                name=KB_NAME,
                description="Llama 2 paper (arXiv 2307.09288) and WVSU employee handbook",
                embedding_model=emb,
                embedding_dim=dim,
                qdrant_collection=qdrant.collection_name(emb, dim),
                chunk_config={},
            )
            s.add(kb)
            await s.commit()
        existing = (
            (await s.execute(select(Document).where(Document.kb_id == kb.id, Document.deleted_at.is_(None))))
            .scalars()
            .all()
        )
    ready = {d.filename for d in existing if d.status == "ready"}
    if not reingest and all(d["file"] in ready for d in docs):
        return kb, {}
    for d in existing:
        await delete_document_fully(str(d.id))
        async with get_sessionmaker()() as s:
            row = await s.get(Document, d.id)
            if row:
                row.deleted_at = datetime.now(UTC)
                await s.commit()
    await ensure_bucket()
    timings: dict[str, Any] = {}
    for d in docs:
        data = (DOCS / d["file"]).read_bytes()
        doc_id = uuid7()
        key = f"orgs/{org.id}/kb/{kb.id}/{doc_id}/{d['file']}"
        await put_bytes(key, data, "application/pdf")
        async with get_sessionmaker()() as s:
            s.add(
                Document(
                    id=doc_id,
                    org_id=org.id,
                    kb_id=kb.id,
                    filename=d["file"],
                    mime_type="application/pdf",
                    size_bytes=len(data),
                    storage_key=key,
                    status="queued",
                )
            )
            await s.commit()
        t0 = time.perf_counter()
        metrics = await ingest_document(str(doc_id))
        timings[d["file"]] = {"seconds": round(time.perf_counter() - t0, 1), **metrics}
    return kb, timings


# ---------------------------------------------------------------- running
async def run_case(
    case: dict[str, Any], org: Organization, agent: Agent, user: User, cfg: AgentConfig
) -> dict[str, Any]:
    q = case["question"]
    async with get_sessionmaker()() as s:
        conv = Conversation(
            org_id=org.id,
            agent_id=agent.id,
            channel="playground",
            playground_user_id=user.id,
            flags={"eval": True},
        )
        s.add(conv)
        await s.flush()
        ctx = await prepare_run(
            s, org=org, agent=agent, cfg=cfg, channel="playground", conversation=conv, user_text=q
        )
        await s.commit()
    outcome = await run_once(ctx, q)
    state = outcome.state
    chunks = state.get("rag_chunks") or []
    usage = state.get("usage") or []
    page = case.get("page")
    hit = (
        any(
            (c.get("page_start") or 0) <= page <= (c.get("page_end") or c.get("page_start") or 0)
            for c in chunks
        )
        if page
        else None
    )
    answer = state.get("answer") or ""
    declined = None
    if case.get("expected_no_answer"):
        declined = bool(state.get("no_answer")) or bool(await judge_declined(q, answer, new_spend()))
    return {
        "question": q,
        "doc": case.get("doc"),
        "kind": case.get("kind", "text"),
        "reference": case["reference"],
        "answer": answer,
        "contexts": [c.get("text", "") for c in chunks],
        "pages": [[c.get("page_start"), c.get("page_end")] for c in chunks],
        "page_hit": hit,
        "declined": declined,
        "error": outcome.error,
        "latency_ms": outcome.latency_ms,
        "retrieve_ms": int(1000 * float((state.get("timings") or {}).get("rag_retrieve", 0.0))),
        "agent_cost_usd": round(sum(float(u.get("est_cost_usd", 0.0)) for u in usage), 6),
        "cost_by_node": {
            n: round(sum(float(u.get("est_cost_usd", 0.0)) for u in usage if u.get("node") == n), 6)
            for n in {u.get("node") for u in usage}
        },
    }


def gateway_judge() -> Any:
    """RAGAS judge that calls JUDGE_MODEL through the app's LLM gateway, so any configured provider works."""
    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms.base import BaseRagasLLM

    rm = gateway.platform_model(model_string=JUDGE_MODEL)

    class GatewayJudge(BaseRagasLLM):
        async def agenerate_text(  # type: ignore[override]
            self,
            prompt: Any,
            n: int = 1,
            temperature: float | None = 0.01,
            stop: Any = None,
            callbacks: Any = None,
        ) -> LLMResult:
            r = await gateway.chat(rm, [{"role": "user", "content": prompt.to_string()}], max_tokens=4096)
            return LLMResult(generations=[[Generation(text=r.content)]])

        def generate_text(self, *args: Any, **kw: Any) -> LLMResult:  # type: ignore[override]
            return asyncio.run(self.agenerate_text(*args, **kw))

        def is_finished(self, response: LLMResult) -> bool:
            return True

    return GatewayJudge()


def ragas_scores(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    ragas = _ragas()
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        FactualCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    judge = gateway_judge()
    emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.default_embedding_model.removeprefix("openai/"),
            base_url=settings.embedding_api_base,
            api_key="sk-no-key",
            check_embedding_ctx_length=False,
        )
    )
    ds = ragas.EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "response": r["answer"],
                "retrieved_contexts": r["contexts"] or [""],
                "reference": r["reference"],
            }
            for r in rows
        ]
    )
    res = ragas.evaluate(
        ds,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            FactualCorrectness(),
        ],
        llm=judge,
        embeddings=emb,
        run_config=RunConfig(max_workers=CONCURRENCY * 2, timeout=240),
        show_progress=False,
    )
    df = res.to_pandas()
    keys = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]
    fc = next(c for c in df.columns if c.startswith("factual_correctness"))
    return [
        {
            "faithfulness": row["faithfulness"],
            "answer_relevancy": row["answer_relevancy"],
            "context_precision": row["llm_context_precision_with_reference"],
            "context_recall": row["context_recall"],
            "factual_correctness": row[fc],
        }
        for _, row in df[keys + [fc]].iterrows()
    ]


def mean(vals: list[Any]) -> float | None:
    xs = [float(v) for v in vals if v is not None and v == v]  # drop None and NaN
    return round(statistics.mean(xs), 3) if xs else None


async def test_rag_ragas() -> None:
    seeded = await demo_org_agent()
    if seeded is None:
        pytest.skip("run `make seed` first")
    org, agent, user = seeded
    spec = yaml.safe_load((HERE / "datasets" / "rag_pdfs.yaml").read_text(encoding="utf-8"))
    cases = spec["cases"][: int(os.environ.get("RAG_EVAL_LIMIT", "0")) or None]
    if os.environ.get("RAG_EVAL_CASES"):
        cases = [spec["cases"][int(i)] for i in os.environ["RAG_EVAL_CASES"].split(",")]
    download(spec["documents"])
    kb, ingest_s = await ensure_kb(org, spec["documents"])

    cfg = AgentConfig.model_validate(agent.draft_config)
    cfg.instructions.system_prompt = SYSTEM_PROMPT
    cfg.guardrails.allowed_topics = ""
    cfg.tools.rag.enabled = True
    cfg.tools.rag.kb_ids = [kb.id]
    cfg.tools.sql.enabled = cfg.tools.web_search.enabled = False

    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded(c: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            for attempt in range(3):
                r = await run_case(c, org, agent, user, cfg)
                if r["error"] is None or attempt == 2:
                    return r
                await asyncio.sleep(15 * (attempt + 1))
            raise AssertionError("unreachable")

    results = list(await asyncio.gather(*(bounded(c) for c in cases)))
    answerable = [r for r in results if r["declined"] is None and r["error"] is None]
    for r, s in zip(answerable, await asyncio.to_thread(ragas_scores, answerable), strict=True):
        r["ragas"] = {k: (None if v != v else round(float(v), 3)) for k, v in s.items()}

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        lat = sorted(r["latency_ms"] for r in rows) or [0]
        out: dict[str, Any] = {
            k: mean([r.get("ragas", {}).get(k) for r in rows])
            for k in (
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "factual_correctness",
            )
        }
        hits = [r["page_hit"] for r in rows if r["page_hit"] is not None]
        out["page_hit_rate"] = round(sum(hits) / len(hits), 3) if hits else None
        out["latency_p50_ms"] = int(statistics.median(lat))
        out["retrieve_p50_ms"] = int(statistics.median([r["retrieve_ms"] for r in rows] or [0]))
        out["avg_contexts"] = mean([len(r["contexts"]) for r in rows])
        out["avg_cost_usd"] = mean([r["agent_cost_usd"] for r in rows])
        return out

    no_ans = [r["declined"] for r in results if r["declined"] is not None]
    report = {
        "tag": TAG,
        "model": settings.default_chat_model,
        "judge_model": JUDGE_MODEL,
        "embedding_model": settings.default_embedding_model,
        "ingestion": ingest_s,
        "cases": len(results),
        "errors": sum(r["error"] is not None for r in results),
        "metrics": summarize(answerable),
        "no_answer_accuracy": round(sum(no_ans) / len(no_ans), 3) if no_ans else None,
        "by_doc": {d: summarize([r for r in answerable if r["doc"] == d]) for d in ("llama2", "wvsu")},
        "tables": summarize([r for r in answerable if r["kind"] == "table"]),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "results": results,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"rag_{TAG}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(
        f"\n[rag:{TAG}] "
        + json.dumps({k: report[k] for k in ("metrics", "no_answer_accuracy", "by_doc", "tables")}, indent=1)
    )
    assert report["errors"] == 0, [r for r in results if r["error"]]
