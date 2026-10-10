"""Candidate selection for retrieval: page-scaled budgets, MMR diversity, and JEV relevance judging."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import structlog

from app.llm import jev

log = structlog.get_logger()

# Stage 1 budget by total pages in the attached knowledge bases:
# (max pages, relevance top-n, MMR picks, pool fetched from Qdrant for MMR to choose from)
BUDGETS = [
    (30, 6, 4, 20),
    (150, 8, 8, 40),
    (500, 10, 12, 60),
    (None, 12, 18, 100),
]
STAGE2_TOP_N = 8  # stage 2: chunks of the chosen pages, ranked by dense similarity...
STAGE2_MMR = 6  # ...plus this many MMR picks for diversity
MMR_LAMBDA = 0.5  # 1 = pure relevance, 0 = pure diversity
JEV_MIN_SCORE = 0.5  # final stage drops passages JEV scores below this (0-3 scale: mostly "unrelated")
JEV_PASSAGE_CHARS = 2000  # passage text sent to JEV, after stripping table HTML tags
JEV_CRITERIA = [
    "Unrelated to the question",
    "Same general topic, but does not help answer the question",
    "Contains part of the answer, or describes pages that likely contain it",
    "Directly contains the answer",
]


def budget(pages: int) -> tuple[int, int, int]:
    """(top_n, mmr_n, pool) for a knowledge base set with this many pages."""
    for limit, top_n, mmr_n, pool in BUDGETS:
        if limit is None or pages <= limit:
            return top_n, mmr_n, pool
    raise AssertionError("unreachable")


def mmr(
    query: Sequence[float], vectors: Sequence[Sequence[float]], k: int, selected: Sequence[int] = ()
) -> list[int]:
    """Maximal Marginal Relevance: up to k more indexes into `vectors`, each maximising
    MMR_LAMBDA * sim(query, v) - (1 - MMR_LAMBDA) * max sim(v, already selected)."""
    if not len(vectors) or k <= 0:
        return []
    m = np.asarray(vectors, dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    q = np.asarray(query, dtype=np.float32)
    rel = m @ (q / (np.linalg.norm(q) + 1e-9))
    sims = m @ m.T
    chosen = list(selected)
    picked: list[int] = []
    left = [i for i in range(len(m)) if i not in set(chosen)]
    while left and len(picked) < k:
        red = sims[np.ix_(left, chosen)].max(axis=1) if chosen else np.zeros(len(left))
        scores = MMR_LAMBDA * rel[left] - (1 - MMR_LAMBDA) * red
        best = left[int(np.argmax(scores))]
        picked.append(best)
        chosen.append(best)
        left.remove(best)
    return picked


def cosine_sims(query: Sequence[float], vectors: Sequence[Sequence[float]]) -> list[float]:
    if not len(vectors):
        return []
    m = np.asarray(vectors, dtype=np.float32)
    q = np.asarray(query, dtype=np.float32)
    return [float(x) for x in (m @ q) / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-9)]


def _plain(text: str) -> str:
    """Table chunks are HTML; the tags would eat the passage budget before the rows that hold the answer."""
    if "<t" not in text:
        return text
    text = re.sub(r"</t[dh]>", " | ", re.sub(r"</tr>", "\n", text))
    return re.sub(r"[ \t]+", " ", re.sub(r"<[^>]+>", "", text)).strip()


async def jev_scores(query: str, passages: list[str]) -> tuple[list[float], dict[str, Any]] | None:
    """One JEV call, one Score question per passage over a shared state (billed once). None if unavailable."""
    if not jev.enabled() or not passages:
        return None
    state = {
        "user_question": query,
        "passages": {f"p{i}": _plain(p)[:JEV_PASSAGE_CHARS] for i, p in enumerate(passages)},
    }
    questions = {
        f"p{i}": {
            "type": "score",
            "instructions": f"How useful is passage p{i} for answering the user question?",
            "criteria": JEV_CRITERIA,
        }
        for i in range(len(passages))
    }
    try:
        answers, usage = await jev.ask(state, questions, node="rag_select")
        return [float(answers[f"p{i}"]["score"]) for i in range(len(passages))], usage
    except Exception as e:  # noqa: BLE001 - selection falls back to retrieval order
        log.warning("jev_select_failed", error=str(e)[:200])
        return None
