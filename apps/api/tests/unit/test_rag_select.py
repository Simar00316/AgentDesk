"""Retrieval budgets, MMR and page grouping for hierarchical indexing."""

from __future__ import annotations

from app.rag.chunking import Chunk
from app.rag.enrich import group_chunks, is_hierarchical, pages_per_unit
from app.rag.select import _plain, budget, mmr


def test_budget_grows_with_pages() -> None:
    assert budget(10) == (6, 4, 20)
    assert budget(30) == (6, 4, 20)
    assert budget(31) == (8, 8, 40)
    assert budget(400) == (10, 12, 60)
    assert budget(5000) == (12, 18, 100)


def test_mmr_skips_near_duplicates() -> None:
    q = [1.0, 1.0, 0.0]
    vecs = [[1.0, 0.0, 0.0], [1.0, 0.05, 0.0], [0.0, 1.0, 0.0]]  # 0 and 1 are near-duplicates; 2 is different
    assert mmr(q, vecs, 2) == [1, 2]  # top relevance first, then the relevant non-duplicate
    assert mmr(q, vecs, 1, selected=[1]) == [2]
    assert mmr(q, [], 3) == []


def test_hierarchical_threshold_and_grouping() -> None:
    assert not is_hierarchical(30) and is_hierarchical(31) and not is_hierarchical(None)
    assert [pages_per_unit(n) for n in (31, 40, 41, 80, 81, 120, 121, 500)] == [1, 1, 2, 2, 3, 3, 4, 4]
    chunks = [
        Chunk(text=f"c{p}", chunk_index=i, page_start=p, page_end=p, section_title=None)
        for i, p in enumerate([1, 2, 2, 5, 6])
    ]
    # 81 pages -> 3 pages per entry; pages 3-4 have no chunk -> that group is skipped
    assert group_chunks(chunks, 81) == [(1, 3, [0, 1, 2]), (4, 6, [3, 4])]


def test_jev_passages_drop_table_markup() -> None:
    html = "<table><tr><td>LLAMA 2</td><td>70B</td><td>1.5e-4</td></tr><tr><td/><td>7B</td></tr></table>"
    assert _plain(html) == "LLAMA 2 | 70B | 1.5e-4 | \n7B |"
    assert _plain("plain text <3") == "plain text <3"
