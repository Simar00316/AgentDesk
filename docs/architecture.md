# Architecture

AgentDesk is a multi-tenant SaaS: a Next.js web app talks to a FastAPI API; a separate webhook process receives WhatsApp events and hands them to an ARQ worker; PostgreSQL holds application data, Redis handles queues/rate limits/locks/caches, Qdrant stores vectors, MinIO/S3 stores files and DuckDB datasets. LLM calls go through LiteLLM so platform keys, customer keys and custom OpenAI-compatible endpoints share one code path.

```
browser ──> web (Next.js) ──> api (FastAPI) ──┬─> postgres (app data)
                                              ├─> redis (queues, rate limits, locks, caches)
                                              ├─> qdrant (chunks, hybrid dense+BM25)
                                              ├─> minio/s3 (documents, duckdb files, exports)
                                              ├─> embeddings (text-embeddings-inference, optional)
                                              └─> LLM providers via LiteLLM, Tavily
Meta Graph API ──> webhook (FastAPI) ──> redis queue ──> worker (ARQ) ──> agent graph ──> Meta send API
```

## Request flows

**Service API** (`POST /v1/service/chat`, Bearer `adk_` key): key hash lookup → per-key rate limit → the agent's latest published version → same `prepare_run` / graph / persistence path as the Playground with channel `api`, returning JSON or SSE. See [service-api.md](service-api.md).


**Playground message** (`POST /v1/orgs/{org}/playground/sessions/{sid}/messages`, SSE): membership + CSRF + rate limit → load draft config (Redis 60 s cache, busted on save) or a published version → load trimmed history → `prepare_run` builds a `RunContext` (models resolved, dataset context, KB summaries) → the LangGraph run streams `route`, `tool_start`, `tool_result`, `token`, `citations`, `guardrail`, `usage`, `done` events → messages, run trace and usage events are persisted.

**WhatsApp inbound**: `POST /webhooks/whatsapp/{deployment_id}` verifies `X-Hub-Signature-256` with the deployment's app secret (constant-time), dedupes by `wamid` (Redis `SETNX`, 48 h), drops stale (>10 min) events and enqueues `process_whatsapp_message`, answering 200 in milliseconds. The worker serialises per contact with a Redis lock, applies per-contact/per-deployment limits, marks the message read with a typing indicator, runs the graph (non-streaming), formats Markdown → WhatsApp, splits at ≤4,000 chars and sends with retries.

## Agent runtime (`app/runtime`)

One compiled LangGraph handles 0, 1 or many tools:

`input_guardrails → (blocked_response) | annotation_reply → (curated answer) | query_rewrite → router → [rag_retrieve ‖ sql_agent ‖ web_search] → collect_context → generate_answer → output_guardrails → channel_formatter`

- The router is skipped logically for 0/1 tools (fixed decisions) and uses the utility model with structured output for ≥2 tools; multi-tool routes fan out with `Send` and join in `collect_context`, which also applies the fallback chain KB → web → no-answer.
- `sql_agent` invokes the SQL subgraph (`app/sql/agent.py`): intent check → schema selection (≤15 tables inline, otherwise embedded table cards + FK hop) → structured SQL generation → `validate_sql` → optional `interrupt()` for confirm-before-run → read-only execution → reflection/retry (max 3 attempts).
- `annotation_reply` compares the message with the agent's curated Q&A (cosine on cached embeddings) and, above the threshold, answers verbatim and skips straight to the formatter. See [annotations.md](annotations.md).
- Prompts are assembled in a fixed order (platform preamble → agent prompt → guardrails → context → history → message) with a token budget (history ≥2 turns kept → chunks → SQL rows → web results trimmed).
- Non-serialisable dependencies travel in `config["configurable"]["ctx"]`; state is a `TypedDict` with reducers for parallel branches.
- Checkpointer is `MemorySaver` (single process). Use `langgraph-checkpoint-postgres` to run confirm-before-run across multiple API workers.

## Knowledge base (`app/rag`)

Presigned PUT to S3 → magic-byte validation → parse → caption pictures → `chunk_by_title` (1,500/1,200/200/300 defaults) → contextual header → index.

- **Parsing.** PDFs use open-source Unstructured `hi_res` when the `layout` extra plus poppler and tesseract are installed (the worker image ships them): a layout model finds titles, text, tables (structure inferred to HTML) and pictures. Without it, pdfminer with title/table heuristics. DOCX/MD/TXT use Unstructured's partitioners. Page numbers are kept throughout.
- **Pictures.** Each extracted picture (≥ ~4.5 KB, at most 60 per document) is captioned by the utility model (vision) as chart / diagram / photo with a title and summary. Logos and decorative images are dropped; captions become searchable text.
- **Flat index (≤ 30 pages).** Chunks are embedded (dense, Redis-cached + BM25 sparse) into Qdrant.
- **Hierarchical index (31+ pages).** Pages are grouped (1 page per entry up to 40 pages, then 2/3/4 so a document has ~40 entries). The utility model writes one JSON index entry per group (title, summary, questions it answers, topics, tables, images, graphs), which is embedded into Qdrant with its page range. The full-text chunks and their dense vectors go to the Postgres `document_chunks` table.

Retrieval (`retrieval.py`, `select.py`) is two-stage. Stage 1: hybrid RRF search (dense-cosine threshold decides relevance). The candidates are the relevance top-n plus MMR picks from a larger pool, all scaled by the total pages in the attached KBs. JEV scores the candidates (one Score question per passage, shared state) and the top 5 are kept. Stage 2: chosen page summaries expand to their pages' chunks from Postgres, which are ranked by cosine + MMR. JEV scores them again and the top 5 scoring at least 0.5/3 go to the LLM. Without `JEV_API_KEY`, the optional Cohere/local reranker or retrieval order selects instead. Tenant filters: Qdrant payload indexes (`org_id` is_tenant, `kb_id`, `document_id`) and `org_id` on every Postgres chunk query.

## Datasets (`app/sql`)

Postgres introspection reads `pg_catalog`/`information_schema` (tables, views, columns, PKs, FKs, indexes, comments, enums, samples with sensitive-name skipping and cardinality checks) into a versioned snapshot; `apply_snapshot` diffs versions and preserves descriptions. Spreadsheets are previewed with pandas, typed and loaded into a per-dataset DuckDB file. The validator (`validator.py`) parses with sqlglot, rejects anything but a single read-only SELECT, blocks catalogs/functions/table functions, enforces the exposed-table allowlist and hidden columns (star expansion via `qualify`), and injects/caps LIMIT. Execution runs `BEGIN READ ONLY` with 15 s statement timeouts (Postgres) or a read-only DuckDB connection with external access disabled.

## Security

See [security.md](security.md) and the ADRs in [adr/](adr/). Cross-tenant tests parametrise every org-scoped endpoint.

## Observability

structlog JSON logs with secret redaction and request ids, Prometheus metrics at `/metrics`, optional LangSmith tracing via LiteLLM callbacks (`LANGSMITH_TRACING=true`), Sentry DSN hooks in config. `/readyz` checks Postgres, Redis, Qdrant and S3.
