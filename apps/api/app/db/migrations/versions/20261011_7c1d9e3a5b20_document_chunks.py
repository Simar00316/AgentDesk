"""document_chunks: full-text chunks of hierarchically indexed documents

Revision ID: 7c1d9e3a5b20
Revises: 4b40e2c8d02a
Create Date: 2026-10-11 00:30:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "7c1d9e3a5b20"
down_revision: str | None = "4b40e2c8d02a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_chunks",
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("kb_id", sa.Uuid(), nullable=False),
        sa.Column("document_version", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("section_title", sa.Text(), nullable=True),
        sa.Column("element_types", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", postgresql.ARRAY(postgresql.REAL()), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("org_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["org_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_document_chunks_org_id"), "document_chunks", ["org_id"], unique=False)
    op.create_index(
        "ix_document_chunks_doc_pages", "document_chunks", ["document_id", "document_version", "page_start"]
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_doc_pages", table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_org_id"), table_name="document_chunks")
    op.drop_table("document_chunks")
