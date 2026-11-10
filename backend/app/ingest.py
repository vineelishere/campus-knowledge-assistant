"""Ingestion pipeline.

Usage (from the backend/ directory):
    python -m app.ingest ../data/manifest.csv

manifest.csv columns:
    path,title,doc_type,policy_key,department,year,access_level,source_url

* year and policy_key are set by you, not guessed: "which version supersedes which"
  is an academic-office decision, and it drives the outdated-document detection.
* access_level: 0 = all logged-in users, 1 = professors + admins, 2 = admins only.
* Paths are relative to the manifest's folder.
"""
import csv
import hashlib
import sys
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from .db import connection
from .models import embed_passages


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chunk_page(chunk) -> int | None:
    pages = []
    for item in getattr(chunk.meta, "doc_items", []) or []:
        for prov in getattr(item, "prov", []) or []:
            pages.append(prov.page_no)
    return min(pages) if pages else None


def ingest_one(row: dict, base: Path, converter: DocumentConverter, chunker: HybridChunker) -> str:
    path = (base / row["path"]).resolve()
    digest = _file_hash(path)

    with connection() as conn:
        exists = conn.execute("SELECT id FROM documents WHERE content_hash = %s", (digest,)).fetchone()
        if exists:
            return f"skip (already ingested): {row['title']}"

    result = converter.convert(str(path))
    chunks = list(chunker.chunk(dl_doc=result.document))
    if not chunks:
        return f"no text extracted (scanned PDF? enable OCR): {row['title']}"

    # Embed the contextualised text (headings + body) but store the plain text for display/citation.
    embed_inputs = [chunker.contextualize(chunk=c) for c in chunks]
    vectors = embed_passages(embed_inputs)

    year = int(row["year"])
    level = int(row.get("access_level") or 0)
    dept = row.get("department") or None

    with connection() as conn:
        doc_id = conn.execute(
            """INSERT INTO documents (title, source_path, source_url, doc_type, policy_key,
                                      department, year, access_level, content_hash)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (row["title"], str(path), row.get("source_url") or None, row.get("doc_type") or "policy",
             row.get("policy_key") or None, dept, year, level, digest),
        ).fetchone()["id"]

        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (document_id, chunk_index, page, text, department, year,
                                       access_level, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                [
                    (doc_id, i, _chunk_page(c), c.text, dept, year, level, vectors[i])
                    for i, c in enumerate(chunks)
                ],
            )
        conn.commit()
    return f"ok: {row['title']} ({len(chunks)} chunks)"


def main(manifest: str) -> None:
    manifest_path = Path(manifest).resolve()
    converter = DocumentConverter()
    chunker = HybridChunker()
    with manifest_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            print(ingest_one(row, manifest_path.parent, converter, chunker))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
