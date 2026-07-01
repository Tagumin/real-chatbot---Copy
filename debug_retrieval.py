"""
debug_retrieval.py
==================
Script debug untuk pipeline retrieval hybrid RAG.
Menampilkan dan menyimpan ke file TXT:
  ✔ Hasil BM25 (sparse retrieval)
  ✔ Hasil Vector Search (dense retrieval)
  ✔ Hasil setelah Cross Encoder Reranker

Cara pakai:
    python debug_retrieval.py

Output disimpan di folder: debug_output/
"""

import os
import re
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ─────────────────────────────────────────────
# CONFIG  (sesuaikan jika berbeda)
# ─────────────────────────────────────────────
EMBED_MODEL     = "bge-m3"
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "debug_output")

DBS = {
    "law": {
        "path": os.path.join(BASE_DIR, "vectorstore_law"),
        "collection": "indonesian_law_rag",
    },
    "culture": {
        "path": os.path.join(BASE_DIR, "vectorstore_culture"),
        "collection": "balinese_culture_rag",
    },
}

VECTOR_TOP_K = 10
BM25_TOP_K   = 10
RERANK_TOP_K = 5


# ─────────────────────────────────────────────
# SETUP OUTPUT DIR
# ─────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def sep(char="=", width=80) -> str:
    return char * width


def section(title: str) -> str:
    line = sep()
    return f"\n{line}\n  {title}\n{line}"


def fmt_doc(idx: int, doc: Document, extra_scores: Dict = None) -> str:
    """Format satu dokumen menjadi blok teks yang mudah dibaca."""
    meta = doc.metadata or {}
    lines = [
        f"\n  [{idx}] ─────────────────────────────────────────────",
        f"  Domain          : {meta.get('domain', 'N/A')}",
        f"  Article ID      : {meta.get('article_id') or meta.get('id') or meta.get('chunk_id') or 'N/A'}",
        f"  Article Number  : {meta.get('article_number', 'N/A')}",
    ]

    # Skor retrieval
    if extra_scores:
        for k, v in extra_scores.items():
            val = f"{v:.6f}" if isinstance(v, float) else str(v)
            lines.append(f"  {k:<16}: {val}")

    # Skor dari metadata doc itu sendiri
    for key in ["vector_score", "bm25_score", "rerank_score", "rerank_rank"]:
        if key in meta:
            val = meta[key]
            disp = f"{val:.6f}" if isinstance(val, float) else str(val)
            if not extra_scores or key not in extra_scores:
                lines.append(f"  {key:<16}: {disp}")

    lines.append(f"  Retrieval Method: {meta.get('retrieval_method', 'N/A')}")
    lines.append(f"\n  Content (preview 400 chars):")
    content_preview = doc.page_content.strip().replace("\n", " ")[:400]
    lines.append(f"    {content_preview}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# LOAD COMPONENTS
# ─────────────────────────────────────────────
def load_components():
    print("⏳ Memuat embedding model...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    print("⏳ Memuat vectorstores...")
    vectorstores: Dict[str, Chroma] = {}
    for name, cfg in DBS.items():
        vectorstores[name] = Chroma(
            persist_directory=cfg["path"],
            collection_name=cfg["collection"],
            embedding_function=embeddings,
        )
        count = vectorstores[name]._collection.count()
        print(f"  ✅ [{name}] {count} dokumen")

    print("⏳ Membangun BM25 index...")
    bm25_retrievers: Dict[str, "BM25IndexWrapper"] = {}
    for name, vs in vectorstores.items():
        bm25_retrievers[name] = BM25IndexWrapper(vs, name)

    print("⏳ Memuat Cross Encoder Reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)
    print("  ✅ Reranker siap\n")

    return vectorstores, bm25_retrievers, reranker


class BM25IndexWrapper:
    def __init__(self, vectorstore: Chroma, domain: str):
        self.domain = domain
        data = vectorstore.get()
        self.docs: List[Document] = []
        for i in range(len(data["ids"])):
            meta = dict(data["metadatas"][i] or {})
            doc = Document(
                page_content=data["documents"][i],
                metadata={**meta, "doc_id": data["ids"][i]},
            )
            self.docs.append(doc)
        tokenized = [tokenize(d.page_content) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  ✅ BM25 [{domain}] {len(self.docs)} dokumen")

    def retrieve(self, query: str) -> List[Document]:
        scores = self.bm25.get_scores(tokenize(query))
        top_idx = np.argsort(scores)[::-1][:BM25_TOP_K]
        results = []
        for i in top_idx:
            if scores[i] > 0:
                import copy
                doc = copy.deepcopy(self.docs[i])
                doc.metadata["bm25_score"] = float(scores[i])
                doc.metadata["domain"] = self.domain
                doc.metadata["retrieval_method"] = "bm25"
                results.append(doc)
        return results


# ─────────────────────────────────────────────
# RETRIEVE FUNCTIONS
# ─────────────────────────────────────────────
def retrieve_vector(vectorstore: Chroma, query: str, domain: str) -> List[Document]:
    """Ambil top-K dokumen via dense vector similarity search."""
    import copy
    results_with_scores = vectorstore.similarity_search_with_score(query, k=VECTOR_TOP_K)
    docs = []
    for doc, score in results_with_scores:
        d = copy.deepcopy(doc)
        d.metadata["vector_score"] = float(score)
        d.metadata["domain"] = domain
        d.metadata["retrieval_method"] = "vector"
        docs.append(d)
    return docs


def rerank_docs(reranker: CrossEncoder, query: str, docs: List[Document]) -> List[Document]:
    """Jalankan cross-encoder reranking terhadap candidate docs."""
    import copy
    if not docs:
        return []
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    order = np.argsort(scores)[::-1][:RERANK_TOP_K]

    result = []
    for rank, idx in enumerate(order):
        d = copy.deepcopy(docs[idx])
        d.metadata["rerank_score"] = float(scores[idx])
        d.metadata["rerank_rank"] = rank + 1
        result.append(d)
    return result


def deduplicate(docs: List[Document]) -> List[Document]:
    """Deduplikasi dokumen berdasarkan konten (page_content[:120])."""
    seen = {}
    for d in docs:
        key = d.page_content[:120].strip()
        if key not in seen:
            seen[key] = d
        else:
            existing = seen[key]
            # Merge scores jika dokumen sama datang dari dua sumber
            for score_key in ["bm25_score", "vector_score"]:
                if score_key in d.metadata and score_key not in existing.metadata:
                    existing.metadata[score_key] = d.metadata[score_key]
    return list(seen.values())


# ─────────────────────────────────────────────
# FORMAT REPORT
# ─────────────────────────────────────────────
def build_report(
    query: str,
    domain: str,
    bm25_docs: List[Document],
    vector_docs: List[Document],
    reranked_docs: List[Document],
    timestamp: str,
) -> str:
    lines = []

    lines.append(sep())
    lines.append("  RETRIEVAL DEBUG REPORT")
    lines.append(sep())
    lines.append(f"  Timestamp : {timestamp}")
    lines.append(f"  Query     : {query}")
    lines.append(f"  Domain    : {domain}")
    lines.append(f"  Config    : VECTOR_TOP_K={VECTOR_TOP_K}, BM25_TOP_K={BM25_TOP_K}, RERANK_TOP_K={RERANK_TOP_K}")
    lines.append(sep())

    # ── BAGIAN 1: BM25 ──────────────────────────────────────────────────────
    lines.append(section(f"BAGIAN 1 ─ HASIL BM25 SPARSE RETRIEVAL  ({len(bm25_docs)} dokumen)"))
    lines.append("  Dokumen diurutkan berdasarkan BM25 Okapi score (semakin tinggi = semakin relevan).")
    if not bm25_docs:
        lines.append("\n  (Tidak ada dokumen ditemukan)")
    for i, doc in enumerate(bm25_docs):
        lines.append(fmt_doc(i + 1, doc))

    # ── BAGIAN 2: VECTOR ─────────────────────────────────────────────────────
    lines.append(section(f"BAGIAN 2 ─ HASIL VECTOR DENSE RETRIEVAL  ({len(vector_docs)} dokumen)"))
    lines.append("  Dokumen diurutkan berdasarkan cosine distance (semakin KECIL = semakin mirip).")
    if not vector_docs:
        lines.append("\n  (Tidak ada dokumen ditemukan)")
    for i, doc in enumerate(vector_docs):
        lines.append(fmt_doc(i + 1, doc))

    # ── BAGIAN 3: CROSS ENCODER RERANKER ─────────────────────────────────────
    lines.append(section(f"BAGIAN 3 ─ HASIL CROSS ENCODER RERANKER  (top {RERANK_TOP_K} dari gabungan BM25+Vector)"))
    lines.append("  Input: gabungan (deduplikasi) BM25 + Vector results.")
    lines.append("  Output: top-K dokumen paling relevan menurut Cross Encoder score.")
    if not reranked_docs:
        lines.append("\n  (Tidak ada dokumen ditemukan)")
    for i, doc in enumerate(reranked_docs):
        lines.append(fmt_doc(i + 1, doc))

    # ── RINGKASAN ─────────────────────────────────────────────────────────────
    lines.append(section("RINGKASAN"))
    lines.append(f"  BM25 retrieved   : {len(bm25_docs)} dokumen")
    lines.append(f"  Vector retrieved : {len(vector_docs)} dokumen")
    total_combined = len(deduplicate(bm25_docs + vector_docs))
    lines.append(f"  Setelah dedup    : {total_combined} kandidat unik")
    lines.append(f"  Reranker output  : {len(reranked_docs)} dokumen")

    if reranked_docs:
        lines.append(f"\n  Top-{len(reranked_docs)} Reranker Output:")
        lines.append(f"  {'Rank':<6} {'Rerank Score':<16} {'BM25 Score':<16} {'Vector Score':<16} {'Domain':<10} Article")
        lines.append(f"  {sep('-', 78)}")
        for doc in reranked_docs:
            meta = doc.metadata
            rank        = meta.get("rerank_rank", "-")
            rscore      = f"{meta['rerank_score']:.5f}" if "rerank_score" in meta else "N/A"
            bscore      = f"{meta['bm25_score']:.5f}"  if "bm25_score"   in meta else "N/A"
            vscore      = f"{meta['vector_score']:.5f}" if "vector_score" in meta else "N/A"
            dom         = meta.get("domain", "N/A")
            art         = meta.get("article_id") or meta.get("id") or meta.get("article_number") or "N/A"
            lines.append(f"  {rank:<6} {rscore:<16} {bscore:<16} {vscore:<16} {dom:<10} {art}")

    lines.append(f"\n{sep()}")
    lines.append("  END OF REPORT")
    lines.append(sep())

    return "\n".join(lines)


# ─────────────────────────────────────────────
# SAVE TO FILE
# ─────────────────────────────────────────────
def save_report(report: str, query: str, domain: str, timestamp: str) -> str:
    safe_query = re.sub(r"[^\w\s-]", "", query)[:40].strip().replace(" ", "_")
    filename = f"debug_{domain}_{safe_query}_{timestamp}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    return filepath


# ─────────────────────────────────────────────
# MAIN DEBUG RUNNER
# ─────────────────────────────────────────────
def run_debug(query: str, domain: str, vectorstores, bm25_retrievers, reranker):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if domain not in vectorstores:
        print(f"❌ Domain '{domain}' tidak ditemukan. Pilih: {list(vectorstores.keys())}")
        return

    vs   = vectorstores[domain]
    bm25 = bm25_retrievers[domain]

    print(f"\n🔍 Query    : {query}")
    print(f"   Domain   : {domain}")
    print(f"   Timestamp: {timestamp}\n")

    # ── Step 1: BM25 ──
    print("📊 [Step 1/3] Menjalankan BM25 retrieval...")
    bm25_docs = bm25.retrieve(query)
    print(f"   → {len(bm25_docs)} dokumen ditemukan")

    # ── Step 2: Vector ──
    print("🧲 [Step 2/3] Menjalankan Vector (dense) retrieval...")
    vector_docs = retrieve_vector(vs, query, domain)
    print(f"   → {len(vector_docs)} dokumen ditemukan")

    # ── Step 3: Deduplicate + Rerank ──
    print("🎯 [Step 3/3] Menjalankan Cross Encoder Reranker...")
    combined = deduplicate(bm25_docs + vector_docs)
    print(f"   → Kandidat unik (BM25 ∪ Vector): {len(combined)}")
    reranked_docs = rerank_docs(reranker, query, combined)
    print(f"   → Reranker output: {len(reranked_docs)} dokumen")

    # ── Build Report ──
    report = build_report(query, domain, bm25_docs, vector_docs, reranked_docs, timestamp)

    # ── Print to Console ──
    print("\n" + report)

    # ── Save to File ──
    filepath = save_report(report, query, domain, timestamp)
    print(f"\n✅ Laporan disimpan ke: {filepath}")
    return filepath


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  RETRIEVAL DEBUG TOOL")
    print("  BM25 | Vector | Cross Encoder Reranker")
    print("=" * 60)

    # Muat semua komponen sekali di awal
    vectorstores, bm25_retrievers, reranker = load_components()

    print("\nKetik 'exit' untuk keluar.")
    print("Domain tersedia: " + ", ".join(vectorstores.keys()))

    while True:
        print()
        query = input("❓ Query > ").strip()
        if query.lower() in ("exit", "quit", "q"):
            print("👋 Selesai.")
            sys.exit(0)
        if not query:
            continue

        domain = input("🌐 Domain (law / culture) > ").strip().lower()
        if domain not in vectorstores:
            print(f"⚠️  Domain tidak valid. Pilih: {list(vectorstores.keys())}")
            continue

        run_debug(query, domain, vectorstores, bm25_retrievers, reranker)
        print("\n" + "─" * 60)
        print("Siap untuk query berikutnya ↑")
