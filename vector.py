from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List, Dict, Optional
import numpy as np
import re

import os

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
EMBED_MODEL = "bge-m3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DBS = {
    "law": {
        "path": "vectorstore_law",
        "collection": "indonesian_law_rag"
    },
    "culture": {
        "path": "vectorstore_culture",
        "collection": "balinese_culture_rag"
    }
}

VECTOR_TOP_K = 10
BM25_TOP_K   = 10
RERANK_TOP_K = 5

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# ─────────────────────────────
# EMBEDDINGS
# ─────────────────────────────
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# ─────────────────────────────
# VECTOR STORES
# ─────────────────────────────
vectorstores: Dict[str, Chroma] = {}

for name, cfg in DBS.items():
    vectorstores[name] = Chroma(
        persist_directory=cfg["path"],
        collection_name=cfg["collection"],
        embedding_function=embeddings
    )

# ─────────────────────────────
# HELPER: Extract canonical article ID from content
# ─────────────────────────────
def extract_article_number(content: str) -> Optional[str]:
    """Extract article number like '458' from content for dedup fallback."""
    patterns = [
        r"Article\s+(\d+)",
        r"\[ORIGINAL\s+Article\s+(\d+)\]",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25, stripping punctuation so that e.g. "murder?",
    "murder," and "murder." all collapse to the same token "murder".
    Using plain .lower().split() instead would treat those as different
    tokens and silently break keyword matching on punctuation-adjacent words.
    """
    return re.findall(r"\w+", text.lower())


def get_canonical_id(d: Document) -> str:
    """
    Get a canonical ID for deduplication.
    Priority:
    1. metadata.article_id (e.g., "penal_Article_458")
    2. metadata.id (e.g., "ceremony_1", "taboo_1")
    3. metadata.chunk_id
    4. metadata.doc_id (from BM25 retriever)
    5. Extracted article number from content + domain
    6. Fallback to content hash
    """
    meta = d.metadata or {}

    if meta.get("article_id"):
        return str(meta["article_id"])

    if meta.get("id"):
        return str(meta["id"])

    if meta.get("chunk_id"):
        return str(meta["chunk_id"])

    if meta.get("doc_id"):
        return str(meta["doc_id"])

    article_num = extract_article_number(d.page_content)
    if article_num:
        domain = meta.get("domain", "unknown")
        return f"{domain}_article_{article_num}"

    return d.page_content[:120].strip()


# ─────────────────────────────
# DIRECT ARTICLE LOOKUP (EXACT MATCH BY NUMBER)
# ─────────────────────────────
# Matches things like: "article 5", "Article 274", "pasal 5", "Pasal 274"
ARTICLE_NUMBER_PATTERN = re.compile(r"(?:article|pasal)\s+(\d+)", re.IGNORECASE)


def extract_requested_article_number(query: str) -> Optional[int]:
    """
    Look for an explicit article/pasal number in the user's question.
    Returns None if the question doesn't reference a specific article number
    (e.g. "explain the fine of category II" has no article number -> None,
    correctly falls through to semantic search instead of exact lookup).
    """
    match = ARTICLE_NUMBER_PATTERN.search(query)
    if not match:
        return None
    return int(match.group(1))


def direct_article_lookup(vectorstore: Chroma, domain: str, article_number: int) -> List[Document]:
    """
    Fetch the chunk(s) for an exact article number straight from Chroma's
    metadata, bypassing vector/BM25 similarity entirely. This guarantees a
    100% hit when the user names a specific article number, instead of
    relying on embeddings/BM25 to "guess" the right document — which is
    unreliable for short numeric identifiers (see ms-marco/bge debug session).
    """
    try:
        data = vectorstore.get(where={"article_number": article_number})
    except Exception:
        return []

    if not data or not data.get("documents"):
        return []

    results = []
    for doc_text, meta in zip(data["documents"], data["metadatas"]):
        meta = dict(meta or {})
        meta["domain"] = domain
        meta["retrieval_method"] = "direct_article_lookup"
        results.append(Document(page_content=doc_text, metadata=meta))

    return results


# ─────────────────────────────
# BM25 RETRIEVER
# ─────────────────────────────
class BM25Retriever:
    def __init__(self, vectorstore: Chroma, domain: str):
        self.domain = domain
        print(f"📊 Loading BM25 index: {domain}")

        data = vectorstore.get()
        self.docs: List[Document] = []

        for i in range(len(data["ids"])):
            meta = data["metadatas"][i] or {}
            doc_id = data["ids"][i]

            doc = Document(
                page_content=data["documents"][i],
                metadata={
                    **meta,
                    "doc_id": doc_id
                }
            )
            self.docs.append(doc)

        if not self.docs:
            raise ValueError(
                f"Vectorstore for domain '{domain}' is empty (0 documents). "
                f"Check the persist_directory/collection name, and make sure "
                f"this script is run from the correct working directory."
            )

        tokenized = [tokenize(d.page_content) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized)

        print(f"✅ BM25 ready: {domain} ({len(self.docs)} docs)")

    def invoke(self, query: str) -> List[Document]:
        scores = self.bm25.get_scores(tokenize(query))
        top_idx = np.argsort(scores)[::-1][:BM25_TOP_K]

        results = []
        for i in top_idx:
            if scores[i] > 0:
                doc = self.docs[i]
                doc.metadata["bm25_score"] = float(scores[i])
                doc.metadata["domain"] = self.domain
                results.append(doc)

        return results


# ─────────────────────────────
# CROSS ENCODER RERANKER
# ─────────────────────────────
class Reranker:
    def __init__(self):
        print("🎯 Loading reranker...")
        self.model = CrossEncoder(RERANKER_MODEL)
        print("✅ Reranker ready")

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs:
            return []

        pairs = [[query, d.page_content] for d in docs]
        scores = self.model.predict(pairs)

        order = np.argsort(scores)[::-1][:top_k]

        result = []
        for rank, idx in enumerate(order):
            doc = docs[idx]
            doc.metadata["rerank_score"] = float(scores[idx])
            doc.metadata["rerank_rank"] = rank + 1
            result.append(doc)

        return result


# ─────────────────────────────
# SHARED HELPER: retrieve + dedup (used by DomainRetriever & HybridRetriever)
# ─────────────────────────────
def retrieve_candidates(
    vectorstore: Chroma,
    bm25: BM25Retriever,
    query: str,
    domain: str
) -> Dict[str, Document]:
    """
    Fetch candidate documents from vector search + BM25 for a single domain,
    then dedup them by canonical_id. Returned as a dict {canonical_id: Document}
    so results can easily be merged across domains (used by HybridRetriever)
    without rewriting the dedup logic.

    Both the vector score and the BM25 score are stored in metadata
    ("vector_score" / "bm25_score") so they can be inspected later,
    e.g. in a debug print of retrieval results.

    Note: Chroma's similarity_search_with_score returns a DISTANCE-like
    value depending on the collection's configured distance function
    (commonly cosine distance) -> lower vector_score means MORE similar,
    unlike rerank_score where higher is better.
    """
    vec_results = vectorstore.similarity_search_with_score(query, k=VECTOR_TOP_K)

    vec_docs = []
    for doc, score in vec_results:
        doc.metadata["vector_score"] = float(score)
        vec_docs.append(doc)

    bm25_docs = bm25.invoke(query)

    candidates: Dict[str, Document] = {}

    for d in vec_docs + bm25_docs:
        d.metadata.setdefault("domain", domain)
        canonical_id = get_canonical_id(d)

        if canonical_id not in candidates:
            candidates[canonical_id] = d
        else:
            # Same document found from another source (vector vs BM25) -> merge scores
            existing = candidates[canonical_id]
            if "bm25_score" in d.metadata and "bm25_score" not in existing.metadata:
                existing.metadata["bm25_score"] = d.metadata["bm25_score"]
            if "vector_score" in d.metadata and "vector_score" not in existing.metadata:
                existing.metadata["vector_score"] = d.metadata["vector_score"]

    return candidates


# ─────────────────────────────
# DOMAIN RETRIEVER (INDIVIDUAL)
# ─────────────────────────────
class DomainRetriever:
    def __init__(self, vectorstore, bm25, reranker, domain: str):
        self.vectorstore = vectorstore
        self.bm25 = bm25
        self.reranker = reranker
        self.domain = domain

    def invoke(self, query: str) -> List[Document]:
        # STEP 1: if the user names a specific article/pasal number,
        # try an exact metadata lookup first. This sidesteps the
        # unreliability of vector/BM25 similarity for short numeric
        # identifiers entirely (e.g. "article 5" matching unrelated
        # docs that merely contain the digit "5" somewhere).
        article_number = extract_requested_article_number(query)
        if article_number is not None:
            direct_hits = direct_article_lookup(self.vectorstore, self.domain, article_number)
            if direct_hits:
                return direct_hits
            
            # If we're searching culture but looking for an article, skip semantic search
            if self.domain == "culture":
                return []
                
            # If nothing found by exact number in law, fall through to semantic
            # search below -- e.g. user might have mistyped the number,
            # or be asking a conceptual question that merely contains a digit.

        # STEP 2: normal hybrid semantic search (vector + BM25 + rerank)
        candidates = retrieve_candidates(
            self.vectorstore, self.bm25, query, self.domain
        )
        return self.reranker.rerank(query, list(candidates.values()), RERANK_TOP_K)


# ─────────────────────────────
# HYBRID MULTI DOMAIN RETRIEVER
# ─────────────────────────────
class HybridRetriever:
    def __init__(self):
        print("\n🔧 Init Hybrid Retriever (MULTI DOMAIN)")

        self.vectorstores = vectorstores
        self.reranker = Reranker()

        self.bm25 = {
            name: BM25Retriever(vs, name)
            for name, vs in vectorstores.items()
        }

        print("✅ System ready (law + culture)\n")

    def invoke(self, query: str) -> List[Document]:
        # Direct article lookup across all domains first (cheap + exact).
        article_number = extract_requested_article_number(query)
        if article_number is not None:
            direct_hits: List[Document] = []
            for domain, vs in self.vectorstores.items():
                direct_hits.extend(direct_article_lookup(vs, domain, article_number))
            if direct_hits:
                return direct_hits

        # This method used to reimplement the entire retrieve+dedup logic
        # (duplicated with DomainRetriever.invoke). Now it simply calls
        # retrieve_candidates() per domain and merges the results.
        all_candidates: Dict[str, Document] = {}

        for domain, vs in self.vectorstores.items():
            domain_candidates = retrieve_candidates(
                vs, self.bm25[domain], query, domain
            )

            for canonical_id, d in domain_candidates.items():
                if canonical_id in all_candidates:
                    existing = all_candidates[canonical_id]
                    if "bm25_score" in d.metadata and "bm25_score" not in existing.metadata:
                        existing.metadata["bm25_score"] = d.metadata["bm25_score"]
                    if "vector_score" in d.metadata and "vector_score" not in existing.metadata:
                        existing.metadata["vector_score"] = d.metadata["vector_score"]
                else:
                    all_candidates[canonical_id] = d

        return self.reranker.rerank(query, list(all_candidates.values()), RERANK_TOP_K)

    def get_domain_retriever(self, domain: str) -> DomainRetriever:
        if domain not in self.vectorstores:
            raise ValueError(f"Domain not found: {domain}")

        return DomainRetriever(
            vectorstore=self.vectorstores[domain],
            bm25=self.bm25[domain],
            reranker=self.reranker,
            domain=domain
        )


# ─────────────────────────────
# GLOBAL INSTANCE
# ─────────────────────────────
retriever = HybridRetriever()

# DIRECT ACCESS
retriever_law = retriever.get_domain_retriever("law")
retriever_culture = retriever.get_domain_retriever("culture")