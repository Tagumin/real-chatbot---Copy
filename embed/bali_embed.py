#!/usr/bin/env python3
"""
Embedding pipeline for Balinese culture dataset → ChromaDB
Production-ready version:
- Clean batching
- Safe string handling
- Stable Chroma initialization
- Better logging
- Adds `id` metadata so get_canonical_id() resolves to ceremony_X / taboo_X
  instead of falling through to UUIDs / content[:120].
"""

import json
import argparse
import os
import shutil
from typing import List

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm


# ─────────────────────────────
# CONFIG
# ─────────────────────────────
DEFAULT_JSON_FILE = "data/bali.json"
VECTOR_DB = "./vectorstore_culture"
COLLECTION_NAME = "balinese_culture_rag"
EMBED_MODEL = "bge-m3"
BATCH_SIZE = 32


# ─────────────────────────────
# LOAD + TRANSFORM DATA
# ─────────────────────────────
def load_documents(json_path: str) -> List[Document]:
    """Convert structured JSON into LangChain Documents"""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs: List[Document] = []

    # =========================
    # CEREMONIES
    # =========================
    for ceremony in data.get("ceremonies", []):
        ceremony_name = ceremony.get("ceremony_name", "")
        category = ceremony.get("category", "")
        title = ceremony.get("title", "")

        # ── FIX: pull the canonical id from metadata ──
        ceremony_meta = ceremony.get("metadata", {}) or {}
        canonical_id = ceremony_meta.get("id", "")

        for section_name, section_text in ceremony.get("text", {}).items():
            if not section_text:
                continue

            docs.append(
                Document(
                    page_content=f"{title} - {section_name.replace('_', ' ').title()}\n\n{section_text}",
                    metadata={
                        "id": canonical_id,                        # ← ADDED
                        "type": "ceremony",
                        "category": category,
                        "ceremony_name": ceremony_name,
                        "title": title,
                        "section": section_name,
                        "frequency": ceremony_meta.get("frequency", ""),
                        "tags": ",".join(ceremony_meta.get("tags", []) or []),
                        "domain": "culture",
                    },
                )
            )

    # =========================
    # TABOOS
    # =========================
    for taboo in data.get("taboos", []):
        title = taboo.get("title", "")
        category = taboo.get("category", "")
        text = taboo.get("text", "")

        metadata = taboo.get("metadata", {}) or {}

        # ── FIX: pull the canonical id from metadata ──
        canonical_id = metadata.get("id", "")

        content_parts = [
            f"TABOO: {title}",
            f"\n{text}",
        ]

        if metadata.get("rationale"):
            content_parts.append(f"\nRationale: {metadata['rationale']}")

        if metadata.get("consequence"):
            content_parts.append(f"\nConsequence: {metadata['consequence']}")

        if metadata.get("protocol_guide"):
            content_parts.append(f"\nProtocol: {metadata['protocol_guide']}")

        docs.append(
            Document(
                page_content="\n".join(content_parts),
                metadata={
                    "id": canonical_id,                            # ← ADDED
                    "type": "taboo",
                    "category": category,
                    "title": title,
                    "rules_no": str(taboo.get("rules", "")),
                    "severity": metadata.get("severity", "high"),
                    "related_ceremony": ",".join(taboo.get("related_ceremony", []) or []),
                    "domain": "culture",
                },
            )
        )

    return docs


# ─────────────────────────────
# VECTOR DB INITIALIZATION
# ─────────────────────────────
def init_vectorstore(reset: bool, embeddings: OllamaEmbeddings) -> Chroma:
    """Initialize or reset ChromaDB"""

    if reset and os.path.exists(VECTOR_DB):
        shutil.rmtree(VECTOR_DB)
        print("🗑️  Old vector DB removed")

    return Chroma(
        persist_directory=VECTOR_DB,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


# ─────────────────────────────
# EMBEDDING PIPELINE
# ─────────────────────────────
def embed_documents(docs: List[Document], vectorstore: Chroma):
    """Batch embedding into ChromaDB"""

    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Embedding"):
        batch = docs[i : i + BATCH_SIZE]
        vectorstore.add_documents(batch)

    print(f"✅ Embedded {len(docs)} documents")


# ─────────────────────────────
# TEST QUERY
# ─────────────────────────────
def test_query(vectorstore: Chroma):
    print("\n🧪 Running similarity test...")

    results = vectorstore.similarity_search(
        "What should I wear to a temple?", k=3
    )

    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}")
        print(f"ID: {doc.metadata.get('id')}")                       # ← ADDED to verify fix
        print(f"Type: {doc.metadata.get('type')}")
        print(f"Title: {doc.metadata.get('title')}")
        print(f"Preview: {doc.page_content[:120]}...")


# ─────────────────────────────
# MAIN
# ─────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=DEFAULT_JSON_FILE)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    print("\n🚀 Starting Embedding Pipeline")
    print(f"Model: {EMBED_MODEL}")
    print(f"DB: {VECTOR_DB}")
    print(f"Collection: {COLLECTION_NAME}")

    # 1. Load documents
    docs = load_documents(args.json)

    print(f"\n📄 Loaded documents: {len(docs)}")
    print(f"- Ceremonies: {sum(1 for d in docs if d.metadata['type'] == 'ceremony')}")
    print(f"- Taboos: {sum(1 for d in docs if d.metadata['type'] == 'taboo')}")

    # ── Safety check: make sure every doc has an id ──
    missing = [i for i, d in enumerate(docs) if not d.metadata.get("id")]
    if missing:
        print(f"\n⚠️  WARNING: {len(missing)} document(s) missing 'id' metadata!")
        print(f"   First few indexes: {missing[:5]}")
        print(f"   Check your JSON file — these will fall back to UUIDs at retrieval.")
    else:
        print(f"✅ All {len(docs)} documents have canonical `id` metadata")

    # 2. Embeddings
    print("\n🔮 Initializing embedding model...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # 3. Vector DB
    vectorstore = init_vectorstore(args.reset, embeddings)

    # 4. Embed
    print("\n💾 Embedding documents...")
    embed_documents(docs, vectorstore)

    # 5. Test
    test_query(vectorstore)

    print("\n🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main()