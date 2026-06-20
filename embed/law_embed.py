"""
embed.py
========
Load pre-chunked JSON → Embed with bge-m3 (Ollama) → Save to ChromaDB

Usage:
    python embed.py          (Normal run, append to existing DB)
    python embed.py --reset  (Delete old collection and start from scratch)
"""

import json
import argparse
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
# Replace with the path to your chunking output JSON file
JSON_FILE       = "data/data.json" 
VECTOR_DB       = "./vectorstore_law"
COLLECTION_NAME = "indonesian_law_rag"
EMBED_MODEL     = "bge-m3"
BATCH_SIZE      = 32  # Optimal batch size for Ollama
# ─────────────────────────────────────────────


def load_chunks(json_path: str) -> list[Document]:
    documents = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for article in data:

        base_metadata = article.get("metadata", {})

        for chunk in article.get("chunks", []):

            # HANYA COMBINED
            if chunk.get("chunk_type") != "combined":
                continue

            doc = Document(
                page_content=chunk.get("content", ""),
                metadata={
                    "article_id": article.get("article_id"),
                    "article_number": article.get("article_number"),
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_type": chunk.get("chunk_type"),
                    "source_article": chunk.get("source_article"),
                    "source": base_metadata.get("source"),
                    "book": base_metadata.get("book"),
                    "chapter": base_metadata.get("chapter"),
                }
            )

            documents.append(doc)

    return documents

def main():
    parser = argparse.ArgumentParser(description="Embed chunks to ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Reset vector database before embedding")
    args = parser.parse_args()

    print("=" * 60)
    print("   🚀  EMBEDDING — Indonesian Law RAG (bge-m3)")
    print("=" * 60)

    # 1. Load Chunks
    print(f"\n📂 Loading chunks from: {JSON_FILE}")
    try:
        docs = load_chunks(JSON_FILE)
        print(f"   ✅ Successfully loaded {len(docs)} chunks.")
    except FileNotFoundError:
        print(f"   ❌ File not found: {JSON_FILE}")
        return
    except Exception as e:
        print(f"   ❌ Error reading JSON: {e}")
        return

    if not docs:
        print("   ⚠️ No valid chunks to embed.")
        return

    # 2. Initialize Embeddings & ChromaDB
    print(f"\n🔢 Initializing Ollama Embeddings (Model: {EMBED_MODEL})...")
    print("   💡 Make sure Ollama is running: `ollama run bge-m3`")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if args.reset:
        print("\n🗑️  Resetting ChromaDB collection...")
        import shutil
        import os
        db_path = os.path.join(VECTOR_DB, "chroma.sqlite3")
        if os.path.exists(VECTOR_DB):
            shutil.rmtree(VECTOR_DB)
        print("   ✅ Old database deleted.")

    print(f"\n💾 Saving to ChromaDB: {VECTOR_DB} (Collection: {COLLECTION_NAME})")
    
    # 3. Batch Embedding & Saving
    vectorstore = None
    total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Embedding Progress", unit="batch"):
        batch_docs = docs[i : i + BATCH_SIZE]
        
        if vectorstore is None:
            # Create a new collection for the first batch
            vectorstore = Chroma.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                persist_directory=VECTOR_DB,
                collection_name=COLLECTION_NAME
            )
        else:
            # Add to the existing collection
            vectorstore.add_documents(batch_docs)

    print("\n" + "=" * 60)
    print("✅ DONE! All chunks have been successfully embedded and saved.")
    print(f"📁 Database Location: {VECTOR_DB}")
    print("=" * 60)


if __name__ == "__main__":
    main()