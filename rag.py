"""
rag_debug.py
============
RAG Engine + Debug Mode for Terminal Testing

Features:
✔ Full step-by-step logging
✔ Retrieval inspection
✔ Context preview
✔ LLM raw output trace
✔ Safe error handling
"""

import re
import traceback
from datetime import datetime

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from vector import retriever_law, retriever_culture


# ─────────────────────────────
# CONFIG
# ─────────────────────────────
LLM_MODEL = "qwen2.5:3b"
TEMPERATURE = 0.1
MAX_CONTEXT_CHARS = 8000

MODE_LAW = "1"
MODE_CULTURE = "2"

DEBUG = True


# ─────────────────────────────
# LLM
# ─────────────────────────────
llm = OllamaLLM(model=LLM_MODEL, temperature=TEMPERATURE)

PROMPT = """
You are an Indonesian legal assistant.

Use ONLY the provided context to answer.
If no relevant information is found, respond:
"No relevant legal provision found."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

prompt = ChatPromptTemplate.from_template(PROMPT)
chain = prompt | llm | StrOutputParser()


# ─────────────────────────────
# RETRIEVER
# ─────────────────────────────
def get_retriever(mode: str):
    if mode == MODE_LAW:
        return retriever_law, "LAW"
    elif mode == MODE_CULTURE:
        return retriever_culture, "CULTURE"
    return None, None


# ─────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────
def build_context(docs):
    parts = []

    for doc in docs:
        article = doc.metadata.get("article", "")

        if not article or article == "-":
            match = re.search(r'Article\s+(\d+[A-Za-z]*)', doc.page_content)
            article = f"Article {match.group(1)}" if match else "Unknown"

        content = doc.page_content.strip()
        parts.append(f"{article}\n{content}")

    return "\n\n---\n\n".join(parts)[:MAX_CONTEXT_CHARS]


# ─────────────────────────────
# CLEAN OUTPUT
# ─────────────────────────────
def clean(text: str):
    if not text:
        return ""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


# ─────────────────────────────
# DEBUG PRINT HELPERS
# ─────────────────────────────
def debug_print(title, content):
    if not DEBUG:
        return
    print(f"\n🟡 {title}")
    print("-" * 50)
    print(content)
    print("-" * 50)


# ─────────────────────────────
# MAIN RAG FUNCTION
# ─────────────────────────────
def ask_question(question: str, mode: str = MODE_LAW):
    start = datetime.now()

    try:
        debug_print("QUESTION", question)
        debug_print("MODE", mode)

        retriever, domain = get_retriever(mode)

        if retriever is None:
            return {"error": "Invalid mode"}

        # ── RETRIEVAL ──
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(question)
        else:
            docs = retriever.get_relevant_documents(question)

        debug_print("DOCS FOUND", len(docs))

        if DEBUG and docs:
            for i, d in enumerate(docs):

                rerank_score = d.metadata.get("rerank_score", "N/A")
                rerank_rank = d.metadata.get("rerank_rank", "N/A")
                bm25_score = d.metadata.get("bm25_score", "N/A")
                domain = d.metadata.get("domain", "N/A")

                print("\n" + "=" * 80)
                print(f"DOC {i+1}")
                print("=" * 80)

                print(f"Domain       : {domain}")
                print(f"Rerank Rank  : {rerank_rank}")
                print(f"Rerank Score : {rerank_score}")
                print(f"BM25 Score   : {bm25_score}")

                print("\nContent:")
                print(d.page_content[:300])

                print("=" * 80)
        if not docs:
            return {
                "answer": "No relevant documents found.",
                "domain": domain,
                "docs": 0,
                "time": (datetime.now() - start).total_seconds()
            }

        # ── CONTEXT ──
        context = build_context(docs)
        debug_print("CONTEXT PREVIEW", context[:500])

        # ── LLM ──
        raw = chain.invoke({
            "context": context,
            "question": question
        })

        debug_print("RAW LLM OUTPUT", raw)

        # ── PROCESSING ──
        answer = clean(raw)

        debug_print("CLEANED ANSWER", answer)

        # ── RESULT ──
        return {
            "answer": answer,
            "domain": domain,
            "docs": len(docs),
            "time": (datetime.now() - start).total_seconds()
        }

    except Exception as e:
        print("\n❌ ERROR OCCURRED")
        traceback.print_exc()

        return {
            "error": str(e),
            "answer": "System error occurred",
            "time": (datetime.now() - start).total_seconds()
        }


# ─────────────────────────────
# CLI TEST MODE
# ─────────────────────────────
if __name__ == "__main__":
    print("\n🚀 RAG DEBUG MODE")
    print("Type 'exit' to quit\n")

    while True:
        q = input("Ask > ")

        if q.lower() == "exit":
            break

        mode = input("Mode (1=law, 2=culture) > ")

        result = ask_question(q, mode)

        print("\n🧠 FINAL ANSWER")
        print("=" * 60)
        print(result)
        print("=" * 60)