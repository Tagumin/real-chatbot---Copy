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

DEBUG = True


# ─────────────────────────────
# LLM
# ─────────────────────────────
llm = OllamaLLM(model=LLM_MODEL, temperature=TEMPERATURE)

PROMPT = """
You are an Indonesian Legal Assistant.

Rules:
1. Answer legal questions ONLY using your knowledge base. Never invent, assume, or add legal information that is not explicitly available.
2. Never mention terms such as "context", "dataset", "source", or "provided information". Speak as if the knowledge is your own.
3. If the required information is unavailable, say:
   "I do not have sufficient information regarding this matter to provide an accurate answer."
4. For non-legal questions (greetings, thanks, casual chat), respond briefly and politely.
5. Format responses in Markdown:
   - Start with a direct answer.
   - Use **bold** for important terms.
   - Use bullet points or numbered lists when helpful.
   - Use headings for long answers.
   - Keep paragraphs short.

KNOWLEDGE:
{context}

QUESTION:
{question}

ANSWER:
"""
prompt = ChatPromptTemplate.from_template(PROMPT)
chain = prompt | llm | StrOutputParser()


# ─────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────
def build_context(docs):
    parts = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown Source")
        article = doc.metadata.get("article", "")

        if not article or article == "-":
            match = re.search(r'Article\s+(\d+[A-Za-z]*)', doc.page_content)
            article = f"Article {match.group(1)}" if match else "Unknown Article"

        content = doc.page_content.strip()
        parts.append(f"Source: {source}\n{article}\n{content}")

    return "\n\n---\n\n".join(parts)[:MAX_CONTEXT_CHARS]


# ─────────────────────────────
# CLEAN OUTPUT
# ─────────────────────────────
def clean(text: str):
    if not text:
        return ""

    # Strip any stray <tool_call>...</tool_call> blocks the model might emit.
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
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


def debug_print_docs(docs, label: str):
    """Shared debug printer for a list of retrieved docs (used for both
    LAW and CULTURE results) -- replaces the two near-identical loops
    that used to exist in ask_question()."""
    if not DEBUG or not docs:
        return

    for i, d in enumerate(docs):
        print("\n" + "=" * 80)
        print(f"DOC {i + 1} ({label})")
        print("=" * 80)
        print(f"Domain       : {d.metadata.get('domain', 'N/A')}")
        print(f"Rerank Rank  : {d.metadata.get('rerank_rank', 'N/A')}")
        print(f"Rerank Score : {d.metadata.get('rerank_score', 'N/A')}")
        print(f"BM25 Score   : {d.metadata.get('bm25_score', 'N/A')}")
        print("\nContent:")
        print(d.page_content[:300])
        print("=" * 80)


def retrieve_docs(retriever, question: str):
    """Works whether the retriever exposes .invoke() (current LangChain API)
    or the older .get_relevant_documents()."""
    if hasattr(retriever, "invoke"):
        return retriever.invoke(question)
    return retriever.get_relevant_documents(question)


# ─────────────────────────────
# MAIN RAG FUNCTION
# ─────────────────────────────
def ask_question(question: str):
    start = datetime.now()

    try:
        debug_print("QUESTION", question)

        # ── RETRIEVAL (both domains, always) ──
        docs_law = retrieve_docs(retriever_law, question)
        docs_culture = retrieve_docs(retriever_culture, question)

        debug_print("DOCS FOUND (LAW)", len(docs_law))
        debug_print("DOCS FOUND (CULTURE)", len(docs_culture))

        debug_print_docs(docs_law, "LAW")
        debug_print_docs(docs_culture, "CULTURE")

        # ── CONTEXT ──
        context_law = build_context(docs_law)
        context_culture = build_context(docs_culture)
        debug_print("CONTEXT PREVIEW (LAW)", context_law[:500])
        debug_print("CONTEXT PREVIEW (CULTURE)", context_culture[:500])

        # ── LLM ──
        if docs_law:
            raw_law = chain.invoke({"context": context_law, "question": question})
        else:
            raw_law = "I do not have the specific regulations regarding this available at this time."

        if docs_culture:
            raw_culture = chain.invoke({"context": context_culture, "question": question})
        else:
            raw_culture = "I do not have specific cultural information regarding this available at this time."

        debug_print("RAW LLM OUTPUT (LAW)", raw_law)
        debug_print("RAW LLM OUTPUT (CULTURE)", raw_culture)

        # ── PROCESSING ──
        answer_law = clean(raw_law)
        answer_culture = clean(raw_culture)

        debug_print("CLEANED ANSWER (LAW)", answer_law)
        debug_print("CLEANED ANSWER (CULTURE)", answer_culture)

        # ── RESULT ──
        return {
            "answer_law": answer_law,
            "answer_culture": answer_culture,
            "docs_law": len(docs_law),
            "docs_culture": len(docs_culture),
            "time": (datetime.now() - start).total_seconds()
        }

    except Exception as e:
        print("\n❌ ERROR OCCURRED")
        traceback.print_exc()

        return {
            "error": str(e),
            "answer_law": "System error occurred",
            "answer_culture": "System error occurred",
            "time": (datetime.now() - start).total_seconds()
        }


# # ─────────────────────────────
# # CLI TEST MODE
# # ─────────────────────────────
# if __name__ == "__main__":
#     print("\n🚀 RAG DEBUG MODE")
#     print("Type 'exit' to quit\n")

#     while True:
#         q = input("Ask > ")

#         if q.lower() == "exit":
#             break

#         result = ask_question(q)

#         print("\n🧠 FINAL ANSWER")
#         print("=" * 60)
#         print("LAW PERSPECTIVE:")
#         print(result.get("answer_law", result.get("error")))
#         print("\nCULTURE PERSPECTIVE:")
#         print(result.get("answer_culture", ""))
#         print("=" * 60)