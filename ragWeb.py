import re
import traceback
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os
from vector import retriever_law, retriever_culture


# ─────────────────────────────
# CONFIG
# ─────────────────────────────
load_dotenv()

GEMINI_MODEL      = "gemini-3.1-flash-lite"
OLLAMA_MODEL      = "qwen2.5:3b"
TEMPERATURE       = 0.1
MAX_CONTEXT_CHARS = 16000


# ─────────────────────────────
# PROMPTS
# ─────────────────────────────
TRANSLATE_PROMPT = """
Translate the following question to English.
Return ONLY the translated text, nothing else — no explanation, no quotes.

QUESTION:
{question}

TRANSLATION:
"""

PROMPT = """
You are an Indonesian Legal Assistant. Answer using the KNOWLEDGE below.

RULES:
1. Use the KNOWLEDGE as the basis for your answer. If it partially covers the topic,
   explain what it does say, even if not a perfect match to the question's exact wording.
2. Never mention "context", "dataset", "source", or similar terms.
3. Only reply EXACTLY "I do not have sufficient information regarding this matter to provide
   an accurate answer." if KNOWLEDGE is completely empty or entirely unrelated to the question.
4. For non-legal questions (greetings, thanks), respond briefly and politely.
5. Format in Markdown: direct answer first, **bold** key terms, use lists/headings when helpful.
6. No disclaimers or extra commentary — be concise and factual.
7. STRICTLY reply in the same language as the QUESTION.
   - If QUESTION is in English → reply in English.
   - If QUESTION is in Indonesian → reply in Indonesian.
   - IGNORE the language of KNOWLEDGE completely when deciding your response language.

KNOWLEDGE:
{context}

QUESTION:
{question}

ANSWER:
"""

translate_prompt_template = ChatPromptTemplate.from_template(TRANSLATE_PROMPT)
prompt_template           = ChatPromptTemplate.from_template(PROMPT)


# ─────────────────────────────
# LLM & CHAINS — Gemini, fallback ke Ollama
# ─────────────────────────────
def build_chains(llm):
    parser = StrOutputParser()
    return (
        prompt_template           | llm | parser,
        translate_prompt_template | llm | parser,
    )

try:
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=TEMPERATURE,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    answer_chain, translate_chain = build_chains(llm)
    active_llm = "gemini"
    print("Using Gemini.")
except Exception as e:
    print(f"Gemini failed ({e}). Switching to Ollama...")
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=TEMPERATURE)
    answer_chain, translate_chain = build_chains(llm)
    active_llm = "ollama"


# ─────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────
def build_context(docs) -> str:
    parts = []
    for doc in docs:
        source  = doc.metadata.get("source", "Unknown Source")
        article = doc.metadata.get("article", "")

        if not article or article == "-":
            match   = re.search(r'Article\s+(\d+[A-Za-z]*)', doc.page_content)
            article = f"Article {match.group(1)}" if match else "Unknown Article"

        parts.append(f"Source: {source}\n{article}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)[:MAX_CONTEXT_CHARS]


# ─────────────────────────────
# CLEAN OUTPUT
# ─────────────────────────────
def clean(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    return text.strip()


# ─────────────────────────────
# RETRIEVE DOCS
# ─────────────────────────────
def retrieve_docs(retriever, question: str):
    if hasattr(retriever, "invoke"):
        return retriever.invoke(question)
    return retriever.get_relevant_documents(question)


# ─────────────────────────────
# TRANSLATE QUESTION TO ENGLISH
# ─────────────────────────────
def translate_to_english(question: str) -> str:
    translated = translate_chain.invoke({"question": question})
    return clean(translated)


# ─────────────────────────────
# MAIN RAG FUNCTION
# ─────────────────────────────
def ask_question(question: str) -> dict:
    start = datetime.now()
    try:
        query_for_retrieval = translate_to_english(question)

        docs_law     = retrieve_docs(retriever_law,     query_for_retrieval)
        docs_culture = retrieve_docs(retriever_culture, query_for_retrieval)

        context_law     = build_context(docs_law)
        context_culture = build_context(docs_culture)

        raw_law = (
            answer_chain.invoke({"context": context_law,     "question": question})
            if docs_law else "No legal information is currently available."
        )
        raw_culture = (
            answer_chain.invoke({"context": context_culture, "question": question})
            if docs_culture else "No cultural information is currently available."
        )

        time_taken = (datetime.now() - start).total_seconds()
        return {
            "answer_law":     clean(raw_law),
            "answer_culture": clean(raw_culture),
            "docs_law":       len(docs_law),
            "docs_culture":   len(docs_culture),
            "llm_used":       active_llm,
            "time":           time_taken,
        }

    except Exception as e:
        traceback.print_exc()

        return {
            "error":          str(e),
            "answer_law":     "A system error occurred.",
            "answer_culture": "A system error occurred.",
            "llm_used":       active_llm,
            "time":           (datetime.now() - start).total_seconds(),
        }