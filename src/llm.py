"""
llm.py
------
Builds the final prompt from retrieved chunks and calls Gemini to
generate a grounded answer.
"""

from typing import List, Dict
import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_LLM_MODEL

genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below. If the answer is not present in the context, "
    "say you don't have enough information — do not make anything up. "
    "Keep answers clear and cite which source(s) you used."
)


def build_prompt(query: str, contexts: List[Dict]) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['source']} | chunk {c['chunk_id']}]\n{c['text']}"
        for c in contexts
    )
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"### Context\n{context_block}\n\n"
        f"### Question\n{query}\n\n"
        f"### Answer"
    )


def generate_answer(query: str, contexts: List[Dict]) -> str:
    if not contexts:
        return "I couldn't find anything relevant in the indexed documents to answer that."

    prompt = build_prompt(query, contexts)
    model = genai.GenerativeModel(GEMINI_LLM_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip() if response.text else "No answer generated."
