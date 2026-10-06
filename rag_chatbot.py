"""
Retrieval-grounded diabetes education chatbot.

Design (per reviewer feedback on the "LLM-Style" naming):
- Every answer is grounded in a small local knowledge base of public health
  sources (WHO / IDF / ADA fact sheets), and every answer names its source.
- If a local Ollama server is running (free, offline, no API key), the
  retrieved passages are handed to it as context and it drafts the reply
  (retrieval-augmented generation). If Ollama is not running, the same
  retrieved passages are assembled into an answer by a template — so the
  chatbot degrades gracefully instead of failing.
- A hard safety filter (medication/dosage questions) runs BEFORE retrieval
  or generation and always wins: this system never names drugs or doses.
- A `remote_llm_generate` slot is left in place for a future paid API
  (Gemini / OpenAI) — see README. It is not wired to a key by default.
"""

import json
import re
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1"

MEDICATION_PATTERN = re.compile(r"\b(medicine|tablet|insulin|dose|dosage|drug|metformin|prescri\w*)\b", re.I)


def load_knowledge_base():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve(question, kb, top_k=2):
    """Simple keyword-overlap retrieval (no heavy ML dependency required)."""
    q_words = set(re.findall(r"[a-z]+", question.lower()))
    scored = []
    for entry in kb:
        kw = set(w.lower() for w in entry["keywords"])
        text_words = set(re.findall(r"[a-z]+", entry["text"].lower()))
        score = len(q_words & kw) * 3 + len(q_words & text_words)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        # fall back to the general risk-factors entry so the bot never has
        # nothing to say
        fallback = [e for e in kb if e["id"] == "risk_factors"]
        return fallback[:1]
    return [e for _, e in scored[:top_k]]


def ollama_available():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=0.8)
        return r.status_code == 200
    except Exception:
        return False


def ollama_generate(prompt):
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception:
        pass
    return None


def remote_llm_generate(prompt, api_key=None, provider=None):
    """
    Slot for a future paid API. Left intentionally unimplemented so the
    prototype has no hidden network dependency or cost. To enable:
      - Gemini: call generativelanguage.googleapis.com with `api_key`
      - OpenAI: call api.openai.com/v1/chat/completions with `api_key`
    Wire this into `answer_question` below once a key is available.
    """
    return None


def template_answer(question, passages, prediction_context):
    body = " ".join(p["text"] for p in passages)
    sources = passages
    return body, sources


def build_prompt(question, passages, prediction_context):
    context_block = "\n".join(f"- ({p['source_name']}) {p['text']}" for p in passages)
    return (
        "You are a cautious diabetes patient-education assistant for a research prototype "
        "called DiaLLM-BD. Answer ONLY using the context below. Do not name medications, "
        "doses, or give a diagnosis. Keep the answer under 120 words and end by recommending "
        "confirmation with a qualified doctor.\n\n"
        f"Patient's current model result: {prediction_context}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\nAnswer:"
    )


def answer_question(question, prediction_context="No prediction has been made yet."):
    question = (question or "").strip()
    if not question:
        return {
            "answer": "Please type a question first.",
            "sources": [],
            "engine": "none",
        }

    if MEDICATION_PATTERN.search(question):
        return {
            "answer": (
                "Medication and dosage decisions must be made only by a qualified doctor "
                "based on your full medical history and clinical tests. This system cannot "
                "name drugs, doses, or treatment changes."
            ),
            "sources": [],
            "engine": "safety-filter",
        }

    kb = load_knowledge_base()
    passages = retrieve(question, kb, top_k=2)

    if ollama_available():
        prompt = build_prompt(question, passages, prediction_context)
        generated = ollama_generate(prompt)
        if generated:
            return {"answer": generated, "sources": passages, "engine": f"ollama:{OLLAMA_MODEL}"}

    body, sources = template_answer(question, passages, prediction_context)
    answer = (
        f"Based on current model result — {prediction_context} — here is general education: {body} "
        "This is general information, not a diagnosis; please confirm with a qualified doctor."
    )
    return {"answer": answer, "sources": sources, "engine": "retrieval-template"}
