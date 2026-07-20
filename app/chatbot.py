"""A minimal RAG chatbot with a deterministic mock LLM.

Design goals:
- Runs fully offline by default (MockLLM) so the evaluation suite is free,
  fast, and deterministic — the properties you want in CI.
- The same ``RagChatbot`` accepts any object with a ``generate(prompt)``
  method, so a real model client (OpenAI, Bedrock, etc.) can be swapped in
  behind an environment flag without touching the tests.
- ``MockLLM(hallucinate=True)`` deliberately fabricates a claim that is NOT
  in the retrieved context. The test suite uses this to prove the
  hallucination detector actually catches hallucinations — a detector that
  has never seen a failure is untested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .knowledge_base import RetrievedDoc, content_tokens, retrieve

PROMPT_TEMPLATE = """You are an internal employee-benefits assistant.

Rules:
- Answer ONLY from the context below.
- If the context does not contain the answer, reply exactly:
  "I don't have that information in the benefits documentation."
- Do not speculate. Do not invent policies, numbers, or dates.

Context:
{context}

Question: {question}
Answer:"""

FALLBACK = "I don't have that information in the benefits documentation."

_FABRICATED_CLAIM = (
    " Additionally, employees receive a complimentary gym membership worth "
    "$1,200 per year."
)


@dataclass
class ChatResponse:
    question: str
    answer: str
    prompt: str
    retrieved: list[RetrievedDoc] = field(default_factory=list)

    @property
    def context_text(self) -> str:
        return "\n".join(d.text for d in self.retrieved)


class MockLLM:
    """Deterministic stand-in for a hosted LLM.

    Extracts the context sentences most relevant to the question — i.e. it
    behaves like a well-grounded model. With ``hallucinate=True`` it appends
    a fabricated, unsupported claim.
    """

    def __init__(self, hallucinate: bool = False):
        self.hallucinate = hallucinate

    def generate(self, prompt: str) -> str:
        context, question = _parse_prompt(prompt)
        if not context.strip():
            return FALLBACK
        q_tokens = content_tokens(question)
        sentences = _split_sentences(context)
        scored = [
            (len(q_tokens & content_tokens(s)), i, s)
            for i, s in enumerate(sentences)
        ]
        best = [s for score, _, s in sorted(scored, key=lambda t: (-t[0], t[1])) if score > 0][:2]
        if not best:
            return FALLBACK
        answer = " ".join(best)
        if self.hallucinate:
            answer += _FABRICATED_CLAIM
        return answer


class RagChatbot:
    def __init__(self, llm=None, top_k: int = 2):
        self.llm = llm or MockLLM()
        self.top_k = top_k

    def ask(self, question: str) -> ChatResponse:
        retrieved = retrieve(question, top_k=self.top_k)
        context = "\n".join(d.text for d in retrieved)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self.llm.generate(prompt)
        return ChatResponse(
            question=question, answer=answer, prompt=prompt, retrieved=retrieved
        )


def _parse_prompt(prompt: str) -> tuple[str, str]:
    context_match = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.S)
    question_match = re.search(r"Question: (.*?)\nAnswer:", prompt, re.S)
    context = context_match.group(1) if context_match else ""
    question = question_match.group(1) if question_match else ""
    return context, question


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
