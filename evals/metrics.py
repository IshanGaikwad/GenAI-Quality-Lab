"""Deterministic, reference-free evaluation metrics for RAG outputs.

These are transparent lexical implementations of the same metric families
used by LLM-evaluation platforms (Arize Phoenix, Langfuse, DeepEval, Ragas):

- groundedness: is each claim in the answer supported by the retrieved
  context?
- hallucination detection: which claims are NOT supported?
- answer relevance: does the answer address the question?
- retrieval quality: did the retriever surface the documents a human would
  consider necessary?

Deterministic lexical metrics are deliberately chosen for the CI gate:
they are free, fast, and reproducible. LLM-as-judge metrics (see the
optional DeepEval integration in the README) add semantic depth but cost
money and introduce nondeterminism — in a real pipeline they run in a
separate, non-blocking stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.knowledge_base import content_tokens


def split_claims(answer: str) -> list[str]:
    """Split an answer into sentence-level claims."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]


def support_ratio(claim: str, context: str) -> float:
    """Fraction of a claim's content tokens that appear in the context."""
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return 1.0
    context_tokens = content_tokens(context)
    return len(claim_tokens & context_tokens) / len(claim_tokens)


def groundedness(answer: str, context: str) -> float:
    """Mean support ratio across all claims in the answer. Range [0, 1]."""
    claims = split_claims(answer)
    if not claims:
        return 0.0
    return sum(support_ratio(c, context) for c in claims) / len(claims)


@dataclass(frozen=True)
class HallucinationReport:
    flagged_claims: list[str]
    threshold: float

    @property
    def is_hallucinated(self) -> bool:
        return bool(self.flagged_claims)


def detect_hallucinations(
    answer: str, context: str, threshold: float = 0.6
) -> HallucinationReport:
    """Flag claims whose support ratio falls below ``threshold``."""
    flagged = [
        c for c in split_claims(answer) if support_ratio(c, context) < threshold
    ]
    return HallucinationReport(flagged_claims=flagged, threshold=threshold)


def answer_relevance(question: str, answer: str) -> float:
    """Overlap between question content tokens and answer content tokens."""
    q = content_tokens(question)
    if not q:
        return 0.0
    return len(q & content_tokens(answer)) / len(q)


@dataclass(frozen=True)
class RetrievalReport:
    precision: float
    recall: float


def retrieval_quality(
    retrieved_ids: list[str], expected_ids: list[str]
) -> RetrievalReport:
    retrieved, expected = set(retrieved_ids), set(expected_ids)
    if not retrieved:
        return RetrievalReport(precision=0.0, recall=0.0 if expected else 1.0)
    hit = retrieved & expected
    return RetrievalReport(
        precision=len(hit) / len(retrieved),
        recall=len(hit) / len(expected) if expected else 1.0,
    )
