"""Semantic evaluation metrics — the non-blocking "stage 2" the README promises.

The lexical metrics in ``evals/metrics.py`` gate CI because they are free, fast,
and deterministic. But they measure word *overlap*, not *meaning*, and that has a
correctness ceiling:

- a negated claim ("PTO CANNOT be carried over") shares almost every token with
  its source, so lexical groundedness rates it ~0.89 and does NOT flag it, and
- a faithful paraphrase shares few tokens, so lexical groundedness wrongly flags
  it as a hallucination.

This module closes that gap with real models:

- a **bi-encoder** (sentence embeddings) for semantic *relevance* and answer
  similarity — credits topically-correct answers that share no keywords, and
- an **NLI cross-encoder** for entailment-based *groundedness* — classifies each
  answer claim against the context as entailment / neutral / contradiction, so
  it actually catches contradictions.

It is deliberately NOT part of the CI quality gate: it needs torch and
downloaded models (slow, heavier, version-sensitive). It runs as a separate
reporting stage — see ``semantic_eval/run.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from evals.metrics import split_claims  # reuse sentence-level claim splitting

BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-distilroberta-base"
# Output-head order for the NLI cross-encoder above.
NLI_LABELS = ("contradiction", "entailment", "neutral")


@lru_cache(maxsize=1)
def _bi_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(BI_ENCODER_MODEL)


@lru_cache(maxsize=1)
def _nli():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(NLI_MODEL)


def semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity of two texts' sentence embeddings, in [-1, 1]."""
    from sentence_transformers.util import cos_sim

    model = _bi_encoder()
    return float(cos_sim(model.encode(a), model.encode(b)))


def semantic_relevance(question: str, answer: str) -> float:
    """Embedding cosine between question and answer. Unlike lexical relevance,
    this credits an answer that is on-topic without sharing keywords (e.g. a PTO
    answer to a 'vacation' question)."""
    return semantic_similarity(question, answer)


def semantic_answer_similarity(answer: str, reference: str) -> float:
    """Embedding cosine between a produced answer and a reference answer — a
    paraphrase-tolerant complement to token F1."""
    return semantic_similarity(answer, reference)


@dataclass(frozen=True)
class ClaimEntailment:
    claim: str
    label: str
    probabilities: dict[str, float]


def entailment_report(context: str, answer: str) -> list[ClaimEntailment]:
    """Classify each claim (sentence) in ``answer`` against ``context`` as
    entailment / neutral / contradiction using the NLI model."""
    claims = split_claims(answer)
    if not claims or not context.strip():
        return []

    import numpy as np

    logits = np.atleast_2d(_nli().predict([(context, claim) for claim in claims]))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    return [
        ClaimEntailment(
            claim=claim,
            label=NLI_LABELS[int(row.argmax())],
            probabilities={label: float(p) for label, p in zip(NLI_LABELS, row)},
        )
        for claim, row in zip(claims, probs)
    ]


def nli_groundedness(context: str, answer: str) -> float:
    """Fraction of answer claims the context *entails*. Contradictions and
    neutrals are not counted as grounded. Range [0, 1]."""
    report = entailment_report(context, answer)
    if not report:
        return 0.0
    entailed = sum(1 for c in report if c.label == "entailment")
    return entailed / len(report)


def contradicted_claims(context: str, answer: str) -> list[str]:
    """Claims the context actively contradicts — the semantic hallucination
    signal that lexical overlap misses on negation."""
    return [c.claim for c in entailment_report(context, answer) if c.label == "contradiction"]
