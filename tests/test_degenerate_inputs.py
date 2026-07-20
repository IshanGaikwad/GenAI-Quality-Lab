"""Behavior on degenerate inputs: empty answers, stopword-only text, and
empty retrieval.

Evaluation metrics earn their trust at the boundaries. Empty strings,
punctuation-only claims, and all-stopword questions are exactly where a
lexical metric can silently divide by zero or return a misleading score, so
each guard branch has an explicit, documented contract rather than an
accidental one. These tests pin those contracts.
"""

from app.chatbot import FALLBACK, PROMPT_TEMPLATE, MockLLM
from app.knowledge_base import retrieve
from evals.metrics import (
    answer_relevance,
    groundedness,
    retrieval_quality,
    support_ratio,
)


def test_support_ratio_of_contentless_claim_is_vacuously_supported():
    # A claim with no content tokens (all stopwords) makes no factual
    # assertion, so there is nothing to contradict the context: treat it as
    # fully supported rather than dividing by zero.
    assert support_ratio("the a of to", "Full-time employees accrue 20 days of PTO.") == 1.0


def test_groundedness_of_empty_answer_is_zero():
    # No claims means nothing was grounded — score the floor, not a vacuous 1.0
    # that would let an empty answer sail through the gate.
    context = "Full-time employees accrue 20 days of PTO per year."
    assert groundedness("", context) == 0.0
    assert groundedness("   ", context) == 0.0


def test_answer_relevance_of_contentless_question_is_zero():
    # A question with no content tokens gives nothing to measure relevance
    # against; report irrelevance rather than dividing by zero.
    assert answer_relevance("what is the of a", "Employees accrue 20 days of PTO.") == 0.0


def test_retrieval_quality_of_empty_retrieval():
    # Retrieving nothing when documents were expected is zero precision AND
    # zero recall — the worst case, and the one most likely to hide behind a
    # crash if the guard is missing.
    missed = retrieval_quality([], ["pto-001"])
    assert missed.precision == 0.0
    assert missed.recall == 0.0

    # Retrieving nothing when nothing was expected is vacuously perfect recall.
    vacuous = retrieval_quality([], [])
    assert vacuous.precision == 0.0
    assert vacuous.recall == 1.0


def test_mockllm_falls_back_when_context_present_but_irrelevant():
    # Distinct from the empty-context fallback: here the context is non-empty
    # but no sentence overlaps the question, so the model must refuse instead
    # of returning an unrelated sentence.
    prompt = PROMPT_TEMPLATE.format(
        context="Full-time employees accrue 20 days of PTO per year.",
        question="xylophone quokka telescope",
    )
    assert MockLLM().generate(prompt) == FALLBACK


def test_retrieve_returns_nothing_for_contentless_query():
    # A blank or stopword-only query has no signal to rank documents by; return
    # nothing rather than an arbitrary ranking.
    assert retrieve("") == []
    assert retrieve("what is the of a to") == []
