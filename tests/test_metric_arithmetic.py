"""Exact-value tests that pin the *arithmetic* of every metric.

Coverage and threshold assertions (`>= 0.8`) leave the formulas under-verified:
an operator swap that nudges a score without crossing a threshold — or that
happens to be a no-op when precision == recall — survives. Mutation testing
surfaced exactly those survivors. These tests assert exact fractions on
deliberately *asymmetric* inputs (p != r, non-integer ratios) so that changing
any single operator changes the result.
"""

import pytest

from evals.metrics import (
    answer_f1,
    answer_relevance,
    detect_hallucinations,
    groundedness,
    reciprocal_rank,
    retrieval_quality,
    support_ratio,
)


def test_support_ratio_is_an_exact_fraction():
    # 2 of the claim's 3 content tokens appear in the context -> exactly 2/3.
    assert support_ratio("vacation dental salary", "vacation dental") == pytest.approx(2 / 3)


def test_groundedness_is_the_exact_mean_of_claim_support():
    context = "plan covers vacation dental salary"
    # claim 1 fully supported (1.0); claim 2 supported 2/3 -> mean = 5/6.
    answer = "vacation dental salary. vacation dental gamma."
    assert groundedness(answer, context) == pytest.approx(5 / 6)


def test_answer_relevance_is_an_exact_fraction():
    # 2 of the question's 3 content tokens appear in the answer -> exactly 2/3.
    assert answer_relevance("vacation dental salary", "vacation dental only") == pytest.approx(2 / 3)


def test_answer_f1_is_exact_when_precision_differs_from_recall():
    # prediction has 4 tokens, reference 2, sharing 2:
    #   precision = 2/4 = 0.5, recall = 2/2 = 1.0, F1 = 2*0.5*1.0 / 1.5 = 2/3.
    # Asymmetric on purpose: an operator swap in 2*p*r/(p+r) can be a no-op when
    # p == r, but not here.
    assert answer_f1("vacation dental salary bonus", "vacation dental") == pytest.approx(2 / 3)


def test_hallucination_threshold_is_strictly_less_than():
    # A claim supported at exactly the threshold (3/5 = 0.6) must NOT be flagged
    # (`support < 0.6` is False), which pins `<` against `<=`.
    claim = "vacation dental salary bonus gamma"  # 5 content tokens
    context = "vacation dental salary"  # 3 of them supported -> 0.6
    assert detect_hallucinations(claim, context, threshold=0.6).is_hallucinated is False
    # nudge the threshold just above the support and it flips to flagged.
    assert detect_hallucinations(claim, context, threshold=0.61).is_hallucinated is True


def test_retrieval_quality_precision_and_recall_are_exact_fractions():
    # retrieved {a,b,c}, expected {a,z}, hit {a}: precision 1/3, recall 1/2.
    report = retrieval_quality(["a", "b", "c"], ["a", "z"])
    assert report.precision == pytest.approx(1 / 3)
    assert report.recall == pytest.approx(1 / 2)


def test_reciprocal_rank_is_exact_for_a_lower_ranked_hit():
    # relevant doc sits third -> exactly 1/3 (pins the 1-based rank arithmetic).
    assert reciprocal_rank(["x", "y", "z"], ["z"]) == pytest.approx(1 / 3)
