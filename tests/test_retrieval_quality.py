"""Retrieval-layer evaluation: does the retriever surface the documents a
correct answer must be grounded in?

In RAG systems most "hallucinations" are actually retrieval failures — the
model never saw the right document. Evaluating retrieval separately from
generation localizes the fault.
"""

import json
from pathlib import Path

import pytest

from app.knowledge_base import retrieve
from evals.metrics import hit_at_k, reciprocal_rank, retrieval_quality

GOLDEN = json.loads(
    (Path(__file__).parent.parent / "evals" / "datasets" / "golden_set.json").read_text()
)
IN_SCOPE = [c for c in GOLDEN["cases"] if not c["out_of_scope"]]
# Only *clearly* unrelated questions are contractually required to retrieve
# nothing. Hard negatives are near-domain by design: a keyword retriever may
# legitimately surface a related document, and the refusal contract is enforced
# at the answer level instead (see test_groundedness.py).
OFF_TOPIC = [c for c in GOLDEN["cases"] if c.get("type") == "off_topic"]


@pytest.mark.parametrize("case", IN_SCOPE, ids=lambda c: c["id"])
def test_expected_documents_are_retrieved(case):
    retrieved_ids = [d.doc_id for d in retrieve(case["question"])]
    report = retrieval_quality(retrieved_ids, case["expected_doc_ids"])
    assert report.recall == 1.0, (
        f"Retriever missed required docs for {case['id']!r}: "
        f"wanted {case['expected_doc_ids']}, got {retrieved_ids}"
    )


@pytest.mark.parametrize("case", IN_SCOPE, ids=lambda c: c["id"])
def test_expected_document_is_ranked_first(case):
    # Recall alone can't distinguish "right doc ranked first" from "ranked
    # last". These single-answer questions must put the relevant doc at the top,
    # so reciprocal rank is 1.0 — a strict tripwire on ranking regressions.
    ranked = [d.doc_id for d in retrieve(case["question"])]
    rr = reciprocal_rank(ranked, case["expected_doc_ids"])
    assert rr == 1.0, (
        f"{case['id']}: expected doc not ranked first (RR={rr:.2f}); "
        f"ranked={ranked}, expected={case['expected_doc_ids']}"
    )
    assert hit_at_k(ranked, case["expected_doc_ids"], 1) == 1.0


@pytest.mark.parametrize("case", OFF_TOPIC, ids=lambda c: c["id"])
def test_out_of_scope_questions_retrieve_nothing(case):
    retrieved = retrieve(case["question"])
    assert retrieved == [], (
        f"Out-of-scope question {case['id']!r} unexpectedly retrieved "
        f"{[d.doc_id for d in retrieved]} — knowledge base or stopword drift?"
    )


def test_terse_domain_query_survives_tokenization():
    # "401k" and the document's "401(k)" must tokenize alike, or a user typing
    # the common shorthand silently retrieves nothing. Regression guard for the
    # parenthesis-normalization fix.
    ids = [d.doc_id for d in retrieve("What is the 401k match?")]
    assert "ret-001" in ids, "shorthand '401k' failed to match '401(k)' in the corpus"


def test_incidental_word_overlap_is_below_the_relevance_floor():
    # An out-of-scope question that happens to share one generic word ("company")
    # with the corpus must fall below MIN_RELEVANCE and retrieve nothing, rather
    # than pulling unrelated docs the bot would then confabulate an answer from.
    assert retrieve("What is the company holiday party theme?") == []


def test_strong_match_does_not_drag_in_a_weakly_related_doc():
    # A focused question should return only the document that answers it — not a
    # second doc sharing only incidental words ("time", "employees") — otherwise
    # the answer gets padded with grounded-but-off-topic sentences.
    ids = [d.doc_id for d in retrieve("How much PTO do full-time employees accrue?")]
    assert ids == ["pto-001"]


# --- rank-aware metric unit tests: prove the metrics discriminate position,
# which the (perfectly-ranked) golden set alone cannot demonstrate ---

def test_reciprocal_rank_rewards_earlier_positions():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
    assert reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_no_relevant_doc_is_ranked():
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0
    assert reciprocal_rank([], ["a"]) == 0.0


def test_hit_at_k_respects_the_cutoff():
    ranked = ["a", "b", "c"]
    assert hit_at_k(ranked, ["c"], 3) == 1.0
    assert hit_at_k(ranked, ["c"], 2) == 0.0  # relevant doc sits below the cutoff
    assert hit_at_k(ranked, ["a"], 1) == 1.0


def test_hit_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError, match="positive"):
        hit_at_k(["a"], ["a"], 0)
