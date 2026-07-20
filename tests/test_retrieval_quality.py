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
from evals.metrics import retrieval_quality

GOLDEN = json.loads(
    (Path(__file__).parent.parent / "evals" / "datasets" / "golden_set.json").read_text()
)
IN_SCOPE = [c for c in GOLDEN["cases"] if not c["out_of_scope"]]
OUT_OF_SCOPE = [c for c in GOLDEN["cases"] if c["out_of_scope"]]


@pytest.mark.parametrize("case", IN_SCOPE, ids=lambda c: c["id"])
def test_expected_documents_are_retrieved(case):
    retrieved_ids = [d.doc_id for d in retrieve(case["question"])]
    report = retrieval_quality(retrieved_ids, case["expected_doc_ids"])
    assert report.recall == 1.0, (
        f"Retriever missed required docs for {case['id']!r}: "
        f"wanted {case['expected_doc_ids']}, got {retrieved_ids}"
    )


@pytest.mark.parametrize("case", OUT_OF_SCOPE, ids=lambda c: c["id"])
def test_out_of_scope_questions_retrieve_nothing(case):
    retrieved = retrieve(case["question"])
    assert retrieved == [], (
        f"Out-of-scope question {case['id']!r} unexpectedly retrieved "
        f"{[d.doc_id for d in retrieved]} — knowledge base or stopword drift?"
    )
