"""A tiny in-memory knowledge base with a keyword-overlap retriever.

This stands in for the vector store / enterprise search layer of a real RAG
system (e.g. Amazon Kendra, OpenSearch, pgvector). The retrieval contract is
identical — query in, ranked documents out — which is what the evaluation
suite exercises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DOCUMENTS: dict[str, str] = {
    "pto-001": (
        "Full-time employees accrue 20 days of paid time off (PTO) per year. "
        "PTO accrues at a rate of 1.67 days per month. Unused PTO up to 5 days "
        "may be carried over into the next calendar year. Carry-over days must "
        "be used by March 31."
    ),
    "pto-002": (
        "PTO requests must be submitted in the HR portal at least 10 business "
        "days in advance for absences of 3 days or longer. Managers approve or "
        "decline requests within 3 business days."
    ),
    "med-001": (
        "The company offers three medical plans: Basic, Plus, and Premium. "
        "The Premium plan includes dental and vision coverage. Open enrollment "
        "runs from November 1 to November 15 each year."
    ),
    "med-002": (
        "New hires have 30 days from their start date to enroll in a medical "
        "plan. Coverage begins on the first day of the month following "
        "enrollment."
    ),
    "ret-001": (
        "The company matches 401(k) contributions dollar-for-dollar up to 4% "
        "of base salary. Employer matching contributions vest immediately. "
        "Employees may change their contribution rate at any time."
    ),
    "wfh-001": (
        "Employees may work remotely up to 3 days per week with manager "
        "approval. Fully remote arrangements require director-level approval "
        "and an annual review."
    ),
}

_WORD = re.compile(r"[a-z0-9']+")

_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it "
    "may me my of on or our the their they this to up we what when where which "
    "who will with you your".split()
)


def content_tokens(text: str) -> set[str]:
    """Lower-cased content-word tokens (stopwords removed)."""
    return {t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS}


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    text: str
    score: float


def retrieve(query: str, top_k: int = 2) -> list[RetrievedDoc]:
    """Rank documents by content-token overlap with the query."""
    q_tokens = content_tokens(query)
    if not q_tokens:
        return []
    scored = []
    for doc_id, text in DOCUMENTS.items():
        d_tokens = content_tokens(text)
        overlap = len(q_tokens & d_tokens)
        if overlap:
            scored.append(RetrievedDoc(doc_id, text, overlap / len(q_tokens)))
    scored.sort(key=lambda d: (-d.score, d.doc_id))
    return scored[:top_k]
