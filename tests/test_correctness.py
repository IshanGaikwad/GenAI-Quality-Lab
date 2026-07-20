"""Answer-correctness evaluation via reference-answer token F1.

Groundedness asks "is the answer supported by the context?" and `must_mention`
asks "does the answer contain this fact?" — neither asks "does the answer match
a known-good answer?" A verbose answer that pads the correct fact with extra
sentences passes both yet is a worse answer. Token F1 against a curated
reference answer captures that: it rewards covering the reference (recall) while
penalizing unwarranted extra tokens (precision).
"""

import json
from pathlib import Path

import pytest

from app.chatbot import RagChatbot
from evals.metrics import answer_f1

GOLDEN = json.loads(
    (Path(__file__).parent.parent / "evals" / "datasets" / "golden_set.json").read_text()
)
WITH_REFERENCE = [c for c in GOLDEN["cases"] if c.get("reference_answer")]

# A produced answer must share at least half its tokens (harmonic mean) with the
# reference. The current extractive system often appends a second, on-corpus but
# off-question sentence, which drags several answers below a perfect 1.0 — F1 is
# what makes that over-answering visible, and this floor is the regression trip.
F1_THRESHOLD = 0.5

bot = RagChatbot()


# --- metric unit tests (also give the new metric full branch coverage) ---

def test_identical_answer_scores_one():
    assert answer_f1("The Premium plan includes dental.", "The Premium plan includes dental.") == 1.0


def test_normalization_ignores_articles_and_punctuation():
    # "the" (article) and the period are stripped; the rest matches exactly.
    assert answer_f1("The 20 days.", "20 days") == 1.0


def test_disjoint_answers_score_zero():
    assert answer_f1("remote work policy", "dental vision coverage") == 0.0


def test_partial_overlap_is_between_zero_and_one():
    score = answer_f1("20 days of paid time off", "20 vacation days")
    assert 0.0 < score < 1.0


def test_empty_prediction_scores_zero():
    assert answer_f1("", "20 days") == 0.0


def test_two_empty_strings_score_one():
    # Degenerate but well-defined: nothing to predict, nothing missed.
    assert answer_f1("", "") == 1.0


# --- correctness gate over the golden set ---

@pytest.mark.parametrize("case", WITH_REFERENCE, ids=lambda c: c["id"])
def test_answers_match_reference(case):
    response = bot.ask(case["question"])
    score = answer_f1(response.answer, case["reference_answer"])
    assert score >= F1_THRESHOLD, (
        f"{case['id']}: answer F1 {score:.2f} < {F1_THRESHOLD} vs reference.\n"
        f"reference: {case['reference_answer']}\n"
        f"answer:    {response.answer}"
    )
