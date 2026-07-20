"""Generation-layer evaluation: every answer must be grounded in the
retrieved context, and must contain the facts the golden set requires.
"""

import json
from pathlib import Path

import pytest

from app.chatbot import FALLBACK, RagChatbot
from evals.metrics import answer_relevance, groundedness

GOLDEN = json.loads(
    (Path(__file__).parent.parent / "evals" / "datasets" / "golden_set.json").read_text()
)
IN_SCOPE = [c for c in GOLDEN["cases"] if not c["out_of_scope"]]
OUT_OF_SCOPE = [c for c in GOLDEN["cases"] if c["out_of_scope"]]

GROUNDEDNESS_THRESHOLD = 0.8
RELEVANCE_THRESHOLD = 0.5

bot = RagChatbot()


@pytest.mark.parametrize("case", IN_SCOPE, ids=lambda c: c["id"])
def test_answers_are_grounded_in_context(case):
    response = bot.ask(case["question"])
    score = groundedness(response.answer, response.context_text)
    assert score >= GROUNDEDNESS_THRESHOLD, (
        f"{case['id']}: groundedness {score:.2f} < {GROUNDEDNESS_THRESHOLD} "
        f"— answer contains claims not supported by retrieved context.\n"
        f"Answer: {response.answer}"
    )


@pytest.mark.parametrize("case", IN_SCOPE, ids=lambda c: c["id"])
def test_answers_contain_required_facts(case):
    response = bot.ask(case["question"])
    for fact in case["must_mention"]:
        assert fact.lower() in response.answer.lower(), (
            f"{case['id']}: answer is missing required fact {fact!r}.\n"
            f"Answer: {response.answer}"
        )


@pytest.mark.parametrize("case", IN_SCOPE, ids=lambda c: c["id"])
def test_answers_are_relevant_to_question(case):
    response = bot.ask(case["question"])
    score = answer_relevance(case["question"], response.answer)
    assert score >= RELEVANCE_THRESHOLD, (
        f"{case['id']}: relevance {score:.2f} < {RELEVANCE_THRESHOLD}.\n"
        f"Answer: {response.answer}"
    )


@pytest.mark.parametrize("case", OUT_OF_SCOPE, ids=lambda c: c["id"])
def test_out_of_scope_questions_get_the_exact_fallback(case):
    """The assistant must refuse rather than improvise. The fallback string
    is part of the product contract, so it is asserted exactly."""
    response = bot.ask(case["question"])
    assert response.answer == FALLBACK, (
        f"{case['id']}: expected exact fallback, got: {response.answer}"
    )
