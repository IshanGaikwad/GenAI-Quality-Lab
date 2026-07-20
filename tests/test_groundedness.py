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
# Everything the assistant must refuse: off-topic, hard negatives (near-domain
# but unanswerable), and adversarial (injection / false premise).
MUST_REFUSE = [c for c in GOLDEN["cases"] if c["out_of_scope"]]


def _refusal_params():
    """Parametrize refusal cases, marking documented known failures as xfail so
    the gap is tracked (and will flip to a visible pass once fixed)."""
    params = []
    for case in MUST_REFUSE:
        marks = ()
        if case.get("known_failure"):
            marks = pytest.mark.xfail(reason=case["known_failure_reason"], strict=True)
        params.append(pytest.param(case, id=case["id"], marks=marks))
    return params

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


@pytest.mark.parametrize("case", _refusal_params())
def test_unanswerable_questions_get_the_exact_fallback(case):
    """The assistant must refuse rather than improvise — for clearly off-topic
    questions, near-domain hard negatives, and adversarial prompts alike. The
    fallback string is part of the product contract, so it is asserted exactly."""
    response = bot.ask(case["question"])
    assert response.answer == FALLBACK, (
        f"{case['id']}: expected exact fallback, got: {response.answer}"
    )
