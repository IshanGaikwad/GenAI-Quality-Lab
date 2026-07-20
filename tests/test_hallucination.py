"""Hallucination detection — including a seeded-failure test.

A hallucination detector that has only ever seen clean answers is itself
untested. ``MockLLM(hallucinate=True)`` injects a fabricated claim so the
suite proves the detector catches real fabrications (true positive) while
not flagging grounded answers (false positive).

Note the design point demonstrated by ``test_mean_groundedness_can_mask_
hallucinations``: an answer with two grounded claims and one fabricated one
can still score ~0.79 mean groundedness. Claim-level detection exists
precisely because aggregate scores hide point failures.
"""

from app.chatbot import MockLLM, RagChatbot
from evals.metrics import detect_hallucinations, groundedness

QUESTION = "How many PTO days do full-time employees get per year?"


def test_grounded_answer_is_not_flagged():
    response = RagChatbot().ask(QUESTION)
    report = detect_hallucinations(response.answer, response.context_text)
    assert not report.is_hallucinated, (
        f"False positive: grounded answer was flagged: {report.flagged_claims}"
    )


def test_seeded_hallucination_is_flagged():
    bot = RagChatbot(llm=MockLLM(hallucinate=True))
    response = bot.ask(QUESTION)
    report = detect_hallucinations(response.answer, response.context_text)
    assert report.is_hallucinated, (
        "Detector missed a deliberately fabricated claim — the gym-membership "
        "sentence is not supported by any retrieved document."
    )
    assert any("gym" in claim.lower() for claim in report.flagged_claims)


def test_mean_groundedness_can_mask_hallucinations():
    """Documents WHY claim-level detection is the gate, not the mean score."""
    bot = RagChatbot(llm=MockLLM(hallucinate=True))
    response = bot.ask(QUESTION)
    mean_score = groundedness(response.answer, response.context_text)
    report = detect_hallucinations(response.answer, response.context_text)
    # The mean stays deceptively high even though a claim is fabricated.
    assert mean_score > 0.6
    assert report.is_hallucinated
