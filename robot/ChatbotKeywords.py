"""Robot Framework keyword library wrapping the chatbot and eval metrics."""

from app.chatbot import FALLBACK, RagChatbot
from evals.metrics import detect_hallucinations, groundedness


class ChatbotKeywords:
    ROBOT_LIBRARY_SCOPE = "SUITE"

    def __init__(self):
        self._bot = RagChatbot()
        self._response = None

    def ask_question(self, question: str):
        self._response = self._bot.ask(question)
        return self._response.answer

    def answer_should_contain(self, expected: str):
        answer = self._require_response().answer
        if expected.lower() not in answer.lower():
            raise AssertionError(
                f"Answer does not contain {expected!r}. Answer: {answer}"
            )

    def answer_should_be_grounded(self, threshold: str = "0.8"):
        response = self._require_response()
        score = groundedness(response.answer, response.context_text)
        if score < float(threshold):
            raise AssertionError(
                f"Groundedness {score:.2f} below threshold {threshold}. "
                f"Answer: {response.answer}"
            )

    def answer_should_have_no_hallucinated_claims(self):
        response = self._require_response()
        report = detect_hallucinations(response.answer, response.context_text)
        if report.is_hallucinated:
            raise AssertionError(
                f"Hallucinated claims detected: {report.flagged_claims}"
            )

    def answer_should_be_the_fallback(self):
        answer = self._require_response().answer
        if answer != FALLBACK:
            raise AssertionError(f"Expected exact fallback, got: {answer}")

    def _require_response(self):
        if self._response is None:
            raise AssertionError("Call 'Ask Question' first.")
        return self._response
