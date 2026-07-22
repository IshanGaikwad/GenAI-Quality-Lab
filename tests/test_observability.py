"""The observability layer scores each interaction with the same metrics the
gate uses and hands a serializable trace to a sink. It must stay a pure, offline
no-op when no sink is wired, and build the trace identically regardless of which
backend (Langfuse, Phoenix, ...) ultimately consumes it.
"""

from app.chatbot import RagChatbot
from observability.tracing import build_trace, score_response, traced_ask

bot = RagChatbot()
QUESTION = "How many PTO days do full-time employees get per year?"


def test_score_response_reports_the_gate_metrics():
    scores = score_response(bot.ask(QUESTION))
    assert set(scores) == {"groundedness", "relevance", "hallucinated"}
    assert scores["groundedness"] == 1.0
    assert scores["hallucinated"] is False


def test_traced_ask_is_a_noop_without_a_sink():
    # No sink, no network — the property the whole suite depends on.
    response, scores = traced_ask(bot, QUESTION)
    assert "20" in response.answer
    assert scores["groundedness"] == 1.0


def test_traced_ask_emits_one_scored_trace_to_the_sink():
    captured: list[dict] = []
    response, scores = traced_ask(bot, QUESTION, sink=captured.append)

    assert len(captured) == 1
    trace = captured[0]
    assert trace["output"] == response.answer
    assert trace["scores"] == scores
    assert [s["name"] for s in trace["spans"]] == ["retrieval", "generation"]
    # the retrieval span carries the ranked documents
    assert trace["spans"][0]["output"][0]["doc_id"]


def test_build_trace_shape_is_backend_agnostic():
    response = bot.ask(QUESTION)
    trace = build_trace(QUESTION, response, score_response(response))
    assert trace["name"] == "benefits-assistant"
    assert trace["input"] == QUESTION
    assert trace["spans"][1]["name"] == "generation"
    assert set(trace["scores"]) == {"groundedness", "relevance", "hallucinated"}
