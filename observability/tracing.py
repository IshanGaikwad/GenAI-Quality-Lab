"""Optional observability: score each interaction and export it as a trace.

The evaluation suite never depends on a network service — observability is
additive. This module is deliberately **SDK-agnostic**: it scores an answer and
builds a serializable trace, then hands that trace to a ``sink`` you provide.
Wire the sink to Langfuse, Arize Phoenix, or your own store (see the README for
a Langfuse example). Keeping any specific SDK out of this module keeps it fully
testable and free of version-specific breakage — with no sink it is a pure,
offline no-op.

    from observability.tracing import traced_ask
    response, scores = traced_ask(bot, "How many PTO days do I get?", sink=my_sink)
"""

from __future__ import annotations

from typing import Callable

from evals.metrics import answer_relevance, detect_hallucinations, groundedness

# A sink receives one trace dict and exports it however it likes.
TraceSink = Callable[[dict], None]


def score_response(response) -> dict[str, float | bool]:
    """Attach the same eval scores the CI gate uses to a chatbot response."""
    context = response.context_text
    return {
        "groundedness": round(groundedness(response.answer, context), 3),
        "relevance": round(answer_relevance(response.question, response.answer), 3),
        "hallucinated": detect_hallucinations(response.answer, context).is_hallucinated,
    }


def build_trace(question: str, response, scores: dict) -> dict:
    """A serializable trace — a retrieval span, a generation span, and the eval
    scores — the shape observability backends (Langfuse, Phoenix) expect."""
    return {
        "name": "benefits-assistant",
        "input": question,
        "output": response.answer,
        "spans": [
            {
                "name": "retrieval",
                "input": question,
                "output": [
                    {"doc_id": d.doc_id, "score": d.score} for d in response.retrieved
                ],
            },
            {"name": "generation", "input": response.prompt, "output": response.answer},
        ],
        "scores": scores,
    }


def traced_ask(bot, question: str, sink: TraceSink | None = None):
    """Ask the bot, score the answer, and emit a scored trace to ``sink``.

    Returns ``(response, scores)``. With no sink it is a pure, offline no-op —
    exactly what the test suite relies on.
    """
    response = bot.ask(question)
    scores = score_response(response)
    if sink is not None:
        sink(build_trace(question, response, scores))
    return response, scores
