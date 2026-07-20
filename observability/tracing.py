"""Optional observability hooks (Langfuse), no-op by default.

The evaluation suite never depends on a network service — observability is
additive. If LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set and the
``langfuse`` package is installed, each chatbot interaction is exported as a
trace with the retrieval and generation steps as spans, and the eval scores
attached — the same pattern used with Langfuse/Arize Phoenix in production.

Usage:
    from observability.tracing import traced_ask
    response, scores = traced_ask(bot, "How many PTO days do I get?")
"""

from __future__ import annotations

import os

from evals.metrics import answer_relevance, detect_hallucinations, groundedness


def _get_langfuse():
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import Langfuse  # type: ignore

        return Langfuse()
    except ImportError:
        return None


def score_response(response) -> dict[str, float | bool]:
    context = response.context_text
    return {
        "groundedness": round(groundedness(response.answer, context), 3),
        "relevance": round(answer_relevance(response.question, response.answer), 3),
        "hallucinated": detect_hallucinations(response.answer, context).is_hallucinated,
    }


def traced_ask(bot, question: str):
    """Ask the bot a question; export a scored trace if Langfuse is configured."""
    response = bot.ask(question)
    scores = score_response(response)

    client = _get_langfuse()
    if client is not None:
        trace = client.trace(name="benefits-assistant", input=question, output=response.answer)
        trace.span(
            name="retrieval",
            input=question,
            output=[{"doc_id": d.doc_id, "score": d.score} for d in response.retrieved],
        )
        trace.span(name="generation", input=response.prompt, output=response.answer)
        for name, value in scores.items():
            trace.score(name=name, value=float(value))
        client.flush()

    return response, scores
