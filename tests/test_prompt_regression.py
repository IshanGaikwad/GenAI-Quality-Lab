"""Prompt regression tests.

The system prompt is production configuration: silent edits change model
behavior exactly like untested code changes. These tests pin the guardrail
clauses and the prompt's structural contract so any modification fails CI
and forces a deliberate review.
"""

from app.chatbot import FALLBACK, PROMPT_TEMPLATE, RagChatbot

REQUIRED_GUARDRAILS = [
    "Answer ONLY from the context below",
    "Do not speculate",
    "Do not invent policies, numbers, or dates",
]


def test_guardrail_clauses_are_present():
    for clause in REQUIRED_GUARDRAILS:
        assert clause in PROMPT_TEMPLATE, (
            f"Guardrail clause removed or reworded: {clause!r}. "
            "If intentional, update this test in the same change."
        )


def test_fallback_string_is_pinned_in_prompt():
    """The prompt instructs the model to emit the exact fallback string; the
    application code matches on that same string. Pin the coupling."""
    assert FALLBACK in PROMPT_TEMPLATE


def test_prompt_structure_context_before_question():
    """RAG prompts must interpolate context before the question — inverting
    the order changes model behavior and breaks the parser contract."""
    assert 0 <= PROMPT_TEMPLATE.find("{context}") < PROMPT_TEMPLATE.find("{question}")


def test_prompt_is_fully_interpolated_at_runtime():
    response = RagChatbot().ask("How many PTO days do full-time employees get per year?")
    assert "{context}" not in response.prompt
    assert "{question}" not in response.prompt
    assert response.question in response.prompt
