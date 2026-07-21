"""Run the semantic evaluation stage and emit a report.

Non-blocking by design: this prints a scorecard and writes ``report.json``; it
never fails the build. Its job is to surface where the fast lexical gate and
richer semantic judgment disagree — the two places the README calls out:

- **negation**, which NLI catches but token overlap certifies as grounded, and
- **paraphrase**, which embeddings credit but token overlap flags or under-scores.

Usage:  python -m semantic_eval.run
"""

from __future__ import annotations

import json
from pathlib import Path

from app.chatbot import RagChatbot
from evals.metrics import answer_relevance, detect_hallucinations, groundedness
from collections import Counter

from semantic_eval.metrics import (
    entailment_report,
    semantic_relevance,
)

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = json.loads((ROOT / "evals" / "datasets" / "golden_set.json").read_text())
REPORT_PATH = Path(__file__).resolve().parent / "report.json"

# Canonical adversarial pairs from the README's argument: identical topic,
# opposite/paraphrased meaning — where lexical overlap provably fails.
CONTEXT = "Unused PTO up to 5 days may be carried over into the next calendar year."
PROBES = [
    ("false-negation", "Unused PTO cannot be carried over into the next calendar year."),
    ("true-paraphrase", "Workers keep unused vacation, rolling over up to five days annually."),
]


def _score_golden() -> list[dict]:
    bot = RagChatbot()
    rows = []
    for case in GOLDEN["cases"]:
        if case["out_of_scope"]:
            continue
        response = bot.ask(case["question"])
        context = response.context_text
        labels = Counter(c.label for c in entailment_report(context, response.answer))
        rows.append(
            {
                "id": case["id"],
                "answer": response.answer,
                "lexical_groundedness": round(groundedness(response.answer, context), 3),
                # Per-claim NLI verdicts. Contradiction is the strong signal
                # (definitely unfaithful); neutral is the ambiguous middle
                # (unverifiable), so a low entailment count is NOT a failure.
                "entailed": labels["entailment"],
                "neutral": labels["neutral"],
                "contradicted": labels["contradiction"],
                "lexical_relevance": round(answer_relevance(case["question"], response.answer), 3),
                "semantic_relevance": round(semantic_relevance(case["question"], response.answer), 3),
            }
        )
    return rows


def _score_probes() -> list[dict]:
    rows = []
    for name, claim in PROBES:
        report = entailment_report(CONTEXT, claim)
        rows.append(
            {
                "probe": name,
                "claim": claim,
                "lexical_groundedness": round(groundedness(claim, CONTEXT), 3),
                "lexical_flags_hallucination": detect_hallucinations(claim, CONTEXT).is_hallucinated,
                "nli_label": report[0].label if report else None,
            }
        )
    return rows


def run() -> dict:
    return {"golden": _score_golden(), "probes": _score_probes()}


def _print(report: dict) -> None:
    print("\n=== Semantic vs lexical on the golden set ===")
    print(f"{'case':22} {'lex_grnd':>9} {'NLI(e/n/c)':>11} {'lex_rel':>8} {'sem_rel':>8}")
    for r in report["golden"]:
        nli = f"{r['entailed']}/{r['neutral']}/{r['contradicted']}"
        flag = "  <- CONTRADICTION" if r["contradicted"] else ""
        print(
            f"{r['id']:22} {r['lexical_groundedness']:>9} {nli:>11} "
            f"{r['lexical_relevance']:>8} {r['semantic_relevance']:>8}{flag}"
        )
    print("  NLI(e/n/c) = entailed / neutral / contradicted claims")

    print("\n=== Adversarial probes: where lexical overlap fails ===")
    for r in report["probes"]:
        print(f"[{r['probe']}] {r['claim']}")
        print(
            f"    lexical: groundedness={r['lexical_groundedness']} "
            f"flagged_hallucination={r['lexical_flags_hallucination']}  |  "
            f"NLI: {r['nli_label']}"
        )


def main() -> None:
    report = run()
    _print(report)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH.relative_to(ROOT)} (non-blocking stage — no gate).")


if __name__ == "__main__":
    main()
