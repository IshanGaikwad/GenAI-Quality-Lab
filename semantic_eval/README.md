# Semantic eval stage (non-blocking)

The lexical metrics in [`evals/`](../evals) gate CI because they are free, fast,
and deterministic. But they measure **word overlap, not meaning**, which has a
correctness ceiling this stage exists to expose:

- A **negated claim** ("PTO *cannot* be carried over") shares almost every token
  with its source, so lexical groundedness rates it ~0.89 and does **not** flag
  it — a false negative.
- A **faithful paraphrase** shares few tokens, so lexical groundedness flags it
  as a hallucination — a false positive.

This stage closes that gap with real models, and reports where the two
approaches disagree:

| Signal | Model | Fixes |
|---|---|---|
| Entailment-based groundedness | NLI cross-encoder (`nli-distilroberta-base`) | Catches **contradiction / negation** |
| Semantic relevance & answer similarity | Bi-encoder (`all-MiniLM-L6-v2`) | Credits **paraphrase** (no shared keywords) |

## Why it is separate and non-blocking

It needs `torch` and downloaded models — slow, heavier, and version-sensitive,
the opposite of what a *blocking* gate should be. So it lives outside `evals/`
(untouched by the 100%-coverage lexical gate) and runs as its own
[GitHub Actions workflow](../.github/workflows/semantic-eval.yml) that reports
rather than gates. This is the "separate, non-blocking evaluation stage" the
top-level README describes.

## Run it

```bash
pip install -r semantic_eval/requirements.txt   # torch + models, ~separate env
python -m semantic_eval.run                      # prints a scorecard, writes report.json
```

## Honest caveat

NLI groundedness is **conservative**: it counts only strict entailment, so
correct extractive answers often land on `neutral` (unverifiable), not
`entailment`. A low entailment count is therefore **not** a failure — the strong,
reliable signal here is **contradiction detection**, which is what lexical
overlap provably misses.
