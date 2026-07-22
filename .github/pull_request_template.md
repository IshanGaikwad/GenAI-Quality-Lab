<!-- Thanks for contributing! Keep changes small and focused. -->

## What & why

<!-- What does this change, and why? Link any related issue. -->

## Type of change

- [ ] Bug fix
- [ ] New metric / feature
- [ ] Golden-set case(s)
- [ ] Docs / CI
- [ ] Refactor (no behavior change)

## Checklist

- [ ] `pytest --cov=app --cov=evals --cov=observability --cov-fail-under=100` passes — 100% coverage held, no test weakened to pass
- [ ] New behavior comes with tests
- [ ] New metrics have a **documented** threshold, not a magic number
- [ ] Golden-set cases follow the `type` schema; a case the system can't yet meet uses `strict` xfail rather than a lowered threshold
- [ ] The blocking gate stays fast & offline — anything needing `torch`, model downloads, or network lives in `semantic_eval/`, not the gate or root `requirements.txt`
- [ ] Commit messages use conventional prefixes (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`)
