# su26-ai301-contribution

Open-source contribution log for **SU26 AI301**. This repository documents my contributions to real open-source projects — for each issue I work through the full lifecycle: selecting the issue, understanding and reproducing it, designing and testing a fix, and opening a pull request with maintainer follow-up.

**Student:** Chen-Kuan (Brian) Liao  
**Project I'm contributing to:** [google/flax](https://github.com/google/flax) — the JAX-based neural network library (NNX API).

Each contribution is tracked in its own Markdown file, organized into four phases:

- **Phase I — Issue Selection:** issue link, problem summary, and why I chose it.
- **Phase II — Understanding, Reproduction & Solution Approach:** root-cause analysis, a reproducible repro, and a proposed solution + plan.
- **Phase III — Testing Strategy & Implementation Notes:** test design and implementation details.
- **Phase IV — Pull Request & Maintainer Feedback:** PR link, summary, and the maintainer review log.

---

## Contributions

| # | Issue | Title | Status |
|---|-------|-------|--------|
| 1 | [#5483](https://github.com/google/flax/issues/5483) | `metrics` does not work well with `vmap` | PR [#5491](https://github.com/google/flax/pull/5491) open — regression test only; **approved by @samanklesaria**, awaiting extra reviewer sign-off |
| 2 | [#5512](https://github.com/google/flax/issues/5512) | `flax.nnx.cond` causes tracing cache misses | PR [#5518](https://github.com/google/flax/pull/5518) open — fix + regression tests, expanded to all four control-flow ops; **approved by @samanklesaria** |

### [Contribution 1 — `metrics` does not work well with `vmap`](issue-1-5483.md)

`nnx.metrics.Average` (and metrics built on it) could lose their batched state shape on `reset()` when constructed under `nnx.vmap`, collapsing shape `(N,)` to a scalar `()` and breaking later vmapped updates. The crash reproduces on the reporter's released versions (flax 0.12.0 / jax 0.7.2) but not on current `main`, so the contribution is scoped to a **regression test only** that locks in the current vmap `reset()` behavior. Full writeup: [issue-1-5483.md](issue-1-5483.md).

### [Contribution 2 — `flax.nnx.cond` causes tracing cache misses](issue-2-5512.md)

`nnx.cond` / `nnx.switch` wrap each branch function in a freshly constructed `SimpleCondFn` on every call. Because JAX's tracing cache keys on callable identity, the new wrapper objects defeat the cache, forcing re-tracing (and blocking the persistent compilation cache) on every invocation. The fix memoizes the wrappers in a `WeakKeyDictionary` keyed by `(function, graph)`. During review the scope grew — at the maintainer's request — to cover all four NNX control-flow wrappers (`cond`, `switch`, `while_loop`, `fori_loop`) via one shared caching helper. This bug reproduces on current `main` (flax 0.12.7 / jax 0.10.1). Full writeup: [issue-2-5512.md](issue-2-5512.md).

---

## Resources

- Flax repository: https://github.com/google/flax
- My Flax fork: https://github.com/chenkuanliao/flax
