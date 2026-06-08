# su26-ai301-contribution
# Contribution 1: `metrics` does not work well with `vmap`

## Phase I: Issue Selection

**Contribution Number:** 1  
**Student:** Chen-Kuan (Brian) Liao  
**Issue:** https://github.com/google/flax/issues/5483  
**Issue Title:** `metrics` does not work well with `vmap`  
**Project:** google/flax  
**Status:** Phase I Complete

---

## Problem Summary

This Flax issue reports that `flax.nnx.metrics.Average` does not behave correctly when used with `nnx.vmap`. The issue author's workflow initializes multiple models in parallel, one per seed, and returns a separate optimizer and `nnx.MultiMetric` for each model. Before calling `metrics.reset()`, the metric state is correctly batched: fields such as `count` and `total` have shape `(3,)`, matching the three vmapped models.

After calling `metrics.reset()`, those same metric states become scalar arrays instead of keeping their vmapped shape. For example, `count` changes from an array with shape `(3,)` to a scalar `Array(0, dtype=int32)`, and `total` changes from shape `(3,)` to scalar `Array(0., dtype=float32)`. This shape change later causes a `vmap` error because the transformed code expects an axis-0 dimension, but the reset metric state no longer has one.

In short, the metric object appears to be valid immediately after vmapped initialization, but `reset()` loses the batched state shape. The issue is likely related to how Flax NNX metric state is reset inside `flax/nnx/training/metrics.py`.

---

## Why I Chose This Issue

I chose this issue because it connects directly to my interests in ML systems, especially framework behavior around JAX transformations and state management. The bug involves the interaction between `vmap`, mutable NNX state, and user-facing training utilities, which is the kind of systems-level framework issue I want to understand better.

This is also a good fit for my background as a CS PhD student working on ML systems such as JAX and PyTorch. I am interested in how deep learning frameworks preserve shape, dtype, and state semantics across transformations like `vmap`, `jit`, and gradient-based training loops. This issue is small enough to be approachable for a first open-source contribution, but it still requires understanding real framework internals instead of only making a documentation or formatting change.

---

## Phase I Notes

For Phase I, I am only documenting the issue link, a problem summary, and why I chose this issue. I have not started the reproduction process, solution design, testing strategy, implementation, pull request, or maintainer feedback log yet.

Those will be handled in later phases:

- **Phase II:** understanding the issue, reproduction process, and solution approach
- **Phase III:** testing strategy and implementation notes
- **Phase IV:** pull request link, summary, and maintainer feedback log

---

## Phase II: Understanding, Reproduction, and Solution Approach

### Understanding the Issue

#### Problem Description

[In your own words, what is broken or missing?]

#### Expected Behavior

[What should happen?]

#### Current Behavior

[What actually happens?]

#### Affected Components

[Which parts of the codebase are involved?]

### Reproduction Process

#### Environment Setup

[Notes on setting up your local development environment, including challenges you faced and how you solved them]

#### Steps to Reproduce

1. [Step 1]
2. [Step 2]
3. [Observed result]

#### Reproduction Evidence

- **Commit showing reproduction:** [Link to commit in your fork]
- **Screenshots/logs:** [If applicable]
- **My findings:** [What you discovered during reproduction]

### Solution Approach

#### Analysis

[Your analysis of the root cause: what is causing the issue?]

#### Proposed Solution

[High-level description of your fix approach]

#### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** [Restate the problem]

**Match:** [What similar patterns/solutions exist in the codebase?]

**Plan:** [Step-by-step implementation plan]
1. [Modify file X to do Y]
2. [Add function Z]
3. [Update tests]

**Implement:** [Link to your branch/commits as you work]

**Review:** [Self-review checklist - does it follow the project's contribution guidelines?]

**Evaluate:** [How will you verify it works?]

---

## Phase III: Testing Strategy and Implementation Notes

### Testing Strategy

#### Unit Tests

- [ ] Test case 1: [Description]
- [ ] Test case 2: [Description]
- [ ] Test case 3: [Description]

#### Integration Tests

- [ ] Integration scenario 1
- [ ] Integration scenario 2

#### Manual Testing

[What you tested manually and results]

### Implementation Notes

#### Week [X] Progress

[What you built this week, challenges faced, decisions made]

#### Week [Y] Progress

[Continue documenting as you work]

#### Code Changes

- **Files modified:** [List]
- **Key commits:** [Links to important commits]
- **Approach decisions:** [Why you chose certain approaches]

---

## Phase IV: Pull Request and Maintainer Feedback

### Pull Request

**PR Link:** [GitHub PR URL when submitted]

**PR Description:** [Draft or final PR description - much of the content above can be adapted]

**Maintainer Feedback:**
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

**Status:** [Awaiting review / Iterating / Approved / Merged]

---

## Learnings & Reflections

### Technical Skills Gained

[What you learned technically]

### Challenges Overcome

[What was hard and how you solved it]

### What I'd Do Differently Next Time

[Reflection on your process]

---

## Resources Used

- GitHub issue: https://github.com/google/flax/issues/5483
- Flax repository: https://github.com/google/flax
