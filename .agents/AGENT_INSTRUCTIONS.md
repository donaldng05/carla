# AGENT_INSTRUCTIONS.md

## Purpose

This repository contains a CARLA-based autonomous driving perception project governed by three source-of-truth documents:

1. `docs/carla_project_docs_v1.md`
2. `docs/carla_plan_v4.md`
3. `docs/AI_COLLABORATION_PROTOCOL.md`

Any coding agent working in this repository must read and obey those three documents before making changes.

These instructions define how to behave inside the repo. They are not suggestions.

---

## Governance Order

Always interpret project decisions using this order of precedence:

### Tier 1 — Hard Constraints

`docs/carla_project_docs_v1.md`

* These are non-negotiable rules.
* Never silently violate them.
* If a requested change conflicts with the Constitution, stop and report the conflict.

### Tier 2 — Current Project Intent

`docs/carla_plan_v4.md`

* This defines the current architecture, current phase ordering, milestone structure, and implementation priorities.
* Follow it unless reality makes it impossible.
* If implementation reality contradicts the plan, explicitly flag the contradiction before proceeding.

### Tier 3 — Collaboration Procedure

`docs/AI_COLLABORATION_PROTOCOL.md`

* This defines how to interpret ambiguity, when to implement directly, and when to escalate.
* Use it to decide whether something is:

  * a rule conflict,
  * a plan contradiction,
  * or a normal implementation detail.

---

## Required Startup Behavior

Before writing or editing code, always do the following:

1. Read:

   * `docs/carla_project_docs_v1.md`
   * `docs/carla_plan_v4.md`
   * `docs/AI_COLLABORATION_PROTOCOL.md`
   * `CURRENT_STATE.md` if it exists

2. Output a short **Governance Check** before coding.

Use this exact format:

```text
Governance Check
- Current phase: <phase name / number>
- Current task: <task being implemented>
- Tier classification: <Tier 1 / Tier 2 / Tier 3>
- Files to touch: <list>
- Relevant Constitution rules: <rule numbers>
- Relevant Plan section: <phase / module / milestone>
- Conflicts detected: <none or describe>
- Assumptions: <short list>
```

Do not skip this.

---

## Behavioral Rules

### 1. Never silently deviate from the Constitution

If a request conflicts with the Constitution:

* stop,
* explain the conflict clearly,
* propose a compliant alternative.

### 2. Never silently drift from the Plan

If the requested task conflicts with `docs/carla_plan_v4.md`:

* identify the exact contradiction,
* explain whether it is a small implementation adaptation or a real plan deviation,
* do not quietly change architecture.

### 3. Implement Tier 3 details directly

If something is only an implementation detail and does not conflict with Tier 1 or Tier 2:

* implement it directly,
* do not ask unnecessary questions,
* state assumptions briefly.

### 4. Prefer small, reviewable changes

Default to:

* one module,
* one test suite,
* one refactor,
* or one integration step at a time.

Do not attempt to implement an entire phase in one step unless explicitly asked.

### 5. Say what files you intend to modify before modifying them

For non-trivial work, always state the target file list first.

### 6. Stop at clear checkpoints

After finishing the requested task:

* summarize what was completed,
* list validation steps,
* list any remaining follow-up items,
* stop unless explicitly asked to continue.

---

## Repository Standards

### Source tree discipline

* Production logic belongs in `src/`
* Tests belong in `tests/`
* Notebooks are for exploration only
* No production logic may live only in notebooks

### Coordinate transform discipline

All coordinate math must go through `src/transforms/`.

Do not:

* duplicate transform logic,
* inline rotation math in random files,
* create alternate transform utilities elsewhere.

If a new transform is needed:

* add it to `src/transforms/`
* add tests for it

### Testing discipline

For any geometry, transform, occupancy, LiDAR projection, or safety logic:

* add or update tests
* do not treat visual plausibility as sufficient proof

### Visualization discipline

For geometry-heavy modules, visualization is part of validation.

### Reproducibility discipline

Any training or evaluation code must:

* use config files,
* preserve reproducibility,
* avoid hardcoded experiment settings,
* support logging and checkpointing where relevant

### C++ discipline

Any C++ work must maintain:

* proper CMake setup
* testability
* formatting consistency
* no throwaway one-file demo code

---

## Project-Specific Expectations

This project is:

* dataset-driven
* perception-centric
* CPU-first except for the behavioral cloning training phase
* explicitly not an RL project
* explicitly not a live CARLA simulator project

The current v4 plan includes a LiDAR-to-depth conversion workflow because the selected dataset provides LiDAR rather than dense depth. Treat that as part of the intended architecture, not as optional experimentation.

Do not introduce:

* reinforcement learning
* simulator-control features
* new autonomy subsystems outside the plan
* feature creep that breaks the project narrative

The core narrative must remain:

1. build occupancy representation,
2. improve it with temporal fusion,
3. detect failure modes with shadow-mode evaluation,
4. show downstream effects in BC model behavior,
5. show safety monitor triggers on the same classes of scenarios.

Protect this causal chain.

---

## Definition of Done

A task is only done when all of the following are true where applicable:

* code is placed in the correct repo location
* tests are added or updated
* imports are clean
* assumptions are stated
* validation steps are provided
* no Constitution rules were violated
* no silent Plan deviations were introduced

For geometry-heavy code, “done” also means:

* there is a reasonable validation path,
* and visual or numerical checks exist.

---

## Preferred Response Pattern for Coding Tasks

Use this response structure for substantial work:

### Before coding

```text
Governance Check
- Current phase: ...
- Current task: ...
- Tier classification: ...
- Files to touch: ...
- Relevant Constitution rules: ...
- Relevant Plan section: ...
- Conflicts detected: ...
- Assumptions: ...
```

### After coding

```text
Completed
- <short bullet list of what was implemented>

Validation
- <commands or steps to run>

Notes
- <important assumptions / limitations>

Next logical task
- <one small next step>
```

---

## Escalation Rules

Escalate instead of coding if:

1. the task violates the Constitution
2. the task contradicts the current Plan in a non-trivial way
3. required dataset assumptions are false
4. the requested work would cause major scope creep
5. the task requires choosing between multiple architecture directions not already resolved by the docs

When escalating:

* identify the issue precisely
* cite which governance layer caused the stop
* propose 1–2 compliant alternatives

---

## Default Philosophy

Optimize for:

* correctness,
* testability,
* reproducibility,
* reviewability,
* and alignment with the project narrative.

Do not optimize for:

* maximum feature count,
* unnecessary architectural complexity,
* or impressive-sounding but weakly justified additions.

When in doubt:

* choose the simpler implementation,
* preserve the causal chain,
* and keep the repo defensible in a technical interview.
