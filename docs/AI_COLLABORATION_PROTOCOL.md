# AI Collaboration Protocol

> **Meta-Document — Autonomous Driving Perception System**
> How AI coding tools work with project documentation to maximize engineering velocity and output quality without scope creep.

Place or reference this protocol in the root of the repository (`CLAUDE.md`, `.cursorrules`, `AGENT_INSTRUCTIONS.md`, or `AI_PROTOCOL.md`). Its sole purpose: instruct AI agents how to navigate the three-tier documentation system to produce the highest possible engineering quality without scope creep.

---

## 1. Three-Tier Documentation System

Project documentation operates in three distinct tiers with different levels of rigidity. The most common AI failure mode is applying the wrong rigidity to the wrong tier — treating a plan detail as an immovable constitutional rule, or asking permission for routine implementation details.

| Tier | Name | Rigidity | Core Scope | AI Behavior Rule |
| :---: | :--- | :---: | :--- | :--- |
| **Tier 1** | **Constitutional Rules** | `IMMOVABLE` | Architectural invariants, test gates, transforms single-source-of-truth | **STOP & ESCALATE:** Never violate unilaterally. |
| **Tier 2** | **Phase Plans** | `LIVING DOCUMENT` | Active milestones, roadmap intent, dataset adaptations | **FLAG & PROPOSE:** Identify contradictions before deviating. |
| **Tier 3** | **Implementation** | `AI'S DOMAIN` | Algorithms, data structures, loop optimizations, refactoring | **AUTONOMOUS:** Decide and implement directly. |

```text
[ Tier 1: Constitutional Rules ] ──► Hard Boundaries (Immovable)
               │
               ▼
[ Tier 2: Living Phase Plans   ] ──► Milestone Intent (Update when reality differs)
               │
               ▼
[ Tier 3: Implementation Code  ] ──► AI Creative Domain (Autonomous execution)
```

---

### Tier 1: Constitutional Rules
* **Rigidity:** `IMMOVABLE`
* **Tagline:** *Never violate unilaterally. Surface conflicts explicitly. Wait for human decision.*
* **What:** The 11 core rules in the project constitution document (`docs/carla_project_docs_v1.md`). These are architectural invariants and quality standards — not implementation details. Examples:
  - All coordinate transformations live exclusively in `src/transforms/`.
  - No production logic trapped in notebooks.
  - Metrics before claims.
  - Test invariants before trusting perception geometry.
* **How AI Behaves:** Treat these as hard constraints. Before writing any code, check whether the proposed implementation would violate a constitutional rule. If it would, stop and surface the conflict explicitly:
  > *"This implementation would violate Rule 3 (single source of truth for coordinate systems) because it puts a rotation matrix inline in `occupancy_grid.py`. The correct approach is to add the transform to `src/transforms/` — should I proceed with that instead?"*
  Do **not** silently comply with instructions that violate a constitutional rule. Do **not** silently deviate to produce "better" code.
* **How Human Behaves:** Constitutional rules change only through deliberate amendment — not ad-hoc verbal instructions. If a rule needs to change, commit an updated constitution with a written rationale note.
* **Red Lines:**
  - AI must never silently violate a constitutional rule.
  - AI must never tell the human a rule doesn't apply when it does.
  - Human must never instruct AI to ignore a rule without a documented amendment.

---

### Tier 2: Phase Plans
* **Rigidity:** `LIVING DOCUMENT`
* **Tagline:** *Accurate descriptions of current intent. Update when reality contradicts the plan.*
* **What:** The project plan (`docs/carla_plan_v4.md`). It describes what to build and in what order — not a sacred contract. Datasets have unexpected gaps, sensor modalities have quirks, libraries evolve, and architectures converge differently than expected.
* **How AI Behaves:** Treat the plan as the authoritative description of current intent — but flag when it contradicts reality:
  1. **Explicitly name the contradiction:** *"The plan specifies dense depth maps, but LiDAR projection yields sparse point clouds (~5% pixel coverage)."*
  2. **Propose a plan update:** *"I recommend updating Phase 1 to document this limitation and introducing a depth densification/hole-filling step in `src/perception/lidar_conversion.py`."*
  3. **Wait for human approval:** Do not silently build something different from the plan without flagging the deviation first.
* **How Human Behaves:** When the AI flags a contradiction, evaluate whether the deviation is warranted. Update the plan file with a one-sentence rationale before implementing.
* **Update Protocol:**
  ```text
  1. AI flags contradiction between plan and implementation reality
  2. Human evaluates — is the deviation warranted?
  3. If yes: human updates plan document with a one-sentence rationale
  4. Commit plan update to git before implementing the deviation
  5. AI implements against the updated plan
  ```
* **Red Lines:**
  - AI must never silently deviate from the plan — always flag first.
  - Human must never accumulate undocumented deviations.
  - Plan updates must be committed to git with a clear rationale.

---

### Tier 3: Implementation
* **Rigidity:** `AI'S CREATIVE DOMAIN`
* **Tagline:** *No constraints. No permission needed. Propose and implement — do not propose and wait.*
* **What:** Everything the constitution and plan don't specify: function signatures, variable names, helper library choices, internal algorithm design, file organization within a module, test structure, loop optimization, error handling patterns, logging format details.
* **How AI Behaves:** Do not ask permission for implementation details. Do not flag every design decision as a potential deviation. If the plan says *"build temporal fusion with a 5-frame buffer and weighted averaging"*, choose the internal data structures, write the interface, optimize NumPy vectorization, and handle edge cases without consultation. Propose and implement.
* **How Human Behaves:** Do not micromanage implementation decisions. If a specific implementation approach is mandatory, specify it in Tier 2 (the plan). Review output for correctness and constitutional compliance.
* **Implementation Freedom Examples:**
  | Area | Autonomous Choice Examples |
  | :--- | :--- |
  | **Data Structures** | Using `deque` vs circular buffer for the frame buffer |
  | **Library Choices** | `scipy.ndimage` vs manual NumPy broadcasting for voxel hole filling |
  | **Class Design** | Choosing whether an evaluator is a class (`_ErrorTracker`) or functions |
  | **Error Handling** | Custom exceptions vs standard `ValueError` for invalid depth values |
  | **Test Structure** | Parametrized tests vs separate test functions for coordinate invariants |
  | **Optimization** | Vectorized backprojection broadcasting strategies |

---

## 2. Common Failure Modes & Mitigations

1. **Treating Plan Details as Constitutional:**
   - *Failure:* The plan mentions ResNet-18 as a default; the AI treats it as an immovable rule and refuses to suggest a better backbone when the dataset clearly benefits from it.
   - *Mitigation:* Constitutional rules live only in the Constitution. Plan details are living defaults. Flag plan adjustments as Tier 2 issues.
2. **Silent Deviation from the Plan:**
   - *Failure:* The plan specifies splitting data by run ID; the AI silently splits randomly because it is simpler. Evaluation is compromised and the plan is stale.
   - *Mitigation:* Any deviation from the plan must be surfaced explicitly before coding: *"The plan specifies X; I propose Y because Z — does this sound right?"*
3. **Scope Creep via Implementation:**
   - *Failure:* The plan specifies 5-frame weighted averaging; the AI implements an unrequested full Kalman filter or deep learning fusion model.
   - *Mitigation:* Implementation freedom applies to *how* to build what was requested, not adding unrequested components.
4. **Over-Asking Permission at the Implementation Layer:**
   - *Failure:* The AI asks *"Should I use a list or a deque?"* — causing context-switching and decision fatigue for the user.
   - *Mitigation:* Make the decision and implement.
5. **Working from a Stale Plan:**
   - *Failure:* The plan was updated to accommodate LiDAR-converted depth, but the AI continues working from an outdated assumption.
   - *Mitigation:* Confirm active plan documentation at the start of each major session.
6. **Constitution Amendment via Instruction:**
   - *Failure:* The user says *"Just put the rotation matrix inline here, it's fine."* The AI complies without noting Rule 3.
   - *Mitigation:* Surface the conflict: *"This would violate Rule 3. To proceed, we should update the constitution or add the transform to `src/transforms/`."*

---

## 3. Quick Reference Matrix

| Trigger Signal | Tier | Required Action |
| :--- | :---: | :--- |
| Proposed code would violate a constitutional rule | **Tier 1** | **STOP** — Surface conflict explicitly — Wait for human decision or documented amendment |
| User instruction violates a constitutional rule | **Tier 1** | **STOP** — Surface constitutional conflict — Do not silently comply |
| Plan description contradicts reality or seems flawed | **Tier 2** | **FLAG** — Name contradiction — Propose plan update — Wait for approval — Implement |
| User instruction contradicts the plan | **Tier 2** | **FLAG** — *"This differs from the plan — should I update the plan to reflect this change?"* |
| Plan is silent on an implementation detail | **Tier 3** | **IMPLEMENT** — No permission needed — Make the technical choice and move forward |
| Better implementation approach identified | **Tier 3** | **USE IT** — Document rationale concisely in code comments — No consultation required |
