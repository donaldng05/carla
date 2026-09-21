# CARLA Project Constitution & Dataset Contract

> **Version:** `v1.0`  
> **Governance Scope:** All development decisions, architecture design, and evaluation pipelines for the CARLA-based autonomous driving perception project.

---

# Part 1: CARLA Project Constitution

## 1. Purpose & Core Philosophy
This document governs all development decisions for the CARLA-based autonomous driving perception project. It exists to prevent scope creep, maintain architectural integrity, and ensure every deliverable can be defended under rigorous technical review. Any AI coding assistant or human developer working on this project must treat these rules as hard constraints, not suggestions.

* **Primary Goal:** Build and evaluate a perception-centric autonomous driving system using CARLA-generated datasets. The system demonstrates engineering maturity in 3D geometry, temporal sensor fusion, and failure-mode analysis — not production deployment readiness.
* **Core Deliverables:**
  1. A 3D semantic occupancy grid generated from RGB, depth, and semantic segmentation data.
  2. Temporal fusion across a 5-frame rolling buffer to stabilize scene representations under occlusion and sensor sparsity.
  3. A shadow-mode evaluation framework quantifying disagreement between baseline single-frame and temporal perception models.
  4. Scenario-stratified evaluation (Scenario Memory) measuring empirical IoU improvement across distinct operational domains.
  5. A disciplined coordinate transformation system acting as the single source of truth for all spatial math.
  6. A unified failure analysis correlating sensor sparsity and modality disagreement.
* **Non-Goals:**
  1. Achieving state-of-the-art benchmark leaderboard performance.
  2. Production deployment or real vehicle hardware integration.
  3. Reinforcement learning or multi-agent planning.
  4. HD map generation or full SLAM.
  5. Sensor fusion beyond the provided modalities (RGB, depth, segmentation).
  6. Live CARLA simulator execution — dataset-driven replay only.

---

## 2. The 11 Constitutional Rules

### Rule 1: Correctness Before Performance
> *No optimization may be introduced without first demonstrating the existing implementation is correct.*

Verify coordinate transforms before vectorizing them. Verify occupancy outputs before parallelizing them. Verify model behavior before accelerating inference. A 10x speedup on an incorrect implementation is simply a 10x faster wrong answer.

* ❌ **Bad:** Vectorize the backprojection loop for speed before verifying output geometry.
* ✅ **Good:** Run backprojection on golden data, verify visualized point cloud / BEV grid, then vectorize.

---

### Rule 2: Tests Before Trust — Geometry Is Dangerous
> *Geometry and transformation code must never be trusted without automated tests.*

Transformation bugs do not crash — they produce plausible-looking nonsense. A transposed rotation matrix looks like valid data. An inverted axis looks like sensor noise.
Mandatory test invariants:
- Identity transform leaves points unchanged.
- Pure rotation preserves inter-point distances.
- Inverse transform cleanly recovers original points ($T^{-1} T p = p$).
- Projection-backprojection round-trip is consistent within floating-point tolerance.

* ❌ **Bad:** Visually inspect the occupancy grid and assume it is correct because it "looks okay".
* ✅ **Good:** Run `pytest tests/test_transforms.py` verifying mathematical invariants before touching occupancy logic.

---

### Rule 3: Single Source of Truth for Coordinate Systems
> *All coordinate conversions must be implemented exclusively in `src/transforms/`.*

No coordinate math may be duplicated elsewhere. Any module requiring transforms must import from `src/transforms/`. This prevents frame-convention drift — the single most common silent failure mode in AV perception. Every transform function must have dedicated unit tests.

* ❌ **Bad:** Inline rotation matrix multiplication inside `occupancy_grid.py` or `temporal_fusion.py`.
* ✅ **Good:** `from src.transforms import camera_to_vehicle; p_veh = camera_to_vehicle(p_cam, extrinsics)`.

---

### Rule 4: No Production Logic in Notebooks
> *Notebooks exist only for exploration, visualization, and post-hoc reporting. All reusable functionality lives in `src/`.*

If code is duplicated between two notebooks, it belongs in `src/`. This distinction is what allows `pytest`, `mypy`, and CI workflows to function. A reviewer checking the repository should be able to run tests and verify behavior without opening Jupyter.

* ❌ **Bad:** Define `occupancy_grid()` or fusion loops inside a notebook cell.
* ✅ **Good:** `from src.perception.occupancy_grid import build_occupancy_grid` — the notebook only calls the module.

---

### Rule 5: Visualization Is a Validation Tool, Not a Luxury
> *Every geometry-heavy component must have a visualization method before it is considered complete.*

A visual plot reveals spatial bugs faster than 100 assertions. Required visualizers:
- Bird's-eye view (BEV) of the occupancy grid (free space, road, vehicles, pedestrians).
- Point cloud / depth overlay on RGB images for projection alignment.
- Temporal alignment overlays (frame $t$ and frame $t-1$ transformed into frame $t$, both rendered).
If a vehicle appears underground, depth backprojection has a sign error. If fused frames smear along trajectories, the pose transform chain is broken.

* ❌ **Bad:** Occupancy grid implemented, tests pass, immediately move to next task without visual inspection.
* ✅ **Good:** Occupancy grid implemented, unit tests pass, BEV visualization rendered and reviewed.

---

### Rule 6: Simple Before Clever
> *Prefer deterministic methods, interpretable logic, and transparent architectures over unnecessary complexity and premature abstraction.*

The goal is explainability and architectural clarity. A 5-frame pose-aligned occupancy accumulation with explicit weighting is vastly superior to an opaque black-box neural net if every geometric assumption can be defended in an engineering review.

* ❌ **Bad:** Add a complex diffusion-based trajectory predictor or recurrent model because it sounds impressive.
* ✅ **Good:** Implement crisp, vectorized ego-motion temporal fusion with clear threshold ablations.

---

### Rule 7: Reproducibility Is Mandatory
> *Every experiment and evaluation run must be fully reproducible from a configuration file and a deterministic seed.*

Requirements:
- Fixed random seeds in data loading and sampling routines.
- Structured configuration classes/files for all parameters (`configs/` or typed dataclasses).
- Explicit serialization of run summaries, parameters, and metric breakdowns to disk.
A result that cannot be reproduced from the repository is considered invalid.

* ❌ **Bad:** Run evaluation with interactive notebook parameters and hardcode results in markdown.
* ✅ **Good:** Execute evaluation via CLI / automated runners generating serialized JSON artifacts.

---

### Rule 8: Metrics Before Claims
> *Every performance claim must be supported by quantitative measurements from the evaluation pipeline.*

Qualitative language (*"appears more stable"*, *"seems to resolve occlusion"*) is forbidden unless backed by explicit metrics. This applies to occupancy grid resolution, temporal fusion IoU deltas, and shadow-mode disagreement rates.

* ❌ **Bad:** *"Temporal fusion visibly reduces sensor noise and improves vehicle detection."*
* ✅ **Good:** *"Temporal fusion improved vehicle-class IoU from 0.6039 to 0.6833 (+7.95% gain) across 50 evaluated anchors in high-disagreement scenarios."*

---

### Rule 9: Favor Engineering Depth Over Feature Count
> *When deciding between adding an unrequested new feature versus improving testing, documentation, or evaluation rigor, always choose depth.*

A well-tested, cleanly refactored 3D occupancy grid with property tests, shadow-mode disagreement clustering, and scenario-memory evaluation is worth far more than a half-baked multi-task repository.

* ❌ **Bad:** Add a toy lane-detection module or second model backbone to pad feature count.
* ✅ **Good:** Build scenario-stratified candidate scoring to evaluate temporal fusion under edge cases.

---

### Rule 10: Code Cleanliness & Static Verification
> *All codebase modules must adhere to strict typing, formatting, and linting standards.*

The codebase must pass `ruff check`, `ruff format --check`, and `mypy` with zero warnings. Dead exploratory code, unmaintained API scaffolding, and redundant math duplication must be aggressively pruned to keep the codebase maintainable.

* ❌ **Bad:** Leave unused REST API endpoints or recursive exploratory functions lingering in source files.
* ✅ **Good:** Enforce single sources of truth and prune dead code paths across data and perception packages.

---

### Rule 11: The Causal Chain Is the Core Narrative — Preserve It
> *The project's intellectual core is a causal narrative chain: identify failure modes with shadow-mode disagreement → prove temporal fusion resolves these failures in scenario memory → ground all claims in empirical data.*

Every evaluation and experiment must serve this narrative. Shadow mode is not an afterthought — it is the diagnostic instrument that identifies where single-frame perception fails (occlusion, LiDAR sparsity) and guides targeted evaluation.

* ❌ **Bad:** Disconnect evaluation metrics from shadow-mode failure analysis.
* ✅ **Good:** Use shadow-mode disagreement clustering to select anchors for scenario-memory IoU evaluation.

---

# Part 2: CARLA Dataset Contract

## 1. Required Modalities & Quality Gates

| Modality | Required? | Primary Use | Minimum Standard | Preferred Standard |
| :--- | :---: | :--- | :--- | :--- |
| **RGB Images** | **Yes** | Context & visualization | Front-facing camera $\ge 224 \times 224$ | Synchronized $800 \times 600$ front RGB |
| **Depth / LiDAR** | **Yes** | 3D backprojection & voxelization | Converted depth pixel-aligned with RGB | Dense metric depth in meters |
| **Semantic Masks** | **Yes** | Voxel class labeling | Road, Vehicle, Pedestrian labels | Canonical CARLA 23-class taxonomy |
| **Ego Vehicle Pose** | **Yes** | Temporal frame alignment | 3D position $(x,y,z)$ + rotation matrix/quaternion | 6-DoF vehicle pose synchronized at 10Hz+ |
| **Camera Intrinsics** | **Yes** | 3D backprojection math | Focal length $(f_x, f_y)$ & principal point $(c_x, c_y)$ | Full $3 \times 3$ intrinsic matrix $K$ |
| **Control Labels** | Optional | Behavioral imitation targets | Steering, throttle, brake per frame | Clean autopilot expert demonstrations |

---

## 2. Coordinate System Standards

The dataset and perception pipeline must adhere strictly to these coordinate definitions:
1. **Camera Frame:** $X$ right, $Y$ down, $Z$ forward (standard optical convention).
2. **Vehicle (Ego) Frame:** $X$ forward, $Y$ right, $Z$ up (ISO vehicle convention).
3. **World Frame:** Fixed Cartesian coordinate system in meters.
4. **Transform Rule:** Any conversion between camera, vehicle, and world must use explicit $4 \times 4$ homogeneous transform matrices handled exclusively in `src/transforms/`.

---

## 3. Dataset Candidate Evaluation

1. **CARLA Autopilot Multimodal Dataset (Selected):**
   - *Source:* Hugging Face (`immanuelpeter/carla-autopilot-multimodal-dataset`).
   - *Size:* 30 autopilot runs, 82,600 frames.
   - *Modalities:* Front RGB ($800 \times 600$), semantic segmentation, 32-channel LiDAR point clouds, 6-DoF ego state, control inputs.
   - *Decision:* Selected as the primary foundation. Required Phase 1 engineering: converting 32-channel sparse LiDAR returns into metric depth maps via forward projection and hole filling.
2. **nuScenes:**
   - Evaluated as benchmark alternative; direct dense depth, but complex coordinate conventions and licensing constraints.
3. **CARLA CIL Dataset:**
   - Evaluated; lacked complete sensor modality synchrony without simulator re-runs.
