# Temporal 3D Semantic Occupancy Grid

[![CI](https://github.com/donaldng05/carla/actions/workflows/ci.yml/badge.svg)](https://github.com/donaldng05/carla/actions)
[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Type Checked: mypy](https://img.shields.io/badge/type_checked-mypy-blue)](https://mypy-lang.org/)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-76%20passed%20%7C%2085%25%20cov-brightgreen)](https://github.com/donaldng05/carla)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/donaldng05/carla/blob/main/notebooks/phase2_5_scenario_memory_eval.ipynb)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

A perception-centric autonomous driving system engineered to maintain spatial awareness under **severe visual occlusion** and **sparse sensor regimes** in dense traffic.

Instead of relying on fragile 3D bounding-box tracking that flickers when obstacles block line of sight, this system constructs a **3D semantic occupancy grid** from sparse LiDAR-projected depth and camera semantic segmentation, then executes **5-frame temporal ego-motion fusion** to retain persistent spatial memory across vehicle motion.

---

## Visual Demonstration: Resolving Occlusion & Sensor Sparsity

![BEV Occlusion Recovery](docs/assets/bev_occlusion_recovery.png)

*Figure 1: Bird's-Eye View (BEV) evaluation on CARLA validation split (Frame 475).*
- **Left (Baseline $t$):** Single-frame observation exhibits severe vehicle hull dropouts caused by line-of-sight occlusion and 32-channel LiDAR beam sparsity.
- **Center (5-Frame Temporal Fusion):** 6-DoF ego-motion alignment across rolling history ($t-4..t$) reconstructs occluded vehicle geometry without introducing spatial drift.
- **Right (Future Pseudo-Label):** Unbiased supervision target constructed by unioning future observations ($t..t+4$) back into the anchor frame, confirming reconstructed voxels match true vehicle geometry.

---

## 📋 Executive / Recruiter Summary

- **3D Spatial Discretization:** Engineered a $200 \times 200 \times 20$ voxel grid ($50\text{m} \times 50\text{m} \times 5\text{m}$ at $0.25\text{m}$ resolution) backprojecting multi-modal RGB, semantic masks, and LiDAR returns into the vehicle ISO coordinate frame.
- **Temporal Ego-Motion Fusion:** Formulated $4 \times 4$ homogeneous SE(3) pose transformations ($T_{\text{target}\leftarrow\text{source}}$) across a 5-frame rolling buffer with strict decay weighting to eliminate occlusion dropouts.
- **Empirical Validation (2,000-Frame Benchmark):** Designed a scenario-stratified evaluation benchmark demonstrating a **+7.95% absolute vehicle IoU improvement** under occlusion and high modality disagreement, and **+1.03% IoU** under extreme LiDAR sparsity.
- **Engineering Rigor:** Built with modern Python 3.11 tooling (`uv`, `ruff`, strict `mypy`), backed by 76 tests (85% branch coverage), including Hypothesis property-based testing of geometric invariants.

---

## Why 3D Occupancy Grids? (Paradigm Comparison)

| Perception Paradigm | Failure Mode in Heavy Traffic | How This System Solves It |
| :--- | :--- | :--- |
| **3D Bounding Boxes** | Bounding box flicker, aspect-ratio jitter, and identity switching when vehicles occlude one another | Continuous voxelization models free, occupied, and occluded space without requiring discrete instance separation |
| **Single-Frame Perception** | Sensor sparsity or momentary occlusions instantly erase critical obstacles from the planner's world model | 5-Frame temporal fusion maintains persistent geometric memory across vehicle motion |
| **Standard Dataset Benchmarks** | Circular evaluation against current-frame ground truth penalizes memory when current frame has occlusion | Multi-frame future pseudo-labels ($t..t+4$) provide unbiased geometric targets |

---

## Quantitative Results (2,000-Frame Validation Benchmark)

Evaluation over 2,000 continuous frames and 1,992 candidate scenes from `immanuelpeter/carla-autopilot-multimodal-dataset`. Evaluated across 50 selected anchor scenes per condition slice using strict decay fusion (`weights=[1.0, 0.5, 0.25, 0.1, 0.05]`, `threshold=0.40`):

| Scenario Slice | Anchor Count | Class | Baseline IoU | 5-Frame Temporal IoU | Absolute Gain ($\Delta$) | True Positive Intersection ($T$ vs $B$) | Primary Finding |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **High Disagreement** | **50** | **Vehicle** | **0.6039** | **0.6833** | **+0.0795 (+7.95%)** | **1,163 vs 1,003** | **Significant Occlusion Recovery (+16% voxels)** |
| | | **Road** | 0.2032 | 0.2082 | +0.0051 (+0.51%) | 1,304 vs 1,262 | Consistent road alignment |
| **Sparse Depth** | **50** | **Vehicle** | **0.2816** | **0.2919** | **+0.0103 (+1.03%)** | **622 vs 600** | **LiDAR sparsity mitigation on hull** |
| | | Road | 0.1991 | 0.1991 | -0.0000 | 5,559 vs 5,557 | Stable ground plane |
| **Vehicle Presence Change** | 50 | Vehicle | 0.2162 | 0.2171 | +0.0008 (+0.08%) | 1,912 vs 1,904 | Stable across dynamic scene entry/exit |
| | | Road | 0.2100 | 0.2100 | -0.0000 | 7,095 vs 7,095 | Zero geometric drift |
| **Dense Traffic** | 50 | Vehicle | 0.0877 | 0.0877 | ±0.0000 | 20 vs 20 | High-density baseline consistency |
| | | Road | 0.2022 | 0.2022 | -0.0000 | 8,108 vs 8,108 | Consistent baseline |

> **Key Empirical Takeaway:** Uniform-stream evaluation over open, unoccluded highways yields neutral gains (as expected when single-frame observations already capture complete geometry). Scenario-stratified evaluation proves temporal accumulation acts as an effective occlusion-resilient perception memory specifically where single-frame perception fails.

---

## Core Perception Architecture

```
Raw 32-Channel LiDAR ──► Pinhole Camera Projection ──► Dense Front Depth Map
                                                               │
                                                               ▼
Front Semantic Mask   ────────────────────────────────► 3D Backprojection (K, Extrinsics)
                                                               │
                                                               ▼
                                                  200x200x20 Voxel Grid (0.25m)
                                                               │
                                                               ▼
6-DOF Ego Pose Delta  ────────────────────────────────► 5-Frame Temporal Ego-Fusion
                                                               │
                                                               ▼
                                                  Bird's-Eye View (BEV) + Shadow Mode
```

1. **LiDAR-to-Depth Projection:** Projects 32-channel LiDAR point clouds onto the front camera image plane using camera intrinsics ($K$) and extrinsics, resolving ray collisions by nearest-depth arbitration.
2. **3D Backprojection & Voxelization:** Backprojects valid depth and front semantic labels into vehicle coordinates, discretizing space into a $200 \times 200 \times 20$ voxel grid ($50\text{m} \times 50\text{m} \times 5\text{m}$ at $0.25\text{m}$ resolution).
3. **Temporal Ego-Motion Fusion:** Aligns the rolling 5-frame occupancy history into the current vehicle frame using $4 \times 4$ homogeneous SE(3) transforms:
   $$T_{\text{target}\leftarrow\text{source}} = T_{\text{target}}^{-1} T_{\text{source}}$$
   Accumulates weighted evidence across time with strict decay weights to eliminate occlusion dropouts without boundary smearing.
4. **Shadow-Mode Evaluation:** Compares baseline single-frame perception against temporal fusion in parallel, quantifying voxel disagreement and clustering failure modes across weather, lighting, and traffic density.

---

## Key Engineering Decisions & Trade-Offs

- **Voxel Discretization Resolution ($0.25\text{m}$):** Chosen to preserve vehicle corner contours and pedestrian presence while capping memory footprint at $\sim 800\text{KB}$ per voxel grid tensor ($200 \times 200 \times 20 \times 1\text{ byte}$), enabling real-time rolling-buffer caching on CPU.
- **Coordinate System Single Source of Truth (`src/transforms/`):** Coordinate frame divergence (Unreal Engine left-handed system vs ISO vehicle frame vs optical camera frame) is the most frequent silent failure mode in autonomous driving perception. All spatial math is isolated in `src/transforms/` and protected by **Hypothesis property-based tests** verifying mathematical invariants (invertibility $T^{-1}T = I$, distance preservation under rigid rotation).
- **Static World vs Dynamic Actor Assumption:** Ego-motion compensation assumes a static world frame. For high-speed dynamic obstacles, 5-frame accumulation can introduce slight trailing edges. In downstream integration, static background accumulation is decoupled from dynamic actors via instance-level motion compensation.
- **Defensive Evaluation Boundaries:** Rather than claiming global superiority, the evaluation pipeline explicitly isolates scenarios by environmental attributes, providing transparent bounds on where temporal memory delivers measurable engineering value.

---

## Repository Structure

```
.
├── .github/          # CI/CD workflows (Python 3.10/3.11 matrix, Ruff, Mypy, Pytest)
├── .agents/          # Autonomous agent governance instructions & protocols
├── configs/          # Provisional CARLA sensor rig extrinsics & camera intrinsics
├── docs/             # Technical specifications, phase closeouts, and contracts
│   └── assets/       # Curated visualization figures for documentation
├── notebooks/        # Exploratory verification and BEV visualization notebooks
├── outputs/          # Serialized benchmark metrics, JSON summaries, and BEV PNGs
│   └── phase2/
│       └── scenario_memory/
├── src/
│   ├── transforms/   # Mandatory single source of truth for 3D coordinate math
│   ├── perception/   # LiDAR conversion, occupancy grid, and 5-frame temporal fusion
│   ├── data/         # Streamed CARLA multimodal dataset loading & split auditing
│   └── evaluation/   # Class IoU metrics & shadow-mode disagreement evaluator
├── tests/            # Automated test suites (Pytest + Hypothesis property tests)
├── Makefile          # Standard developer command shortcuts
├── pyproject.toml    # Dependency declarations and tool configurations
└── uv.lock           # Deterministic cross-platform lockfile
```

---

## Quickstart

### 1. Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic package and environment management:

```bash
# Clone the repository
git clone https://github.com/donaldng05/carla.git
cd carla

# Install dependencies and create virtual environment
uv sync --all-extras
# or with make:
make install
```

### 2. Quality & Verification Checks

Run static analysis, type checking, and automated test suite:

```bash
# Lint and format checks
uv run ruff check src tests
uv run ruff format --check src tests

# Static type checking
uv run mypy src tests

# Test suite execution with coverage (76 tests, 85% coverage)
uv run pytest --cov=src --cov-report=term-missing

# Or run all verification steps with one command:
make check
```

### 3. Interactive Exploration

To explore the occupancy grid generation and visual comparisons interactively:
- Open [notebooks/phase2_5_scenario_memory_eval.ipynb](notebooks/phase2_5_scenario_memory_eval.ipynb) locally or via Google Colab.
- Pre-computed benchmark summaries and BEV images are available in `outputs/phase2/scenario_memory/`.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
