# Temporal 3D Semantic Occupancy Grid

A perception-centric autonomous driving system designed to maintain spatial awareness under **severe visual occlusion** in dense traffic.

Instead of relying on fragile bounding-box tracking that flickers when obstacles block one another, this system constructs a **3D semantic occupancy grid** from sparse LiDAR-projected depth and semantic segmentation, then applies **5-frame temporal ego-motion fusion** to retain persistent spatial memory across time.

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

1. **LiDAR-to-Depth Projection:** Projects 32-channel LiDAR point clouds onto the front camera image plane, resolving nearest-depth ties and preserving unobserved pixels as `NaN`.
2. **3D Backprojection & Voxelization:** Backprojects valid depth and front semantic labels into vehicle coordinates, discretizing space into a $200 \times 200 \times 20$ voxel grid ($50\text{m} \times 50\text{m} \times 5\text{m}$ at $0.25\text{m}$ resolution).
3. **Temporal Ego-Motion Fusion:** Aligns the last 5 occupancy frames into the current vehicle frame using $4 \times 4$ homogeneous transforms, accumulating weighted evidence to eliminate occlusion dropouts.
4. **Shadow-Mode Evaluation:** Compares baseline single-frame perception against temporal fusion, quantifying voxel disagreement and clustering failure modes across weather, lighting, and traffic density.

---

## Repository Structure

```
.
├── configs/          # Provisional CARLA sensor rig extrinsics & camera intrinsics
├── notebooks/        # Exploratory verification and BEV visualization notebooks
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

# Test suite execution with coverage
uv run pytest --cov=src --cov-report=term-missing
```

Or run all verification steps with one command:
```bash
make check
```
