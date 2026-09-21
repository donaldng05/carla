# CARLA Perception Project Plan (v4)

> **Project Identity:** 3D Semantic Occupancy Grid with Temporal Ego-Motion Fusion and Scenario-Stratified Memory Evaluation for Occlusion-Resilient Autonomous Vehicle Perception.

---

## 1. Project Directory Structure

```text
carla/
├── src/
│   ├── transforms/             # Single source of truth for all coordinate math
│   ├── perception/             # LiDAR-to-depth conversion, 3D occupancy, temporal fusion
│   ├── data/                   # Hugging Face dataset streaming & split audit
│   └── evaluation/             # IoU metrics, shadow mode, scenario memory evaluation
├── tests/                      # Unit, property, and integration test suites
│   ├── test_transforms.py
│   ├── test_occupancy.py
│   ├── test_temporal_fusion.py
│   ├── test_lidar_conversion.py
│   ├── test_phase2_carla_eval.py
│   └── test_split_audit.py
├── notebooks/                  # Visualization and post-hoc inspection only
│   ├── phase2_bounded_iou_evaluation.ipynb
│   └── phase2_5_scenario_memory_eval.ipynb
├── outputs/                    # Structured evaluation artifacts (BEV images, summaries, metrics)
│   └── phase2/
│       └── scenario_memory/
├── configs/                    # YAML configuration files
└── docs/                       # Project constitution, contracts, and closeout records
```

---

## 2. Tier 2 Deviation: LiDAR-to-Depth Conversion

* **Date:** 2025-06-10
* **Status:** Implemented & Validated
* **Rationale:** The selected dataset (`immanuelpeter/carla-autopilot-multimodal-dataset`) provides RGB, semantic segmentation, 6-DoF ego state, and 32-channel LiDAR point clouds rather than dense simulator depth maps.
* **Resolution:** Rather than switching datasets, we engineered a dedicated `src/perception/lidar_conversion.py` pipeline:
  1. Project 3D LiDAR points into the front camera optical frame using camera intrinsics ($K$) and extrinsics.
  2. Compute depth per pixel (Z coordinate in camera frame), keeping the nearest point on ray collisions.
  3. Apply morphological closing / hole-filling to mitigate sensor sparsity.
  4. Explicitly model sensor sparsity as a primary motivation for multi-frame temporal fusion.

---

## 3. Phase Roadmap (Perception Core)

### Phase 1: Dataset Pipeline & Modality Verification
* **Objective:** Establish reliable streaming, validate sensor synchrony, implement LiDAR-to-depth conversion, and ensure leakage-free data splits.
* **Key Tasks:**
  - **Hugging Face Streaming Integration:** Implement zero-download streaming loaders in `src/data/carla_dataset.py`.
  - **Split Integrity Audit:** Build `src/data/split_audit.py` to mathematically verify that recording run IDs do not leak across `train`, `validation`, and `test` splits.
  - **Modality Verification:** Verify per-frame alignment across RGB, projected depth, semantic masks, and ego poses.
  - **LiDAR-to-Depth Module:** Build and unit test `src/perception/lidar_conversion.py` with invariant tests (nearest-point arbitration, boundary clipping).
* **Milestone:** Validated data quadruplets (RGB, depth, segmentation, pose) streamable without local storage overhead; split integrity verified with zero run-ID overlap.

---

### Phase 2: 3D Semantic Occupancy & Ego-Motion Temporal Fusion
* **Objective:** Construct 3D voxel grids from sensor inputs, align historical observations into the current vehicle frame via ego-pose deltas, and accumulate occupancy evidence over time.
* **Key Tasks:**
  - **3D Voxel Backprojection:** Project metric depth pixels into 3D camera space, convert to ego frame via `src/transforms/`, and discretize into a $200 \times 200 \times 20$ grid ($50\text{m} \times 50\text{m} \times 5\text{m}$ at $0.25\text{m}$ resolution).
  - **Semantic Class Assignment:** Assign canonical CARLA semantic IDs (road, vehicle, pedestrian) to occupied voxels.
  - **Ego-Motion Temporal Fusion:** Maintain a rolling 5-frame past buffer. Transform historical grids into the current ego pose ($T_{\text{target}\leftarrow\text{source}} = T_{\text{target}}^{-1} T_{\text{source}}$) and accumulate voxel occupancy using strict decay weights `[1.0, 0.5, 0.25, 0.1, 0.05]`.
  - **Shadow-Mode Evaluation:** Run single-frame baseline and 5-frame temporal models in parallel. Log per-voxel disagreement and cluster divergence across environmental conditions.
* **Milestone:** Production-grade occupancy grid engine capable of temporal feature alignment; shadow-mode pipeline logging disagreement hotspots.

---

### Phase 2.5: Scenario-Stratified Memory Evaluation (Capstone)
* **Objective:** Move beyond naive uniform-stream evaluation to prove that temporal fusion provides statistically significant gains where it matters most (occlusion, high modality disagreement, and sparse depth).
* **Key Tasks:**
  - **Multi-Frame Future Pseudo-Labels:** Form high-quality supervision targets by unioning future frames ($t..t+4$) into the anchor vehicle frame, avoiding circular single-frame comparisons.
  - **Scenario Memory Candidate Scoring:** Scan continuous validation sequences (2,000 frames) and score 1,992 candidates across 5 stratified condition slices:
    1. `high_disagreement`: Frames where baseline and temporal models diverge ($\ge 5\%$ voxel disagreement).
    2. `sparse_depth`: Frames with severe LiDAR sparsity (bottom 25% quantile).
    3. `vehicle_change`: Scenes with dynamic vehicle presence and rapid occupancy transitions.
    4. `dense_traffic`: High vehicle density environments ($\ge 10$ nearby vehicles).
    5. `turning`: High lateral acceleration and yaw rate changes.
  - **Quantitative IoU Validation:** Measure per-class intersection-over-union across 50 selected anchors per slice.
* **Verified Capstone Results:**
  - **High Disagreement Slice:** Vehicle IoU increased from **`0.6039` to `0.6833` (+7.95% absolute gain)**; road IoU gained **+0.51%**.
  - **Sparse Depth Slice:** Vehicle IoU increased from **`0.2816` to `0.2919` (+1.03% gain)**.
  - **Vehicle Change Slice:** Temporal IoU matched or slightly exceeded single-frame baseline (+0.08%).
* **Milestone:** Rigorous quantitative proof conforming to Constitutional Rule 8 ("Metrics Before Claims") that temporal fusion resolves occlusions and sensor sparsity.

---

## 4. Downstream Systems (Future Scope)

While the repository's active deliverable is a self-contained 3D AV perception stack, the architecture is designed to integrate seamlessly with downstream consumers:
- **Phase 3 (Planning):** Feeding temporally stabilized BEV representations or camera contexts into behavioral cloning policy heads.
- **Phase 4 (Control & Safety):** Ingesting 3D occupancy grids into high-frequency (50Hz) C++ safety monitors for deterministic forward-cone collision overrides.
