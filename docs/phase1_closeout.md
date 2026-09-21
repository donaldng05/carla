# Phase 1 Closeout

Phase 1 is accepted as functionally complete for the path needed before Phase 2.

The project uses the CARLA Autopilot Multimodal dataset from Hugging Face as a streamed
source. The local `data/` directory intentionally does not contain a full dataset download.
The modality verification notebook streams bounded samples directly from Hugging Face and
serves as the visual validation artifact for Phase 1.

## Completed

- Dataset choice: CARLA Autopilot Multimodal from Hugging Face.
- LiDAR-to-depth conversion: implemented in `src/perception/lidar_conversion.py`.
- Calibration policy: front-camera intrinsics derived from dataset-card image size and FOV;
  LiDAR-to-camera extrinsics use the documented provisional CARLA rig in
  `configs/phase1_provisional_carla_rig.json`.
- Modality verification: performed in `notebooks/phase1_modality_verification.ipynb` using
  streamed samples, RGB/LiDAR overlay inspection, depth coverage summaries, and a bounded
  multi-run scan.
- Data loading: implemented in `src/data/carla_dataset.py` with aligned RGB, converted depth,
  segmentation, control, LiDAR, and metadata outputs.
- Split audit: implemented in `src/data/split_audit.py`; the lightweight datasets-server audit
  found no observed train/validation/test run overlap in the sampled windows.

## Accepted Limitations

- No local golden dataset is committed for Phase 1 because the project avoids downloading the
  large CARLA dataset locally. Streamed notebook verification is the accepted Phase 1 artifact.
- Split verification is a bounded evidence audit, not an exhaustive row-by-row proof. This is
  accepted for Phase 1 because the dataset is treated as a trusted upstream source and the
  observed split windows show distinct run IDs.
- Condition balance is trusted from the dataset provenance and split design. Exhaustive
  condition-distribution reporting can be added later if evaluation results require it.
- Converted depth is sparse because it is projected from 32-channel LiDAR. Unknown pixels remain
  `NaN` unless an explicit preprocessing profile enables hole filling.

## Phase 2 Readiness

Phase 2 may proceed using the existing DataLoader output schema:

- `rgb`
- `depth`
- `segmentation`
- `control`
- `lidar`
- `metadata`

Occupancy-grid work should continue to treat `NaN` depth as unobserved space and should document
any artifacts caused by sparse LiDAR coverage.
