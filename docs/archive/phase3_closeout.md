# Phase 3 Closeout (Archived)

> [!NOTE]
> **Archival Notice:** This document is archived. Phase 3 behavioral cloning was explored and implemented on the isolated branch `codex/phase3-behavioral-cloning`. The active `main` branch baseline is Phase 2.5 (3D Semantic Occupancy Grid & Scenario-Stratified Temporal Fusion).

Phase 3 behavioral-cloning infrastructure is implemented on the isolated
`codex/phase3-behavioral-cloning` branch.

## Implemented

- Run-safe temporal windows with a four-frame past context.
- Explicit BC target ordering: `[steering, throttle]`.
- Camera-only ResNet-18 plus Transformer model contract.
- Training-only horizontal shift augmentation with steering correction.
- Configurable steering-weighted loss and hard-control proxy weighting.
- Resumable epoch checkpoints containing model, optimizer, scheduler, config,
  and validation state.
- Offline action metrics and a testable training epoch function.
- `configs/bc_v1.yaml` experiment defaults.

## Validation

- Existing pytest suite plus Phase 3 tests pass.
- Mypy passes for `src` and `tests`.
- Two-epoch golden-data training smoke test passes, including checkpoint creation.
- The local environment does not include `torchvision`, so the ResNet model shape
  test is skipped locally; `torchvision` is declared in `requirements.txt`.

## Limitations

- Full Hugging Face streaming orchestration remains notebook/integration-specific;
  `train_model` accepts injected loaders to keep the core training path testable.
- The recovery weighting signal is a hard-control proxy based on steering magnitude,
  not direct lane-edge deviation because that label is not present in the dataset schema.
- Phase 2 occupancy evidence remains a projected-depth pseudo-label diagnostic and
  does not establish dense-ground-truth temporal-fusion improvement.
