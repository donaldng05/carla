# Phase 1 Calibration Assumptions

This note records the calibration policy currently used for Phase 1 modality verification on the CARLA Autopilot Multimodal dataset.

## What comes from the dataset card

- Front RGB camera resolution: `800 x 600`
- Front RGB camera field of view: `90 degrees`
- Bounding boxes are defined with respect to the front camera
- Sensors are recorded in synchronous mode

From the front-camera spec, we derive the pinhole intrinsics used for validation:

```text
fx = fy = 800 / (2 * tan(90 / 2)) = 400
cx = 400
cy = 300

K = [[400,   0, 400],
     [  0, 400, 300],
     [  0,   0,   1]]
```

These intrinsics are treated as `dataset-derived`.

## What is still an assumption

The dataset card and streamed samples do not publish exact LiDAR-to-front-camera extrinsics.

For Phase 1 validation only, we use a provisional CARLA-style rig stored in [phase1_provisional_carla_rig.json](/D:/carla/configs/phase1_provisional_carla_rig.json):

- Front camera CARLA pose: `(x=1.5, y=0.0, z=2.4, pitch=0, yaw=0, roll=0)`
- LiDAR CARLA pose: `(x=0.0, y=0.0, z=2.4, pitch=0, yaw=0, roll=0)`

The checked-in `lidar_to_camera_extrinsics` matrix is not just a raw translation. It is already expressed in the pinhole camera frame expected by the current LiDAR projection code, where:

- `Z` is forward depth
- `X` is image-right
- `Y` is image-down

This extrinsics artifact is treated as `assumed rig`.

## Acceptance rule

Downstream occupancy or preprocessing work may proceed only after the provisional rig passes visual sanity checks on real dataset frames:

- projected LiDAR lands on plausible RGB structure
- no obvious global flip or severe shift is present
- depth coverage and sparsity look consistent with a 32-channel LiDAR

If those checks fail, the rig assumption must be revised before the DataLoader or occupancy phases continue.
