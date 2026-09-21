**CARLA Autopilot Multimodal Dataset**

This dataset contains synchronized multimodal driving data collected in the CARLA simulator using the autopilot feature. It provides RGB images from multiple cameras, semantic segmentation, LiDAR point clouds, 2D bounding boxes, and ego-vehicle state/control signals across varied weather, maps, and traffic densities.

The dataset is designed for research in autonomous driving, sensor fusion, imitation learning, and self-driving evaluation.

Dataset Summary
Runs: 30 autopilot runs
Sensors:
RGB cameras: front, front-left, front-right, rear (800×600, fov=90°)
Semantic segmentation: front (raw + colorized)
LiDAR: 32-channel ray-cast, 20 Hz, 80 m range
Collision sensor for impact logs
Annotations: 2D bounding boxes and class labels (vehicles, pedestrians) w.r.t front camera
Ego states: position, rotation, velocity, control (throttle/steer/brake), speed (km/h)
Environment: varied weather, time-of-day (sun altitude), NPC traffic (vehicles + pedestrians)
Splits

Train: 67,000 frames
Validation: 8,400 frames
Test: 7,200 frames
Total size: ~365 GB
Relation to Previous Versions
This dataset, CARLA Autopilot Multimodal Dataset, is an extension of the earlier CARLA Autopilot Image Dataset.

Previous version (carla-autopilot-images):
Contained synchronized RGB camera views (front, front-left, front-right, rear) with ego-vehicle states, controls, and environment metadata.

Current version (carla-autopilot-multimodal-dataset):
Adds new sensor modalities and richer annotations, including:

Semantic segmentation (front view)
LiDAR point clouds
2D bounding boxes and labels (vehicles, pedestrians)
Expanded metadata (collisions, weather difficulty, quality metrics)
In short, v2 augments the original dataset with multimodal signals for perception + sensor fusion research, while retaining full compatibility with the core camera + state data from v1.

Features
Each sample contains:

run_id (string): Identifier for the simulation run
frame (int): Frame number
timestamp (float): Relative timestamp (s)
image_front, image_front_left, image_front_right, image_rear (images): RGB views
seg_front (image): Semantic segmentation (front view)
lidar (list[list[float32]]): LiDAR point cloud (x, y, z, intensity)
boxes (list[list[float32]]): 2D bounding boxes in [xmin, ymin, xmax, ymax] format
box_labels (list[string]): Class labels for bounding boxes
location_{x,y,z} (float): Ego position in world coords
rotation_{pitch,yaw,roll} (float): Ego rotation
velocity_{x,y,z} (float): Ego velocity (m/s)
speed_kmh (float): Ego speed (km/h)
throttle, steer, brake (float): Control inputs
nearby_vehicles_50m, total_npc_vehicles, total_npc_walkers (int): Traffic counts
map_name (string): CARLA map used
weather_* (float): Weather conditions (cloudiness, precipitation, fog, sun altitude)
vehicles_spawned, walkers_spawned (int): Number of NPCs
duration_seconds (int): Total run length in seconds
Example Usage
from datasets import load_dataset

ds = load_dataset("immanuelpeter/carla-autopilot-multimodal-dataset", split="train")
sample = ds[0]

# Access RGB image and LiDAR
front_img = sample["image_front"]
lidar = sample["lidar"]
boxes = sample["boxes"]

Collection Process
Data was collected using a custom CARLA Python script that:

Spawns an ego vehicle with autopilot enabled
Spawns configurable NPC vehicles and pedestrians
Randomizes weather and lighting conditions per run
Synchronizes all sensors and saves every N-th frame
Records vehicle state, control signals, collisions, and environment statistics
All sensors operate in synchronous mode for frame alignment.

Intended Use
Training and benchmarking multimodal self-driving models
Research on sensor fusion, perception, and planning
Imitation learning from autopilot trajectories
Evaluation under diverse weather and traffic conditions
Citation
If you use this dataset, please cite both this work and the CARLA simulator:

@misc{immanuel_peter_2025,
  author       = { Immanuel Peter },
  title        = { CARLA Autopilot Multimodal Dataset },
  year         = 2025,
  url          = { https://huggingface.co/datasets/immanuelpeter/carla-autopilot-multimodal-dataset },
  doi          = { 10.57967/hf/7122 },
  publisher    = { Hugging Face }
}

@inproceedings{Dosovitskiy17,
  title        = {CARLA: An Open Urban Driving Simulator},
  author       = {Alexey Dosovitskiy and German Ros and Felipe Codevilla and Antonio Lopez and Vladlen Koltun},
  booktitle    = {Proceedings of the 1st Annual Conference on Robot Learning},
  pages        = {1--16},
  year         = {2017}
}