#!/usr/bin/env bash
# Replay optimized scene in Isaac Gym.
#   - Loads stable scene from output/<image_name>/stage3/global_scene or global_scene_w_walls
#   - Settles objects under gravity and records an MP4 （output/<image_name>/stage3/global_scene/replay*.mp4）
#   - Optionally opens a viser 3D browser viewer (--viser)
#   - Note that without --viser, Isaac Gym does not display object textures by default


INPUT=demo/custom_cartoon_simpson.jpeg
STEM=$(basename "${INPUT%.*}")

PYTHONPATH="$(pwd):${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python scripts/replay_in_simulator.py \
    --scene_tree "output/${STEM}/stage2/scene_tree.json" \
    --output_dir "output/${STEM}/stage3/global_scene" \
    --settle_steps 120 --viser

