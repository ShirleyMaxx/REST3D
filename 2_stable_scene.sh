#!/usr/bin/env bash
# Run to stabilize a 3D scene inferred from a single image (scene_canon):
#  
#
# Output:
#   output/<image_name>/stage3/   (stabilized 3D scene)


INPUT=demo/custom_cartoon_simpson.jpeg
STEM=$(basename "${INPUT%.*}")

PYTHONPATH="$(pwd):${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python scripts/stable_scene.py \
    --scene_dir  "output/${STEM}/stage2" \
    --output_dir "output/${STEM}/stage3"
