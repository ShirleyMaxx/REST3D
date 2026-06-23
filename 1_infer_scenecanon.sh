#!/usr/bin/env bash
# Run to infer a 3D scene from a single image:
#  
# Before running, export your VLM API key in the shell:
#   export GEMINI_API_KEY=...
#
# Output:
#   output/<image_name>/stage1/   (scene tree, masks)
#   output/<image_name>/stage2/   (3D scene: scene_canon)

INPUT=demo/custom_cartoon_simpson.jpeg

CUDA_VISIBLE_DEVICES=0 python scripts/infer_scenetree.py \
    --image_folder "$INPUT" \
    || { printf "\033[31mERROR: stage1 failed; stage2 will NOT be run.\033[0m\n" >&2; exit 1; }

CUDA_VISIBLE_DEVICES=0 python scripts/infer_scene3d.py \
    --image_folder "$INPUT" \
    || { printf "\033[31mERROR: stage2 failed.\033[0m\n" >&2; exit 1; }
