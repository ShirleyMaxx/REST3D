"""rest3d — single-image scene reconstruction (SAM 3 + SAM-3D-Objects + VLM).

Top-level package. Re-exports the main building blocks so users can write::

    from rest3d.models.build_sam3dobj import Inference, make_scene_untextured_mesh
    from rest3d.utils.vlm import set_vlm_backend, analyze_scene_object_lists
    from rest3d.utils.mesh import y_align
"""
import os

__version__ = "0.1.0"

# Absolute path to the bundled prompt directory. Used by utils.vlm as the
# default lookup location for VLM prompt templates.
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

__all__ = ["PROMPTS_DIR", "__version__"]
