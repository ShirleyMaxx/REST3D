# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

# not ideal to put that here
os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
os.environ["LIDRA_SKIP_INIT"] = "true"

from typing import Union, Optional, List, Callable
import numpy as np
from PIL import Image
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate, get_method
import torch
import shutil
import trimesh
import subprocess
from copy import deepcopy
import builtins
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

import sam3d_objects  # REMARK(Pierre) : do not remove this import
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap

__all__ = ["Inference"]

WHITELIST_FILTERS = [
    lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
]

BLACKLIST_FILTERS = [
    lambda target: get_method(target)
    in {
        builtins.exec,
        builtins.eval,
        builtins.__import__,
        os.kill,
        os.system,
        os.putenv,
        os.remove,
        os.removedirs,
        os.rmdir,
        os.fchdir,
        os.setuid,
        os.fork,
        os.forkpty,
        os.killpg,
        os.rename,
        os.renames,
        os.truncate,
        os.replace,
        os.unlink,
        os.fchmod,
        os.fchown,
        os.chmod,
        os.chown,
        os.chroot,
        os.fchdir,
        os.lchown,
        os.getcwd,
        os.chdir,
        shutil.rmtree,
        shutil.move,
        shutil.chown,
        subprocess.Popen,
        builtins.help,
    },
]

_R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T

class Inference:
    # public facing inference API
    # only put publicly exposed arguments here
    def __init__(self, config_file: str, compile: bool = False):
        # load inference pipeline
        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"  # overwrite to disable nvdiffrast
        config.compile_model = compile
        config.workspace_dir = os.path.dirname(config_file)
        check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
        self._pipeline: InferencePipelinePointMap = instantiate(config)

    def merge_mask_to_rgba(self, image, mask):
        mask = mask.astype(np.uint8) * 255
        mask = mask[..., None]
        # embed mask in alpha channel
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
        return rgba_image

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Optional[Union[None, Image.Image, np.ndarray]],
        seed: Optional[int] = None,
        pointmap=None,
    ) -> dict:
        image = self.merge_mask_to_rgba(image, mask)
        return self._pipeline.run(
            image,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=pointmap,
        )


def make_scene_untextured_mesh(*outputs, in_place=False, compute_axis=False):

      if not in_place:
          outputs = [deepcopy(output) for output in outputs]

      all_meshes = []
      for output in outputs:
          mesh = output["glb"]
          if mesh is None:
              continue

          # GLB is Y-up, transforms are Z-up; convert, apply, convert back
          vertices = mesh.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
          R_l2c = quaternion_to_matrix(output["rotation"])

          # Detect upside-down placement: R_l2c[0,2,:] (row 2) is where local +Z
          # (canonical "up") lands in the scene. If Y-component < 0, object is inverted.
          # Fix with a 180° rotation; which axis depends on R_l2c[0,1,2]:
          #   R_l2c[1,2] > 0: Y-axis flip (negate X,Z) — local -Y → scene -Z (faces cam)
          #   R_l2c[1,2] < 0: X-axis flip (negate Y,Z) — local +Y → scene -Z (faces cam)
          # Both are proper rotations (det=+1), so winding order is preserved.
          local_z_in_scene = R_l2c[0, 2, :]
          if local_z_in_scene[1] < 0:
              if R_l2c[0, 1, 2] > 0:
                  vertices[:, 0] *= -1
                  vertices[:, 2] *= -1
              else:
                  vertices[:, 1] *= -1
                  vertices[:, 2] *= -1

          vertices_tensor = torch.from_numpy(vertices).float().to(output["rotation"].device)
          l2c_transform = compose_transform(
              scale=output["scale"],
              rotation=R_l2c,
              translation=output["translation"],
          )
          vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
          mesh.vertices = vertices.squeeze(0).cpu().numpy()
          all_meshes.append(mesh)

      if not all_meshes:
          return None

      return all_meshes


def check_target(
    target: str,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    if any(filt(target) for filt in whitelist_filters):
        if not any(filt(target) for filt in blacklist_filters):
            return
    raise RuntimeError(
        f"target '{target}' is not allowed to be hydra instantiated, if this is a mistake, please do modify the whitelist_filters / blacklist_filters"
    )


def check_hydra_safety(
    config: DictConfig,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    to_check = [config]
    while len(to_check) > 0:
        node = to_check.pop()
        if isinstance(node, DictConfig):
            to_check.extend(list(node.values()))
            if "_target_" in node:
                check_target(node["_target_"], whitelist_filters, blacklist_filters)
        elif isinstance(node, ListConfig):
            to_check.extend(list(node))


def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image


def load_mask(path):
    mask = load_image(path)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    return mask
