"""Scene IO helpers for the stable_scene pipeline."""
import json
import logging
import os

import numpy as np

from rest3d.utils.mesh import load_trimesh_any
from rest3d.utils.quat import local_group_child_offset_parent_frame, quat_to_rotmat

_logger = logging.getLogger("stable_scene_opt")


def load_local_group_from_dir(layout, local_group_dir):
    _logger.info(f"\n[local_group load] Bypassing local_group, loading from: {local_group_dir}")
    group_child_offsets = {}

    for g_idx, group in enumerate(layout.groups):
        root_name   = group[0]
        child_names = group[1:]

        if len(child_names) == 0:
            group_child_offsets[g_idx] = {}
            _logger.info(f"  [G{g_idx}] {root_name}: no children, skip")
            continue

        json_path = os.path.join(local_group_dir, f"local_group_{root_name}.json")
        if not os.path.isfile(json_path):
            json_path = os.path.join(local_group_dir, f"best_poses_{root_name}.json")  # legacy fallback
        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"[local_group load] No pose file found for group {g_idx} ({root_name}) in {local_group_dir}. "
                f"Expected local_group_{root_name}.json")

        _logger.info(f"  [G{g_idx}] {root_name}: loading from {os.path.basename(json_path)}")
        with open(json_path, "r") as f:
            data = json.load(f)
        objects = data["objects"]

        _default_ref = [0., layout.base_dy, 0.]
        root_pos_json  = np.array(
            objects.get(root_name, {"pos": _default_ref})["pos"], dtype=np.float32)
        root_obj       = objects.get(root_name, {})
        root_quat_json = np.array(
            root_obj.get("rot", [0., 0., 0., 1.]), dtype=np.float32)

        offsets = {}
        for c in child_names:
            if c not in objects:
                _logger.warning(f"  [G{g_idx}] child '{c}' not found in {json_path}, using zero offset")
                offsets[c] = (np.zeros(3, dtype=np.float32),
                              np.array([0., 0., 0., 1.], dtype=np.float32))
                continue
            child_pos  = np.array(objects[c]["pos"],  dtype=np.float32)
            child_quat = np.array(objects[c]["rot"],  dtype=np.float32)
            pos_ofs, q_rel = local_group_child_offset_parent_frame(
                root_quat_json, root_pos_json, child_pos, child_quat)
            offsets[c] = (pos_ofs, q_rel)

        group_child_offsets[g_idx] = offsets
        _logger.info(f"  [G{g_idx}] loaded offsets for {len(offsets)} children")

    return group_child_offsets


def load_global_from_dir(args, layout, group_child_offsets, global_dir):
    from rest3d.models.cem_opt import place_all_descendants  # avoid circular at module level

    p2_json    = os.path.join(global_dir, "global_best_poses.json")
    _gs_dir    = os.path.join(args.output_dir, "global_scene")
    output_dir = os.path.join(_gs_dir, "obj_files")
    os.makedirs(output_dir, exist_ok=True)

    _logger.info(f"\n[global load] Bypassing global, loading from: {global_dir}")
    with open(p2_json, "r") as f:
        p2_data = json.load(f)
    entity_poses = p2_data["objects"]

    all_child_names    = {n for g in layout.groups for n in g[1:]}
    entity_names       = [n for n in entity_poses if n not in all_child_names]
    group_root_to_gidx = {g[0]: g_idx for g_idx, g in enumerate(layout.groups)}
    _logger.info(f"  [global load] entity roots ({len(entity_names)}): {entity_names}")

    poses = {}
    for name in entity_names:
        entry = entity_poses[name]
        pos  = np.array(entry["pos"],  dtype=np.float32)
        quat = np.array(entry["rot"],  dtype=np.float32)
        poses[name] = {"pos": pos.tolist(), "rot": quat.tolist(), "lin_vel": [0., 0., 0.]}
        place_all_descendants(name, pos, quat, group_child_offsets, group_root_to_gidx, poses)

    out_json = os.path.join(_gs_dir, "global_scene_poses.json")
    with open(out_json, "w") as f:
        json.dump({"base_dy": layout.base_dy, "objects": poses}, f, indent=2)
    _logger.info(f"  [global load] -> {out_json}  ({len(poses)} objects)")

    all_entity_set = set(entity_names)
    n_exported = 0
    for c_name, c_entry in poses.items():
        if c_name in all_entity_set:
            continue
        c_pos  = np.array(c_entry["pos"],  dtype=np.float32)
        c_quat = np.array(c_entry["rot"],  dtype=np.float32)
        obj_path = os.path.join(layout.obj_dir, f"{layout.file_prefix}{c_name}.obj")
        if not os.path.isfile(obj_path):
            _logger.warning(f"  [global load] child OBJ not found: {obj_path}")
            continue
        mesh = load_trimesh_any(obj_path)
        T = np.eye(4)
        T[:3, :3] = quat_to_rotmat(c_quat)
        T[:3, 3]  = c_pos
        mesh.apply_transform(T)
        out_path = os.path.join(output_dir, f"{c_name}.obj")
        mesh.export(out_path)
        n_exported += 1

    _logger.info(f"[global load] Done: {len(poses)} objects in JSON, "
                 f"{n_exported} child OBJs exported -> {output_dir}")
