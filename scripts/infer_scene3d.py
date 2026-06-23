import os, sys
import json
import copy
import traceback
from datetime import datetime
import numpy as np
import argparse
from glob import glob
import trimesh
import shutil
from collections import defaultdict, deque

from rest3d.models.build_sam3dobj import (
    Inference,
    load_image,
    load_mask,
    make_scene_untextured_mesh,
)
from rest3d.utils.mesh import (
    y_align,
    get_object_bboxes_from_outputs,
    compute_rigid_transform,
    save_obj_with_rigid_transform,
    read_obj_vertices,
    get_y_bounds,
    write_obj_with_y_offset,
)
from rest3d.utils.vis import vis_axes_ply, vis_bboxes_ply
from rest3d.utils.urdf import generate_urdf_files
from rest3d.utils.log import get_logger, attach_file_handler

logger = get_logger("stage2")


def _excepthook(exc_type, exc_value, exc_tb):
    """Route uncaught exceptions through the logger so the per-image FileHandler
    captures the traceback into stage2_log_*.txt before the script exits."""
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.error("Uncaught exception:\n" + tb_text)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _excepthook

logger.info("Running stage2: scene canonicalization (loading SAM 3D model)...")

SAM3D_OBJECTS_ROOT = os.environ.get("SAM3D_OBJECTS_ROOT")
if not SAM3D_OBJECTS_ROOT:
    raise EnvironmentError(
        "SAM3D_OBJECTS_ROOT is not set. Export it to your local sam-3d-objects "
        "checkout, e.g. `export SAM3D_OBJECTS_ROOT=/path/to/sam-3d-objects`."
    )
TAG = os.environ.get("SAM3D_OBJECTS_TAG", "hf")
config_path = os.path.join(SAM3D_OBJECTS_ROOT, "checkpoints", TAG, "pipeline.yaml")
if not os.path.isfile(config_path):
    raise FileNotFoundError(
        f"SAM-3D-Objects pipeline config not found at {config_path}. "
        f"Point SAM3D_OBJECTS_ROOT at your sam-3d-objects checkout and download the "
        f"checkpoints under checkpoints/{TAG}/ — see "
        f"https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md#2-getting-checkpoints"
    )
inference = Inference(config_path, compile=False)


def inference_mesh_with_cache(image, masks_list, object_names, output_obj_dir, output_bbox_axes_dir, output_vis_dir, args):
    """
    Run SAM3D inference with caching support.

    Returns:
        dict: ori_results containing meshes, bboxes, axes, meshes_ref, axes_ref
    """
    use_cache = args.use_cache and not args.force_reinfer
    cache_exists = all(
        (os.path.exists(os.path.join(output_obj_dir, f"{obj_name}.obj")) and
         os.path.exists(os.path.join(output_bbox_axes_dir, f"{obj_name}_bbox_axes.npz")))
        for obj_name in object_names
    )

    ori_results = defaultdict(list)

    if use_cache and cache_exists:
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Found cached raw meshes, bboxes, and axes, loading from disk...")
        logger.info(f"{'='*60}\n")

        # Load cached original meshes, bboxes, and axes
        for idx, obj_name in enumerate(object_names):
            # Load mesh
            cache_mesh_path = os.path.join(output_obj_dir, f"{obj_name}.obj")
            ori_results['meshes'].append(trimesh.load(cache_mesh_path, force='mesh'))

            # Load cached bbox and axes data from NPZ
            cache_npz_path = os.path.join(output_bbox_axes_dir, f"{obj_name}_bbox_axes.npz")
            bbox_axes_data = np.load(cache_npz_path)
            ori_results['bboxes'].append({
                'corners': bbox_axes_data['bbox_corners'],
                'center': bbox_axes_data['bbox_center']
            })
            ori_results['axes'].append(bbox_axes_data['axes'])

            # Identify reference meshes (floor, carpet)
            if "floor" in obj_name or "carpet" in obj_name:
                ori_results['meshes_ref'].append(ori_results['meshes'][-1])
                ori_results['axes_ref'].append(ori_results['axes'][-1])

        logger.info(f"Loaded {len(ori_results['meshes'])} cached meshes ({len(ori_results['meshes_ref'])} reference)")

    else:
        if args.force_reinfer:
            logger.info(f"\n{'='*60}")
            logger.info(f"Force re-inference enabled, running SAM3D...")
            logger.info(f"{'='*60}\n")
        else:
            logger.info(f"\n{'='*60}")
            logger.info(f"No cache found, running SAM3D inference...")
            logger.info(f"{'='*60}\n")

        # Run SAM3D inference
        outputs = []
        for mask_path in masks_list:
            logger.info(f'------>    processing mask: {mask_path}')
            mask = load_mask(mask_path)
            output = inference(image, mask, seed=42)
            outputs.append(output.copy())

        # Save raw pre-transform GLB meshes for debugging
        if args.debug_raw_mesh:
            raw_mesh_dir = os.path.join(os.path.dirname(os.path.dirname(output_obj_dir)), "debug_raw_glb")
            os.makedirs(raw_mesh_dir, exist_ok=True)
            for obj_name, output in zip(object_names, outputs):
                glb = output.get("glb")
                if glb is not None:
                    raw_path = os.path.join(raw_mesh_dir, f"{obj_name}_raw.obj")
                    glb_copy = copy.deepcopy(glb)
                    glb_copy.export(raw_path)
            logger.info(f"Saved raw pre-transform GLB meshes to {raw_mesh_dir}")

        # Export meshes from SAM3D outputs
        ori_results['meshes'] = make_scene_untextured_mesh(*outputs)
        ori_results['bboxes'], ori_results['axes'] = get_object_bboxes_from_outputs(*outputs)

        # Save raw meshes to cache
        logger.info(f"\nSaving {len(ori_results['meshes'])} raw meshes...")
        for mesh, bbox, axes, obj_name in zip(ori_results['meshes'], ori_results['bboxes'], ori_results['axes'], object_names):
            # Save mesh
            mesh.export(os.path.join(output_obj_dir, f"{obj_name}.obj"))
            # Save bbox and axes as npz
            npz_path = os.path.join(output_bbox_axes_dir, f"{obj_name}_bbox_axes.npz")
            np.savez(npz_path, bbox_corners=bbox['corners'], bbox_center=bbox['center'], axes=axes)

            # Identify reference meshes (floor, carpet)
            if "floor" in obj_name or "carpet" in obj_name:
                ori_results['meshes_ref'].append(mesh)
                ori_results['axes_ref'].append(axes)

            # Save visualization PLY files (only if debug_vis)
            if args.debug_vis:
                vis_bboxes_ply(bbox['corners'], os.path.join(output_vis_dir, f"{obj_name}_bbox.ply"))
                vis_axes_ply(bbox['center'], axes, os.path.join(output_vis_dir, f"{obj_name}_axes.ply"))

        logger.info(f"✓ Raw scene meshes saved to {output_obj_dir}")

    return ori_results


def y_align_with_cache(ori_results, object_names, output_obj_dir, output_obj_y_align_dir, output_bbox_axes_y_align_dir, output_vis_dir, args):
    """
    Align each object's up-axis to +Y (y-align), with caching support.

    Returns:
        dict: y_align_results containing meshes, bboxes, axes
    """
    use_cache = args.use_cache and not args.force_reinfer
    cache_exists = all(
        (os.path.exists(os.path.join(output_obj_y_align_dir, f"scene_y_align_{obj_name}.obj")) and
         os.path.exists(os.path.join(output_bbox_axes_y_align_dir, f"scene_y_align_{obj_name}_bbox_axes.npz")))
        for obj_name in object_names
    )

    y_align_results = defaultdict(list)

    if use_cache and cache_exists:
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Found cached y-aligned meshes, bboxes, and axes, loading from disk...")
        logger.info(f"{'='*60}\n")

        # Load cached y-aligned meshes, bboxes, and axes
        for idx, obj_name in enumerate(object_names):
            # Load mesh
            cache_mesh_path = os.path.join(output_obj_y_align_dir, f"scene_y_align_{obj_name}.obj")
            y_align_results['meshes'].append(trimesh.load(cache_mesh_path, force='mesh'))

            # Load cached bbox and axes data from NPZ
            cache_npz_path = os.path.join(output_bbox_axes_y_align_dir, f"scene_y_align_{obj_name}_bbox_axes.npz")
            bbox_axes_data = np.load(cache_npz_path)
            y_align_results['bboxes'].append({
                'corners': bbox_axes_data['bbox_corners'],
                'center': bbox_axes_data['bbox_center']
            })
            y_align_results['axes'].append(bbox_axes_data['axes'])

        logger.info(f"Loaded {len(y_align_results['meshes'])} cached y-aligned meshes")

    else:
        if args.force_reinfer:
            logger.info(f"\n{'='*60}")
            logger.info(f"Force recompute enabled, running y-align...")
            logger.info(f"{'='*60}\n")
        else:
            logger.info(f"\n{'='*60}")
            logger.info(f"No cache found, running y-align...")
            logger.info(f"{'='*60}\n")

        # Snapshot vertices before transform (y_align copies meshes; ori_results stays untouched)
        pre_verts = [mesh.vertices.copy() for mesh in ori_results['meshes']]

        # Run y-align
        y_align_results = y_align(ori_results, object_names=object_names, coarse2fine=True)

        # Cache y-aligned meshes by re-reading the original OBJ and transforming only the xyz columns (preserving color, faces, etc.)
        logger.info(f"\nSaving {len(y_align_results['meshes'])} y-aligned meshes...")
        for i, (mesh, bbox, axes, obj_name) in enumerate(zip(y_align_results['meshes'], y_align_results['bboxes'], y_align_results['axes'], object_names)):
            orig_obj_path = os.path.join(output_obj_dir, f"{obj_name}.obj")
            y_align_obj_path = os.path.join(output_obj_y_align_dir, f"scene_y_align_{obj_name}.obj")

            # Compute the rigid transform and apply it to the raw OBJ (preserving color, faces, etc.)
            R, t = compute_rigid_transform(pre_verts[i], mesh.vertices)
            save_obj_with_rigid_transform(orig_obj_path, y_align_obj_path, R, t)

            # Save bbox and axes as npz
            npz_path = os.path.join(output_bbox_axes_y_align_dir, f"scene_y_align_{obj_name}_bbox_axes.npz")
            np.savez(npz_path, bbox_corners=bbox['corners'], bbox_center=bbox['center'], axes=axes)

            # Save visualization PLY files (only if debug_vis)
            if args.debug_vis:
                vis_bboxes_ply(bbox['corners'], os.path.join(output_vis_dir, f"scene_y_align_{obj_name}_bbox.ply"))
                vis_axes_ply(bbox['center'], axes, os.path.join(output_vis_dir, f"scene_y_align_{obj_name}_axes.ply"))

        logger.info(f"✓ Y-aligned scene meshes cached to {output_obj_y_align_dir}")

    return y_align_results


def place_to_ground(output_obj_y_align_dir, output_obj_canon_dir, scene_tree_path,
                    ceiling_height_threshold=1.8):
    """
    BFS over parent/child relations in scene_tree.json to adjust per-object height:
    - children of "floor": shift so the lowest vertex sits at y = 0;
    - children of "floor-wall": same as floor children (resting on floor while attached to a wall);
    - children of "wall": apply the median y-offset from floor / floor-wall children (global shift that preserves relative height);
    - children of "ceiling": same median offset as wall children;
    - other children: place the lowest vertex on top of the (already-adjusted) parent's highest vertex;
    - the floor itself is ignored.

    Finally the ceiling height is set to max(scene max-y, ceiling_threshold).
    Returns:
        ceiling_height (float): resolved ceiling height
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Place to ground according to scene tree...")
    logger.info(f"{'='*60}\n")

    with open(scene_tree_path) as f:
        scene_tree = json.load(f)

    # Build parent_map, children_map, and relation_map
    parent_map = {}
    relation_map = {}
    children_map = defaultdict(list)
    VALID_ROOTS = {"floor", "wall", "ceiling", "floor-wall"}
    for edge in scene_tree["edges"]:
        child = edge["child"]
        parent = edge["parent"]
        # Patch: parents starting with "the_floor_" should resolve to the root "floor"
        if parent.startswith("the_floor"):
            parent = "floor"
            edge["parent"] = "floor"
        parent_map[child] = parent
        relation_map[child] = edge.get("relation", "on")
        children_map[parent].append(child)

    # Pre-read all OBJ vertex data
    obj_verts = {}  # obj_name -> (vertices, lines, min_y, max_y)
    all_nodes = set()
    for root in scene_tree["roots"]:
        all_nodes.update(children_map.get(root, []))
    # BFS to collect every node
    bfs_q = deque(all_nodes)
    while bfs_q:
        n = bfs_q.popleft()
        for c in children_map.get(n, []):
            if c not in all_nodes:
                all_nodes.add(c)
                bfs_q.append(c)

    for obj_name in all_nodes:
        if obj_name.startswith("the_floor"):
            continue
        input_path = os.path.join(output_obj_y_align_dir, f"scene_y_align_{obj_name}.obj")
        if os.path.exists(input_path):
            vertices, lines = read_obj_vertices(input_path)
            min_y, max_y = get_y_bounds(vertices)
            obj_verts[obj_name] = (vertices, lines, min_y, max_y)

    # Pass 1: handle floor children, collecting y_offset to compute the reference offset
    floor_offsets = []
    max_y_after = {}
    processed = set()

    for obj_name in children_map.get("floor", []):
        if obj_name.startswith("the_floor") or obj_name not in obj_verts:
            processed.add(obj_name)
            continue
        vertices, lines, min_y, max_y = obj_verts[obj_name]
        y_offset = -min_y
        output_path = os.path.join(output_obj_canon_dir, f"scene_canon_{obj_name}.obj")
        write_obj_with_y_offset(output_path, lines, y_offset)
        max_y_after[obj_name] = max_y + y_offset
        floor_offsets.append(y_offset)
        processed.add(obj_name)
        logger.info(f"  {obj_name}: parent=floor, y=[{min_y:.4f},{max_y:.4f}] -> [0,{max_y + y_offset:.4f}]")

    # Pass 1b: handle floor-wall children (on the floor and attached to a wall); place on the floor like floor children
    for obj_name in children_map.get("floor-wall", []):
        if obj_name.startswith("the_floor") or obj_name not in obj_verts:
            processed.add(obj_name)
            continue
        vertices, lines, min_y, max_y = obj_verts[obj_name]
        y_offset = -min_y
        output_path = os.path.join(output_obj_canon_dir, f"scene_canon_{obj_name}.obj")
        write_obj_with_y_offset(output_path, lines, y_offset)
        max_y_after[obj_name] = max_y + y_offset
        floor_offsets.append(y_offset)
        processed.add(obj_name)
        logger.info(f"  {obj_name}: parent=floor-wall, y=[{min_y:.4f},{max_y:.4f}] -> [0,{max_y + y_offset:.4f}]")

    # Compute the reference offset using only objects whose min_y is meaningfully negative (< -0.05),
    # excluding outliers whose mesh is already near y = 0 (e.g. a floor lamp already reconstructed at y ~= 0)
    significant_offsets = [o for o in floor_offsets if o > 0.05]
    ref_offset = float(np.median(significant_offsets)) if significant_offsets else (
        float(np.median(floor_offsets)) if floor_offsets else 0.0
    )
    logger.info(f"\n  Reference offset from floor/floor-wall children (median, significant only): {ref_offset:.4f}\n")

    # Pass 2: handle wall children with the reference offset (global shift, preserves relative height) and clamp above the floor
    for obj_name in children_map.get("wall", []):
        if obj_name.startswith("the_floor") or obj_name not in obj_verts:
            processed.add(obj_name)
            continue
        vertices, lines, min_y, max_y = obj_verts[obj_name]
        y_offset = ref_offset + max(0.0, -(min_y + ref_offset))  # clamp to floor
        output_path = os.path.join(output_obj_canon_dir, f"scene_canon_{obj_name}.obj")
        write_obj_with_y_offset(output_path, lines, y_offset)
        max_y_after[obj_name] = max_y + y_offset
        processed.add(obj_name)
        logger.info(f"  {obj_name}: parent=wall, y_offset={y_offset:.4f}, y=[{min_y:.4f},{max_y:.4f}] -> [{min_y + y_offset:.4f},{max_y + y_offset:.4f}]")

    # Pass 2b: mark ceiling children as handled; the actual file write is done in Pass 2c (which does ceiling-align + floor clamp)
    for obj_name in children_map.get("ceiling", []):
        processed.add(obj_name)

    # Pass 3: BFS over the remaining children (grandchildren and deeper)
    queue = deque()
    for obj_name in processed:
        for child in children_map.get(obj_name, []):
            if child not in processed:
                queue.append(child)

    while queue:
        obj_name = queue.popleft()
        if obj_name in processed:
            continue

        if obj_name.startswith("the_floor"):
            processed.add(obj_name)
            continue

        if obj_name not in obj_verts:
            logger.info(f"  Warning: {obj_name} obj not found, skipping")
            processed.add(obj_name)
            continue

        vertices, lines, min_y, max_y = obj_verts[obj_name]
        parent = parent_map.get(obj_name, "floor")
        relation = relation_map.get(obj_name, "on")
        output_path = os.path.join(output_obj_canon_dir, f"scene_canon_{obj_name}.obj")

        if parent in max_y_after:
            if relation == "inside":
                # "inside": the y-align reconstruction already positions the child correctly
                # relative to the parent (e.g. on an interior shelf). Apply the same y offset
                # that was applied to the parent to preserve that relative position exactly.
                _, _, p_min_y, p_max_y = obj_verts[parent]
                parent_y_off = max_y_after[parent] - p_max_y
                target_min_y = min_y + parent_y_off
                logger.info(f"  {obj_name}: inside {parent}, parent_y_off={parent_y_off:.4f}, y=[{min_y:.4f},{max_y:.4f}] -> [{target_min_y:.4f},{max_y + parent_y_off:.4f}]")
            elif relation == "hang":
                # "hang": child top aligns with parent top + small clearance (child hangs from parent)
                # target: child_max_y = parent_max_y + 0.005
                # => y_offset = (max_y_after[parent] + 0.005) - max_y
                # => min_y after = min_y + y_offset = max_y_after[parent] + 0.005 - (max_y - min_y)
                target_max_y = max_y_after[parent] + 0.005
                y_offset = target_max_y - max_y
                write_obj_with_y_offset(output_path, lines, y_offset)
                max_y_after[obj_name] = target_max_y
                logger.info(f"  {obj_name}: hang from {parent}, top aligned: y=[{min_y:.4f},{max_y:.4f}] -> [{min_y + y_offset:.4f},{target_max_y:.4f}]")
                processed.add(obj_name)
                for child in children_map.get(obj_name, []):
                    if child not in processed:
                        queue.append(child)
                continue
            else:
                target_min_y = max_y_after[parent] + 0.005

            y_offset = target_min_y - min_y
            write_obj_with_y_offset(output_path, lines, y_offset)
            max_y_after[obj_name] = max_y + y_offset
            logger.info(f"  {obj_name}: parent={parent} ({relation}), y=[{min_y:.4f},{max_y:.4f}] -> [{target_min_y:.4f},{max_y + y_offset:.4f}]")
        else:
            # Parent unprocessed (unexpected): drop the object onto the floor
            y_offset = -min_y
            write_obj_with_y_offset(output_path, lines, y_offset)
            max_y_after[obj_name] = max_y + y_offset
            logger.info(f"  {obj_name}: parent={parent} (unprocessed), fallback to floor, y -> [0,{max_y + y_offset:.4f}]")

        processed.add(obj_name)

        for child in children_map.get(obj_name, []):
            if child not in processed:
                queue.append(child)

    # Determine ceiling height: take max over non-ceiling-child top points and the threshold
    # (ceiling children themselves are excluded to avoid a circular dependency)
    ceiling_children = set(children_map.get("ceiling", []))
    non_ceiling_max = max(
        (v for k, v in max_y_after.items() if k not in ceiling_children),
        default=0.0
    )
    ceiling_height = max(non_ceiling_max, ceiling_height_threshold)
    logger.info(f"\n  Non-ceiling scene max Y: {non_ceiling_max:.4f}, ceiling threshold: {ceiling_height_threshold:.4f}")
    logger.info(f"  Ceiling height: {ceiling_height:.4f}")

    # Pass 2c: shift ceiling children by ref_offset (same as wall children),
    # then push them down only if they overshoot ceiling_height (do not force-align to it).
    # This preserves relative height in the y-align coordinate frame and prevents the ceiling_height threshold
    # from pushing already-reasonable objects (e.g. a chandelier) too high.
    for obj_name in ceiling_children:
        if obj_name not in obj_verts:
            continue
        vertices, lines, min_y, max_y = obj_verts[obj_name]
        # Step 1: shift by ref_offset and clamp above the floor (same rule as wall children)
        y_offset = ref_offset + max(0.0, -(min_y + ref_offset))
        # Step 2: push down if any vertex still exceeds ceiling_height
        if max_y + y_offset > ceiling_height:
            y_offset = ceiling_height - max_y
            y_offset = max(y_offset, -min_y)  # clamp above floor again
        output_path = os.path.join(output_obj_canon_dir, f"scene_canon_{obj_name}.obj")
        write_obj_with_y_offset(output_path, lines, y_offset)
        max_y_after[obj_name] = max_y + y_offset
        logger.info(f"  {obj_name}: ceiling-ref_offset, y=[{min_y:.4f},{max_y:.4f}] -> [{min_y + y_offset:.4f},{max_y + y_offset:.4f}]")

    logger.info(f"  Processed {len(processed)} objects -> {output_obj_canon_dir}")

    return ceiling_height


def main():
    parser = argparse.ArgumentParser(description="Batch inference script for SAM3D-Objects")

    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Path to a single image file, or a folder of images (mutually exclusive with --image_list)"
    )
    parser.add_argument(
        "--image_list",
        type=str,
        default=None,
        help="Path to image list file, each line: image_path, output_name"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Root output directory. Stage-1 input is read from "
             "{output_folder}/{image_stem}/stage1/ and Stage-2 output is written "
             "to {output_folder}/{image_stem}/stage2/ (default: output)"
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=True,
        help="Use cached SAM3D inference results if available (default: True)"
    )
    parser.add_argument(
        "--force_reinfer",
        action="store_true",
        help="Force re-inference even if cache exists"
    )
    parser.add_argument(
        "--debug_vis",
        action="store_true",
        help="Enable debug visualization"
    )
    parser.add_argument(
        "--debug_raw_mesh",
        action="store_true",
        help="Save raw pre-transform GLB meshes to debug_raw_glb/ for orientation diagnosis"
    )
    args = parser.parse_args()

    # Build (img_path, output_name) list, mirroring infer_scenetree's logic.
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    entries = []
    if args.image_list:
        with open(args.image_list) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",", 1)]
                if len(parts) != 2:
                    logger.info(f"  Warning: skipping malformed line: {line!r}")
                    continue
                img_path, output_name = parts
                if not os.path.isfile(img_path):
                    logger.info(f"  Warning: image not found, skipping: {img_path}")
                    continue
                entries.append((img_path, output_name))
    elif args.image_folder:
        if os.path.isfile(args.image_folder):
            found = [args.image_folder]
        else:
            found = []
            for ext in image_extensions:
                found.extend(glob(os.path.join(args.image_folder, f"*{ext}")))
                found.extend(glob(os.path.join(args.image_folder, f"*{ext.upper()}")))
        for img_path in sorted(found):
            output_name = os.path.splitext(os.path.basename(img_path))[0]
            entries.append((img_path, output_name))
    else:
        raise ValueError("Must provide either --image_folder or --image_list")

    logger.info(f"Total images to process: {len(entries)}")
    os.makedirs(args.output_folder, exist_ok=True)

    for img_path, output_name in entries:
        image = load_image(img_path)

        # Per-image, per-stage output layout: {output_root}/{image_stem}/stage2/
        output_dir                = os.path.join(args.output_folder, output_name, "stage2")
        output_obj_dir            = os.path.join(output_dir, "scene_raw", "obj_files")
        output_bbox_axes_dir      = os.path.join(output_dir, "scene_raw", "bbox_axes")
        output_obj_y_align_dir        = os.path.join(output_dir, "scene_y_align", "obj_files")
        output_bbox_axes_y_align_dir  = os.path.join(output_dir, "scene_y_align", "bbox_axes")
        output_obj_canon_dir      = os.path.join(output_dir, "scene_canon", "obj_files")
        output_urdf_canon_dir     = os.path.join(output_dir, "scene_canon", "urdf_files")
        dirs = [output_dir, output_obj_dir, output_bbox_axes_dir,
                output_obj_y_align_dir, output_bbox_axes_y_align_dir,
                output_obj_canon_dir, output_urdf_canon_dir]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

        output_vis_dir = os.path.join(output_dir, "vis")
        if args.debug_vis:
            os.makedirs(output_vis_dir, exist_ok=True)

        # Capture this image's log into a timestamped stage2_log_<YYYYMMDD_HHMMSS>.txt
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_fh = attach_file_handler(logger, os.path.join(output_dir, f"stage2_log_{_ts}.txt"))

        logger.info(f"\n{'='*60}")
        logger.info(f"Running stage2: scene canonicalization, Processing image: {img_path}")
        logger.info(f"{'='*60}")

        # Stage-1 input is read from {output_root}/{image_stem}/stage1/
        sam_dir        = os.path.join(args.output_folder, output_name, "stage1")
        mask_dir       = os.path.join(sam_dir, "segemented_obj")
        scene_tree_src = os.path.join(sam_dir, "scene_tree.json")

        if not os.path.isdir(mask_dir):
            logger.info(f"  Warning: segemented_obj not found at {mask_dir}, skipping")
            continue

        ext_globs = [f"*.{e.lstrip('.')}" for e in image_extensions]
        masks_list = sorted(
            mask
            for ext in ext_globs
            for mask in glob(os.path.join(mask_dir, ext))
        )
        object_names = [os.path.splitext(os.path.basename(m))[0] for m in masks_list]

        # ========================================================================
        # Raw mesh inference with SAM3D-Objects
        # ========================================================================
        ori_results = inference_mesh_with_cache(
            image, masks_list, object_names,
            output_obj_dir, output_bbox_axes_dir, output_vis_dir, args
        )

        # ========================================================================
        # Y-align (up-axis → +Y)
        # ========================================================================
        y_align_with_cache(
            ori_results, object_names,
            output_obj_dir, output_obj_y_align_dir, output_bbox_axes_y_align_dir, output_vis_dir, args
        )

        # ========================================================================
        # Place on Ground according to scene tree
        # ========================================================================
        output_tree = os.path.join(output_dir, "scene_tree.json")
        if not os.path.exists(scene_tree_src):
            raise FileNotFoundError(
                f"scene_tree.json not found at {scene_tree_src}. "
                f"Stage 2 requires the scene tree produced by stage 1 — "
                f"please run stage 1 first and check its scene_tree.json output."
            )
        shutil.copy(scene_tree_src, output_tree)
        place_to_ground(
            output_obj_y_align_dir, output_obj_canon_dir, output_tree
        )

        # ========================================================================
        # Generate URDF files
        # ========================================================================
        generate_urdf_files(object_names, output_urdf_canon_dir, prefix="scene_canon_")

        logger.info(f"\n{'='*60}")
        logger.info(f"Saving stage2 3D canonicalized scene meshes to {os.path.join(output_dir, 'scene_canon')}")
        logger.info(f"{'='*60}")

        logger.removeHandler(_log_fh)
        _log_fh.close()


if __name__ == "__main__":
    main()