import cv2
import argparse
import json
import os
import sys
import traceback
from datetime import datetime
import numpy as np
import pycocotools.mask as mask_utils
from collections import defaultdict
import re
from PIL import Image
from glob import glob
from functools import partial
from rest3d.utils.vis import visualize_masks_on_image_cv2, save_seg_obj
from rest3d.utils.vlm import set_vlm_backend, analyze_scene_object_lists, generate_vlm_response
from rest3d.utils.io import load_image
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference
from sam3.agent.viz import visualize
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from rest3d.utils.log import get_logger, attach_file_handler

logger = get_logger("stage1")


def _excepthook(exc_type, exc_value, exc_tb):
    """Route uncaught exceptions through the logger so the per-image FileHandler
    captures the traceback into stage1_log_*.txt before the script exits."""
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.error("Uncaught exception:\n" + tb_text)
    sys.__excepthook__(exc_type, exc_value, exc_tb)

sys.excepthook = _excepthook


def build_agent_components(args):
    """Build the segmentation agent: a SAM3 processor + a VLM backend."""
    # Build SAM3 processor
    model = build_sam3_image_model()
    sam3_processor = Sam3Processor(model, confidence_threshold=0.5)

    # llm_config is used only to name output files
    llm_config = {"name": args.vlm_backend}

    # Replace the vLLM-server send_generate_request with our VLM backend
    def send_generate_request(messages, _max_retries=3):
        def _convert(msgs):
            converted = []
            for msg in msgs:
                content = msg["content"]
                if isinstance(content, str):
                    converted.append({"role": msg["role"], "content": [{"type": "text", "text": content}]})
                    continue
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image" and isinstance(item.get("image"), str):
                        new_content.append({"type": "image", "image": Image.open(item["image"]).convert("RGB")})
                    else:
                        new_content.append(item)
                converted.append({"role": msg["role"], "content": new_content})
            return converted

        def _clean_think(text):
            """Strip <think> blocks (closed or truncated/unclosed)."""
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            if '<think>' in text:
                text = text[:text.index('<think>')]
            return text.strip()

        FORMAT_HINT = (
            "IMPORTANT: Keep <think> under 100 words. "
            "If multiple masks partially cover the target, just select all of them with select_masks_and_return. "
            "Disambiguation rule for 'X on/in Y' queries: "
            "(1) If Y is a physical container that holds X (vase, bowl, jar, cup, basket, tray, pot), "
            "use Y as the text_prompt to capture the whole unit (e.g. 'flowers in glass vase' → text_prompt='glass vase'). "
            "(2) If Y is furniture or a surface that X merely rests on (table, shelf, bookcase, desk, counter, floor), "
            "X is the actual grounding target — use X as the text_prompt (e.g. 'books on coffee table' → text_prompt='books'). "
            "You MUST end with a <tool> JSON call. Example: "
            '<tool> {"name": "segment_phrase", "parameters": {"text_prompt": "noun phrase"}} </tool>'
        )

        # Detect if this is the examine_each_mask (Accept/Reject) phase by checking system prompt content
        _is_checking_phase = any(
            isinstance(m.get("content"), str) and "detail-oriented visual understanding" in m["content"]
            for m in messages if m.get("role") == "system"
        )

        # Only inject FORMAT_HINT on Round 1 (no prior segment_phrase assistant message yet)
        _is_first_round = not any(
            m.get("role") == "assistant" and
            any(isinstance(c, dict) and "segment_phrase" in c.get("text", "") for c in (m.get("content") if isinstance(m.get("content"), list) else []))
            for m in messages
        )

        prev_response = None
        for attempt in range(_max_retries):
            converted = _convert(messages)
            if prev_response is None:
                # First attempt: inject FORMAT_HINT only in Round 1 (non-checking phase)
                if not _is_checking_phase and _is_first_round:
                    converted.append({"role": "user", "content": [{"type": "text", "text": FORMAT_HINT}]})
            else:
                # Retry: feed the truncated response back to the model
                converted.append({"role": "assistant", "content": [{"type": "text", "text": prev_response}]})
                if _is_checking_phase:
                    converted.append({"role": "user", "content": [{"type": "text", "text":
                        "Your response was cut off. Output ONLY your verdict now: "
                        "<verdict>Accept</verdict> or <verdict>Reject</verdict>."}]})
                else:
                    converted.append({"role": "user", "content": [{"type": "text", "text":
                        "Your response was cut off before the <tool> call. "
                        "Output ONLY the <tool> JSON call now, no thinking. Example: "
                        '<tool> {"name": "segment_phrase", "parameters": {"text_prompt": "noun phrase"}} </tool>'}]})
            response = generate_vlm_response(converted)
            clean = _clean_think(response)
            if _is_checking_phase:
                # checking phase expects <verdict>...</verdict>; do not check <tool>
                return clean
            if "<tool>" in clean and "</tool>" in clean:
                return clean
            prev_response = response
            if attempt < _max_retries - 1:
                logger.info(f"    ⚠️ VLM response missing <tool> tags (attempt {attempt+1}), retrying...")
        return clean

    call_sam_service = partial(
        call_sam_service_orig,
        sam3_processor=sam3_processor,
    )

    return llm_config, send_generate_request, call_sam_service


def decode_agent_masks(output_json_path):
    """Decode the RLE mask from the agent's JSON output into a numpy array."""
    with open(output_json_path, 'r') as f:
        pred = json.load(f)

    h = pred["orig_img_h"]
    w = pred["orig_img_w"]
    rle_masks = pred.get("pred_masks", [])
    scores = pred.get("pred_scores", [])

    masks = []
    valid_scores = []
    for i, rle_str in enumerate(rle_masks):
        rle = {"counts": rle_str, "size": [h, w]}
        binary_mask = mask_utils.decode(rle).astype(np.float32)
        masks.append(binary_mask)
        if i < len(scores):
            valid_scores.append(scores[i])

    if masks:
        masks = np.stack(masks)
    else:
        masks = np.zeros((0, h, w), dtype=np.float32)

    return masks, np.array(valid_scores)


def analyze_scene_tree(image_path, seg_obj_dir, agent_output_dir, image_stem, llm_name, save_dir, save_debug=False):
    """
    After segmentation, build the scene tree by calling the VLM per object on SAM3 viz overlays.
    The object list is taken strictly from filenames under segemented_obj/ (per mask id), skipping the_floor_*.

    Args:
        image_path: path to the source image
        seg_obj_dir: segemented_obj/ directory defining the full object list
        agent_output_dir: directory of agent outputs (contains *_pred.json)
        image_stem: image filename without extension
        llm_name: VLM backend name (used in output filenames)
        save_dir: directory in which to save the scene tree
        save_debug: if True, persist debug_scene_tree/ on disk (per-object overlays
            and the per-message VLM input images). When False, overlays stay
            in-memory only and no debug folder is created.
    """
    debug_dir = os.path.join(save_dir, "debug_scene_tree") if save_debug else None
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    # Step 1: scan segemented_obj/ for the full per-id object list, skipping the_floor_*
    seg_files = sorted(glob(os.path.join(seg_obj_dir, "*.png")))
    obj_ids = []
    for f in seg_files:
        obj_id = os.path.splitext(os.path.basename(f))[0]
        if obj_id.startswith("the_floor"):
            continue
        obj_ids.append(obj_id)

    logger.info(f"\n{'='*40}")
    logger.info(f"Constructing scene tree for {len(obj_ids)} objects (from segemented_obj)...")

    # Parse obj_id -> (prompt_safe, mask_idx), grouped by prompt_safe
    def parse_obj_id(obj_id):
        parts = obj_id.rsplit("_", 1)
        return parts[0], int(parts[1])

    groups = defaultdict(list)
    for obj_id in obj_ids:
        prompt_safe, mask_idx = parse_obj_id(obj_id)
        groups[prompt_safe].append((obj_id, mask_idx))

    # Step 2: generate or reuse an overlay per obj_id, compute mask centers
    obj_data = {}
    pred_cache = {}

    for prompt_safe, items in groups.items():
        base_filename = f"{image_stem}_{prompt_safe}_agent_{llm_name}"
        json_path = os.path.join(agent_output_dir, f"{base_filename}_pred.json")
        if not os.path.exists(json_path):
            logger.info(f"  Scene tree: no pred JSON for '{prompt_safe}', skipping")
            continue

        if json_path not in pred_cache:
            with open(json_path) as f:
                pred_cache[json_path] = json.load(f)
        pred_json = pred_cache[json_path]
        h, w = int(pred_json["orig_img_h"]), int(pred_json["orig_img_w"])

        for obj_id, mask_idx in items:
            if mask_idx >= len(pred_json.get("pred_masks", [])):
                logger.info(f"  Scene tree: mask_idx {mask_idx} out of range for '{obj_id}', skipping")
                continue

            # Reuse an existing overlay if present (only when save_debug is on),
            # otherwise generate it in memory.
            overlay_path = (
                os.path.join(debug_dir, f"overlay_{obj_id}.png")
                if debug_dir is not None else None
            )
            overlay = None
            if overlay_path is not None and os.path.exists(overlay_path):
                try:
                    overlay = Image.open(overlay_path).convert("RGB")
                except Exception:
                    os.remove(overlay_path)  # truncated/corrupted, delete and regenerate
                    overlay = None
            if overlay is None:
                try:
                    pil_mask_i, _ = visualize(pred_json, zoom_in_index=mask_idx)
                    overlay = pil_mask_i
                    if overlay_path is not None:
                        overlay.save(overlay_path)
                except Exception as e:
                    logger.info(f"  Scene tree: failed to visualize '{obj_id}': {e}")
                    continue

            # Compute the center of the single mask
            rle_str = pred_json["pred_masks"][mask_idx]
            rle = {"counts": rle_str, "size": [h, w]}
            binary = mask_utils.decode(rle)
            ys, xs = np.where(binary > 0)
            center = (float(xs.mean()), float(ys.mean())) if len(xs) > 0 else (w / 2.0, h / 2.0)

            obj_data[obj_id] = {"overlay": overlay, "center": center}

    # Step 3: query the VLM per obj_id to determine its parent
    available_parents = ["floor", "wall", "ceiling", "floor-wall"] + obj_ids
    edges = []
    result_lines = []

    # Preload the_floor mask to detect the floor-copy case (and avoid self-loops)
    floor_mask = None
    floor_png = os.path.join(seg_obj_dir, "the_floor.png")
    if os.path.exists(floor_png):
        fm = np.array(Image.open(floor_png))
        floor_mask = fm[:, :, 3] > 0 if fm.ndim == 4 else fm.any(axis=-1) if fm.ndim == 3 else fm > 0

    def _iou_with_floor(obj_id):
        if floor_mask is None or obj_id not in obj_data:
            return 0.0
        obj_png = os.path.join(seg_obj_dir, f"{obj_id}.png")
        if not os.path.exists(obj_png):
            return 0.0
        om = np.array(Image.open(obj_png))
        obj_mask = om[:, :, 3] > 0 if om.ndim == 4 else om.any(axis=-1) if om.ndim == 3 else om > 0
        inter = np.logical_and(floor_mask, obj_mask).sum()
        union = np.logical_or(floor_mask, obj_mask).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    for obj_id in obj_ids:
        if obj_id not in obj_data:
            edges.append({"child": obj_id, "parent": "floor", "relation": "on"})
            result_lines.append(f"{obj_id} -> floor | on")
            continue

        # Find nearby objects by distance
        cx, cy = obj_data[obj_id]["center"]
        distances = []
        for other in obj_ids:
            if other == obj_id or other not in obj_data:
                continue
            ox, oy = obj_data[other]["center"]
            dist = ((cx - ox)**2 + (cy - oy)**2)**0.5
            distances.append((dist, other))
        distances.sort()
        nearby = [n for _, n in distances[:5]]

        # Build VLM message: original image + current obj overlay + nearby obj overlays
        scene_img = Image.open(image_path).convert("RGB")
        content = [
            {"type": "image", "image": scene_img},
            {"type": "text", "text": "Full scene image above.\n\n"},
            {"type": "image", "image": obj_data[obj_id]["overlay"]},
            {"type": "text", "text": f'Current object: "{obj_id}" (colored mask overlay above)\n\nNearby objects:\n'},
        ]
        for nearby_id in nearby:
            content.append({"type": "image", "image": obj_data[nearby_id]["overlay"]})
            content.append({"type": "text", "text": f'Nearby object: "{nearby_id}"\n'})

        parents_str = ", ".join(f'"{p}"' for p in available_parents if p != obj_id)
        content.append({"type": "text", "text": (
            f'\nDetermine what supports or holds "{obj_id}" in this scene.\n'
            f"The parent MUST be one of: {parents_str}\n\n"
            "Rules:\n"
            '- "on": object rests on the TOPMOST surface of the parent — the parent does not extend above the object\n'
            '- "inside": object rests on an INTERMEDIATE horizontal surface of the parent — the parent\'s structure\n'
            "  extends above the object (e.g. item on a countertop that is part of a merged cabinet system which also has upper cabinets; item stored inside a basket, box, or drawer unit).\n"
            "  Judge by looking at the WHOLE parent mask as a single object, not individual parts.\n"
            '- "attach": mounted/fixed to a surface — use "wall" or "ceiling" as parent\n'
            "  - wall attach: picture frame, window, wall shelf, wall-mounted TV\n"
            "  - ceiling attach: hanging lamp, ceiling fan\n"
            '- "hang": object is draped/hung from a rod, rail, or hook — use the rod/rail as parent\n'
            "  - e.g. curtains hang from curtain rod, coats hang from hook\n"
            "- If object sits on another object, parent is that object, not floor\n"
            "- If object rests on floor AND is fixed against/to a wall (radiator, large cabinet, built-in unit):\n"
            '  use parent "floor-wall", relation "on-attach"\n\n'
            "Type rules:\n"
            '- "fixed": immovable furniture (cabinets, shelves, radiators, built-in units, bookcases) or anything attached to wall/ceiling\n'
            '- "movable": can be picked up or pushed (chairs, cups, books, toys, etc.)\n\n'
            f"Output ONLY one line:\n"
            f"{obj_id} -> parent_name | relation | type\n"
        )})

        if debug_dir is not None:
            img_idx = 0
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    item["image"].save(os.path.join(debug_dir, f"msg_{obj_id}_{img_idx}.png"))
                    img_idx += 1

        # Self-loop guard: if the mask is identical to the_floor, force parent=floor
        if _iou_with_floor(obj_id) > 0.99:
            logger.info(f"  ⚠️  '{obj_id}' mask is identical to the_floor (floor-copy fallback) — forcing parent=floor")
            edges.append({"child": obj_id, "parent": "floor", "relation": "on", "type": "movable"})
            result_lines.append(f"{obj_id} -> floor | on | movable  [floor-copy forced]")
            continue

        messages = [{"role": "user", "content": content}]
        logger.info(f"  Querying parent for: '{obj_id}'...")
        response = generate_vlm_response(messages)
        logger.info(f"    Response: {response.strip()}")

        # Parse format: obj_id -> parent | relation | type
        found = False
        for line in response.strip().split("\n"):
            line = line.strip()
            if "->" not in line:
                continue
            parts = line.split("->")
            if len(parts) != 2:
                continue
            rest = parts[1].strip()
            fields = [f.strip() for f in rest.split("|")]
            if len(fields) >= 3:
                parent, relation, obj_type = fields[0], fields[1], fields[2]
            elif len(fields) == 2:
                parent, relation = fields[0], fields[1]
                obj_type = "movable"
            else:
                parent = fields[0]
                relation = "on"
                obj_type = "movable"
            # normalize type
            obj_type = obj_type.lower().strip()
            if obj_type not in ("fixed", "movable"):
                obj_type = "movable"
            parent = parent.strip()
            # Case-insensitive match
            parent_lower = parent.lower()
            matched = next((p for p in available_parents if p.lower() == parent_lower), None)
            if matched is None:
                logger.info(f"    ⚠️ Invalid parent '{parent}' (not in segemented_obj or roots), fallback to floor")
                parent = "floor"
                relation = "on"
                obj_type = "movable"
            else:
                parent = matched  # use canonical casing
            edges.append({"child": obj_id, "parent": parent, "relation": relation, "type": obj_type})
            result_lines.append(f"{obj_id} -> {parent} | {relation} | {obj_type}")
            found = True
            break
        if not found:
            edges.append({"child": obj_id, "parent": "floor", "relation": "on", "type": "movable"})
            result_lines.append(f"{obj_id} -> floor | on | movable")

    # Save aggregated scene tree as JSON
    os.makedirs(save_dir, exist_ok=True)
    scene_tree = {
        "roots": ["floor", "wall", "ceiling", "floor-wall"],
        "nodes": obj_ids,
        "edges": edges,
    }
    tree_path = os.path.join(save_dir, "scene_tree.json")
    with open(tree_path, 'w') as f:
        json.dump(scene_tree, f, indent=2, ensure_ascii=False)
    logger.info(f"Scene tree saved to {tree_path} ({len(edges)} edges)")

    return scene_tree


def _pick_most_floor_like(seg_obj_dir, image_rgb):
    """
    Called when the floor agent fails after 10 rounds. Picks the most floor-like mask
    from the existing segemented_obj/ masks, scored as x_span * bottom_pos * bbox_area_frac.
    The selected mask is COPIED as the_floor (the original PNG is kept).
    Note: the_floor.png will be identical to one object mask; dedup and scene-tree logic handle this case.
    """
    seg_files = sorted(glob(os.path.join(seg_obj_dir, "*.png")))
    if not seg_files:
        return None

    if hasattr(image_rgb, 'shape'):
        img_h, img_w = image_rgb.shape[:2]
    else:
        img_w, img_h = image_rgb.size

    best_score, best_name, best_mask = -1.0, None, None
    for fpath in seg_files:
        name = os.path.splitext(os.path.basename(fpath))[0]
        if name == "the_floor" or name.startswith("the_floor_"):
            continue
        m = np.array(Image.open(fpath).convert("L")) > 0
        if m.sum() == 0:
            continue
        # Use bounding box rather than pixel count: an occluded floor has sparse pixels but a wide bbox
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]
        y_min, y_max = rows[0], rows[-1]
        x_min, x_max = cols[0], cols[-1]
        x_span = (x_max - x_min + 1) / img_w       # horizontal extent: a floor should span the image
        bottom_pos = y_max / img_h                  # bbox-bottom location (lower in image = better)
        bbox_area_frac = ((x_max - x_min + 1) * (y_max - y_min + 1)) / (img_w * img_h)
        score = x_span * bottom_pos * bbox_area_frac
        if score > best_score:
            best_score, best_name, best_mask = score, name, m

    if best_name is None:
        return None

    logger.info(f"    ⚠️  floor fallback (copy): using '{best_name}' as floor (score={best_score:.2f}), original PNG kept")

    # Encode best_mask as RLE and write a synthetic pred.json
    img_h, img_w = best_mask.shape
    rle = mask_utils.encode(np.asfortranarray(best_mask.astype(np.uint8)))
    result_json = {
        "orig_img_h": img_h,
        "orig_img_w": img_w,
        "pred_boxes": [],
        "pred_masks": [rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]],
        "pred_scores": [1.0],
    }
    return result_json


def dedup_seg_masks(seg_obj_dir, agent_output_dir, img_output_dir, image_stem, llm_name,
                    iou_thresh=0.3, overlap_thresh=0.8):
    """
    Pairwise-deduplicate masks inside segemented_obj/:
    - if one side is the_floor, keep the_floor and drop the other;
    - otherwise keep the alphabetically-earlier filename.
    """
    seg_files = sorted(glob(os.path.join(seg_obj_dir, "*.png")))
    names = [os.path.splitext(os.path.basename(f))[0] for f in seg_files]

    def is_floor(n):
        return n == "the_floor" or n.startswith("the_floor_")

    # Load every mask
    masks = {}
    for name, fpath in zip(names, seg_files):
        m = np.array(Image.open(fpath).convert("L")) > 0
        if m.sum() > 0:
            masks[name] = m

    removed = set()
    name_list = list(masks.keys())
    for i in range(len(name_list)):
        a = name_list[i]
        if a in removed:
            continue
        for j in range(i + 1, len(name_list)):
            b = name_list[j]
            if b in removed:
                continue
            ma, mb = masks[a], masks[b]
            intersection = np.logical_and(ma, mb).sum()
            if intersection == 0:
                continue
            union = np.logical_or(ma, mb).sum()
            iou = intersection / union
            overlap_a = intersection / ma.sum()
            overlap_b = intersection / mb.sum()
            if iou > iou_thresh or max(overlap_a, overlap_b) > overlap_thresh:
                # If one is the floor, it is likely the copy-fallback case: warn but do not delete
                if is_floor(a) != is_floor(b):
                    floor_name = a if is_floor(a) else b
                    obj_name = b if is_floor(a) else a
                    logger.info(f"  ⚠️  dedup WARNING: '{obj_name}' and '{floor_name}' have high overlap "
                          f"(IoU={iou:.2f}) — floor is likely a copy fallback, keeping both")
                    continue
                # Choose which to drop: two floor variants -> later one; otherwise the alphabetically later name
                to_remove = b if not is_floor(b) else a
                if is_floor(a) and is_floor(b):
                    to_remove = b
                logger.info(f"  dedup: removing '{to_remove}' (IoU={iou:.2f}, "
                      f"overlap_a={overlap_a:.2f}, overlap_b={overlap_b:.2f})")
                removed.add(to_remove)

    if not removed:
        return

    # Delete files
    for seg_name in removed:
        seg_png = os.path.join(seg_obj_dir, f"{seg_name}.png")
        if os.path.exists(seg_png):
            os.remove(seg_png)
        for extra in [
            os.path.join(agent_output_dir, f"{image_stem}_{seg_name}_agent_{llm_name}_pred.json"),
            os.path.join(img_output_dir, "overlay", f"{seg_name}.jpg"),
        ]:
            if os.path.exists(extra):
                os.remove(extra)
    logger.info(f"  dedup removed {removed}")


def load_cached_results(img_output_dir, agent_output_dir, image_stem, llm_name):
    """
    Try to load existing outputs to skip re-running the agent.

    Returns:
        objects: list[str] or None (None means no cache hit)
        cached_objects: set of object names that already have _pred.json
    """
    # Load the existing object list
    obj_list_path = os.path.join(img_output_dir, "scene_object_lists.txt")
    if not os.path.exists(obj_list_path):
        return None, set()

    with open(obj_list_path, "r") as f:
        objects = [line.strip() for line in f if line.strip()]
    if not objects:
        return None, set()

    # Always append "the floor" in memory (not written to disk)
    if "the floor" not in objects:
        objects.append("the floor")

    # Find objects that already have *_pred.json AND a matching PNG in segemented_obj/
    seg_obj_dir = os.path.join(img_output_dir, "segemented_obj")
    cached = set()
    for name in objects:
        prompt_safe = name.replace("/", "_").replace(" ", "_")
        base_filename = f"{image_stem}_{prompt_safe}_agent_{llm_name}"
        json_path = os.path.join(agent_output_dir, f"{base_filename}_pred.json")
        if not os.path.exists(json_path):
            continue
        # Also require at least one matching PNG inside segemented_obj/
        has_png = any(True for _ in glob(os.path.join(seg_obj_dir, f"{prompt_safe}_*.png")))
        if has_png:
            cached.add(name)

    return objects, cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Path to a single image file, or a folder of images"
    )
    parser.add_argument(
        "--image_list",
        type=str,
        default=None,
        help="Path to a txt file where each line is either an image path or a folder path"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Root output directory. Stage-1 output lands at "
             "{output_folder}/{image_stem}/stage1/ (default: output)"
    )
    parser.add_argument(
        "--vlm_backend",
        type=str,
        default="gemini",
        choices=["gpt4o", "gemini"],
        help="VLM backend to use: gpt4o (OpenAI API) or gemini (Google API)"
    )
    parser.add_argument(
        "--vlm_prompt_file",
        type=str,
        default="list_objects.txt",
        help="VLM prompt file. Relative paths resolve against rest3d/prompts/; "
             "absolute paths are used as-is. (default: list_objects.txt)"
    )
    parser.add_argument(
        "--save_debug",
        action="store_true",
        help="If set, persist the debug_scene_tree/ folder (per-object overlays + "
             "VLM input snapshots). Off by default — overlays stay in memory."
    )
    args = parser.parse_args()

    # List of (abs_image_path, output_name) tuples
    images_list = []

    if args.image_list:
        with open(args.image_list, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) != 2:
                logger.info(f"Warning: skipping malformed line (expected 'path, name'): {line}")
                continue
            img_path, output_name = parts
            if not os.path.isfile(img_path):
                logger.info(f"Warning: image not found, skipping: {img_path}")
                continue
            images_list.append((img_path, output_name))
    elif args.image_folder:
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        if os.path.isfile(args.image_folder):
            found = [args.image_folder]
        else:
            found = []
            for ext in image_extensions:
                found.extend(glob(os.path.join(args.image_folder, f"*{ext}")))
                found.extend(glob(os.path.join(args.image_folder, f"*{ext.upper()}")))
        for img_path in sorted(found):
            output_name = os.path.splitext(os.path.basename(img_path))[0]
            images_list.append((img_path, output_name))
    else:
        raise ValueError("Must provide either --image_folder or --image_list")

    output_root = args.output_folder
    os.makedirs(output_root, exist_ok=True)

    set_vlm_backend(args.vlm_backend)

    # Build agent components (SAM 3 segmentor + VLM backend)
    llm_config, send_generate_request, call_sam_service = build_agent_components(args)
    llm_name = llm_config["name"]

    for img_path, output_name in images_list:
        abs_img_path = os.path.abspath(img_path)
        image_stem = os.path.splitext(os.path.basename(img_path))[0]
        # Per-image, per-stage output: {output_root}/{image_stem}/stage1/
        img_output_dir = os.path.join(output_root, output_name, "stage1")
        agent_output_dir = os.path.join(img_output_dir, "segment_agent_out")
        seg_obj_dir = os.path.join(img_output_dir, "segemented_obj")
        os.makedirs(seg_obj_dir, exist_ok=True)

        # Capture this image's log into a timestamped stage1_log_<YYYYMMDD_HHMMSS>.txt
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_fh = attach_file_handler(logger, os.path.join(img_output_dir, f"stage1_log_{_ts}.txt"))

        logger.info(f"\n{'='*60}")
        logger.info(f"Running stage1: scene tree construction, Processing image: {img_path}")
        logger.info(f"VLM backend set to: {args.vlm_backend}")
        # Show the resolved prompt path (relative names live under rest3d/prompts/)
        _prompt_display = args.vlm_prompt_file if os.path.isabs(args.vlm_prompt_file) \
            else os.path.join("rest3d/prompts", args.vlm_prompt_file)
        logger.info(f"Stage1 prompt: {_prompt_display}")
        logger.info(f"{'='*60}")

        # Lazy load: check for cached results first
        cached_objects, cached_set = load_cached_results(
            img_output_dir, agent_output_dir, image_stem, llm_name
        )

        if cached_objects and len(cached_set) == len(cached_objects):
            # All objects already have agent output; skip to scene-tree analysis
            objects = cached_objects
            logger.info(f"Loaded {len(objects)} cached objects, skipping agent segmentation")
        else:
            # Step 1: VLM generates the object list
            if cached_objects:
                objects = cached_objects
                logger.info(f"Loaded {len(objects)} objects from cache, {len(cached_set)}/{len(objects)} already segmented")
            else:
                objects = analyze_scene_object_lists(abs_img_path, save_dir=img_output_dir, vlm_prompt_file=args.vlm_prompt_file)
                if not objects:
                    logger.info(f"VLM identified no objects; skipping {img_path}")
                    continue
                logger.info(f"Identified {len(objects)} objects: {objects}")

            # Load the image so we can save masks
            image = load_image(img_path, backend="cv2", image_format="bgr")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Step 2: run_single_image_inference for each object
            for obj_prompt in objects:
                if obj_prompt in cached_set:
                    logger.info(f"\n  Skipping '{obj_prompt}' (cached)")
                    continue
                logger.info(f"\n  Agent segmenting: '{obj_prompt}'")

                if obj_prompt == "the floor":
                    # Same VLM+SAM agent path as other objects (max 10 rounds)
                    prompt_safe_floor = "the_floor"
                    base_floor = f"{image_stem}_{prompt_safe_floor}_agent_{llm_name}"
                    floor_json_path = os.path.join(agent_output_dir, f"{base_floor}_pred.json")
                    floor_png_path = os.path.join(seg_obj_dir, "the_floor.png")

                    run_single_image_inference(
                        abs_img_path, obj_prompt, llm_config,
                        send_generate_request, call_sam_service,
                        output_dir=agent_output_dir, debug=args.save_debug,
                    )

                    # Check whether the agent succeeded
                    agent_succeeded = False
                    if os.path.exists(floor_json_path):
                        masks, _ = decode_agent_masks(floor_json_path)
                        if len(masks) > 0:
                            save_seg_obj(image, masks[0], out_path=floor_png_path)
                            logger.info(f"    floor agent succeeded: saved to the_floor.png")
                            agent_succeeded = True

                    if not agent_succeeded:
                        # Floor agent failed in 10 rounds: try SAM3 synonyms first, then fall back to copy
                        FLOOR_SYNONYMS = [
                            "carpet", "rug", "floor mat", "flooring",
                            "hardwood floor", "tile floor", "ground",
                        ]
                        synonym_succeeded = False
                        for synonym in FLOOR_SYNONYMS:
                            logger.info(f"    🔄 floor synonym retry: '{synonym}'")
                            # call_sam_service builds the save path internally; reconstruct it for the check
                            prompt_safe_syn = synonym.replace("/", "_").replace(" ", "_")
                            # output_folder_path subdir name is image_path.replace("/", "-")
                            syn_sub = abs_img_path.replace("/", "-")
                            syn_json = os.path.join(agent_output_dir, "sam_synonym",
                                                    syn_sub, f"{prompt_safe_syn}.json")
                            os.makedirs(os.path.dirname(syn_json), exist_ok=True)
                            call_sam_service(
                                abs_img_path, synonym,
                                output_folder_path=os.path.join(agent_output_dir, "sam_synonym"),
                            )
                            if os.path.exists(syn_json):
                                syn_masks, _ = decode_agent_masks(syn_json)
                                if len(syn_masks) > 0:
                                    # Write the synonym result into floor_json_path (keep format consistent)
                                    with open(syn_json) as f:
                                        result_syn = json.load(f)
                                    result_syn["text_prompt"] = "the floor"
                                    result_syn["image_path"] = abs_img_path
                                    json.dump(result_syn, open(floor_json_path, "w"), indent=4)
                                    save_seg_obj(image, syn_masks[0], out_path=floor_png_path)
                                    logger.info(f"    ✅ floor synonym '{synonym}' succeeded: saved to the_floor.png")
                                    synonym_succeeded = True
                                    break

                        if not synonym_succeeded:
                            # All synonyms failed -> copy the most floor-like object mask
                            result_json = _pick_most_floor_like(seg_obj_dir, image)
                            if result_json is not None:
                                result_json["text_prompt"] = "the floor"
                                result_json["image_path"] = abs_img_path
                                json.dump(result_json, open(floor_json_path, "w"), indent=4)
                                masks, _ = decode_agent_masks(floor_json_path)
                                if len(masks) > 0:
                                    save_seg_obj(image, masks[0], out_path=floor_png_path)
                            else:
                                logger.info(f"    ⚠️  floor: agent+synonyms failed and no object masks available, skipping")
                    continue  # Skip Step 3: floor already handled above
                else:
                    run_single_image_inference(
                        abs_img_path, obj_prompt, llm_config,
                        send_generate_request, call_sam_service,
                        output_dir=agent_output_dir, debug=args.save_debug,
                    )

                # Step 3: decode the RLE mask from the agent JSON and save it
                prompt_for_filename = obj_prompt.replace("/", "_").replace(" ", "_")
                base_filename = f"{image_stem}_{prompt_for_filename}_agent_{llm_name}"
                output_json_path = os.path.join(agent_output_dir, f"{base_filename}_pred.json")

                if not os.path.exists(output_json_path):
                    # Agent failed to pick a mask within 10 rounds; skip without saving a wrong result
                    logger.info(f"    Warning: agent failed to segment '{obj_prompt}'; skipping (no fallback mask written)")
                    continue

                masks, _ = decode_agent_masks(output_json_path)
                logger.info(f"    Found {len(masks)} masks")

                # Decide how to handle multiple masks based on object type:
                # - Fixed installations (cabinet/shelf/radiator...): union all into 1
                # - Everything else: trust the agent — keep all masks it selected
                _FIXED_KW = ["cabinet", "shelf", "shelving", "bookcase", "hutch",
                             "dresser", "wardrobe", "closet", "system", "radiator",
                             "fireplace", "window", "blinds", "built-in"]
                _name = obj_prompt.lower() + " "
                _is_fixed = any(kw in _name for kw in _FIXED_KW)

                if len(masks) > 1:
                    if _is_fixed:
                        # Union all masks → one complete mask for the installation
                        combined = np.zeros_like(masks[0])
                        for m in masks:
                            combined = np.logical_or(combined, m > 0.5).astype(np.float32)
                        masks = [combined]
                        logger.info(f"    Merged into 1 combined mask (fixed installation)")
                    else:
                        # Trust the agent: it selected N masks intentionally → keep all
                        logger.info(f"    Keeping {len(masks)} masks (agent selection)")

                prompt_safe = prompt_for_filename

                vis_overlay_masks_path = os.path.join(img_output_dir, "overlay", f"{prompt_safe}.jpg")
                os.makedirs(os.path.dirname(vis_overlay_masks_path), exist_ok=True)
                visualize_masks_on_image_cv2(image, masks, out_path=vis_overlay_masks_path)

                vis_each_seg_obj_dir = os.path.join(img_output_dir, "segemented_obj")
                os.makedirs(vis_each_seg_obj_dir, exist_ok=True)
                for mask_idx, mask in enumerate(masks):
                    vis_each_seg_obj_path = os.path.join(
                        vis_each_seg_obj_dir,
                        f"{prompt_safe}_{mask_idx:03d}.png"
                    )
                    save_seg_obj(image, mask, out_path=vis_each_seg_obj_path)

        # Step 4: pairwise-deduplicate masks (floor takes priority; otherwise keep either)
        dedup_seg_masks(seg_obj_dir, agent_output_dir, img_output_dir, image_stem, llm_name)

        # Step 5: analyze the scene tree (object list comes from segemented_obj/)
        logger.info(f"\n{'='*80}")
        logger.info(f"Scene tree construction: {output_name}")
        logger.info(f"{'='*80}")
        analyze_scene_tree(abs_img_path, seg_obj_dir, agent_output_dir, image_stem, llm_name, img_output_dir, save_debug=args.save_debug)

        logger.info(f"\n{'='*40}")
        logger.info(f"Saving stage1 object masks + scene tree to {img_output_dir}")
        logger.info(f"{'='*40}")

        logger.removeHandler(_log_fh)
        _log_fh.close()


if __name__ == "__main__":
    main()
