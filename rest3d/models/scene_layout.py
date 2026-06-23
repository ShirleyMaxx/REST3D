"""Scene tree parsing and SceneLayout data class."""
import json
import logging
import os
from collections import defaultdict

import numpy as np

from rest3d.utils.mesh import compute_mesh_info

_logger = logging.getLogger("stable_scene_opt")


def detect_file_prefix(urdf_dir, node_names):
    urdf_files = [f for f in os.listdir(urdf_dir) if f.endswith(".urdf")]
    if not urdf_files or not node_names:
        return ""
    prefix_votes = defaultdict(int)
    for uf in urdf_files:
        base = uf[:-5]
        for name in node_names:
            if base.endswith(name):
                candidate = base[: len(base) - len(name)]
                prefix_votes[candidate] += 1
    if not prefix_votes:
        return ""
    best = max(prefix_votes, key=prefix_votes.get)
    _logger.info(f"[Prefix] detected file prefix: '{best}' "
                 f"({prefix_votes[best]}/{len(node_names)} matches)")
    return best


def parse_scene_tree(path):
    with open(path, "r") as f:
        tree = json.load(f)
    roots = set(tree["roots"])
    edges = tree["edges"]
    all_named = set(tree.get("nodes", [])) | roots
    children  = defaultdict(list)
    parent_of = {}
    node_info = {}
    for e in edges:
        c, p = e["child"], e["parent"]
        children[p].append(c)
        parent_of[c] = p
        node_info[c] = {"relation": e["relation"], "type": e["type"]}
    if "floor" not in roots:
        raise ValueError("scene_tree has no 'floor' in roots")
    for c, p in parent_of.items():
        if p not in all_named:
            _logger.info(f"[WARN][parse] parent '{p}' of '{c}' not in nodes/roots")
            assert p in ["floor", "wall", "ceiling"], f"add floor-wall parent '{p}'"
    for node in parent_of:
        visited = set()
        cur = node
        while cur in parent_of:
            if cur in visited:
                raise ValueError(f"Cycle detected at '{cur}'")
            visited.add(cur)
            cur = parent_of[cur]
    _logger.info(f"[SceneTree] roots={sorted(roots)}, "
                 f"nodes={len(tree.get('nodes', []))}, edges={len(edges)}")
    return roots, dict(children), parent_of, node_info


def split_to_fixed_movable_set(node_info, parent_of, roots):
    fixed = set(roots)
    movable = set()
    for child, info in node_info.items():
        if (info["type"] == "fixed"
                or info["relation"] == "hang"
                or parent_of.get(child) == "wall"
                or parent_of.get(child) == "ceiling"):
            fixed.add(child)
        else:
            movable.add(child)
    return fixed, movable


def split_to_fixed_movable_set_based_on_parent(node_info, parent_of, roots):
    fixed = set(roots)
    movable = set()
    for child, info in node_info.items():
        if (info["relation"] == "hang"
                or parent_of.get(child) == "wall"
                or parent_of.get(child) == "ceiling"):
            fixed.add(child)
        else:
            movable.add(child)
    return fixed, movable


def build_subtree_groups(children, movable_set, roots):
    groups = []

    def dfs(node):
        for child in children.get(node, []):
            dfs(child)
        direct_movable = [c for c in children.get(node, []) if c in movable_set]
        if node not in roots and direct_movable:
            groups.append([node] + direct_movable)

    for root in roots:
        dfs(root)

    return groups


def local_group_movable_support_ancestors(layout, root_name):
    out = []
    seen = set()
    cur = root_name
    while cur in layout.parent_of:
        p = layout.parent_of[cur]
        if p in layout.roots:
            break
        if p in layout.movable_set and p not in seen:
            out.append(p)
            seen.add(p)
        cur = p
    return out


class SceneLayout:
    def __init__(self, args):
        self.scene_canon_dir  = args.scene_canon_dir
        self.urdf_dir  = os.path.join(args.scene_canon_dir, "urdf_files")
        self.obj_dir   = os.path.join(args.scene_canon_dir, "obj_files")
        self.scene_tree_path = os.path.join(
            os.path.dirname(os.path.abspath(args.scene_canon_dir)), "scene_tree.json")

        self.roots, self.children, self.parent_of, self.node_info = \
            parse_scene_tree(self.scene_tree_path)

        if args.use_fixed_type:
            self.fixed_set, self.movable_set = split_to_fixed_movable_set(
                self.node_info, self.parent_of, self.roots)
        else:
            self.fixed_set, self.movable_set = split_to_fixed_movable_set_based_on_parent(
                self.node_info, self.parent_of, self.roots)

        self.groups = build_subtree_groups(self.children, self.movable_set, self.roots)
        self.file_prefix = detect_file_prefix(self.urdf_dir, list(self.node_info.keys()))

        self.all_fixed_names = [
            n for n in self.fixed_set
            if n not in self.roots
            and os.path.isfile(os.path.join(self.urdf_dir, f"{self.file_prefix}{n}.urdf"))
        ]
        self.all_movable_names = [
            n for n in self.movable_set
            if os.path.isfile(os.path.join(self.urdf_dir, f"{self.file_prefix}{n}.urdf"))
        ]

        self.mesh_info = {}
        scene_min_y = np.inf
        for name in self.all_fixed_names + self.all_movable_names:
            op = os.path.join(self.obj_dir, f"{self.file_prefix}{name}.obj")
            if os.path.isfile(op):
                info = compute_mesh_info(op)
                self.mesh_info[name] = info
                scene_min_y = min(scene_min_y, info[2][1])

        self.base_dy = 0.0
        if args.ground_scene and scene_min_y < np.inf:
            self.base_dy = -float(scene_min_y) + 0.002

        self.ref_pos = {
            name: centroid + np.array([0.0, self.base_dy, 0.0])
            for name, (centroid, _, _, _) in self.mesh_info.items()
        }
        self.ref_root_pos = {
            name: np.array([0.0, self.base_dy, 0.0])
            for name in self.mesh_info
        }

        _logger.info(
            f"[SceneLayout] {len(self.groups)} groups, "
            f"{len(self.all_fixed_names)} fixed, {len(self.all_movable_names)} movable, "
            f"base_dy={self.base_dy:.4f}")
