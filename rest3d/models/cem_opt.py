"""Isaac Gym CEM optimizer for stable scene optimization."""
import json
import logging
import math
import os
import time

from isaacgym import gymapi, gymtorch
import numpy as np
import torch
import trimesh

from rest3d.utils.coll_det import (
    quat_to_rotmat_batch,
    compute_geo_pen_batch,
    compute_geo_pair_flags_single,
)
from rest3d.utils.mesh import compute_mesh_info, load_trimesh_any
from rest3d.utils.quat import (
    IDENTITY_QUAT,
    IDENTITY_QUAT_T,
    quat_to_rotmat,
    quat_multiply,
    axis_angle_to_quat_batch,
    quat_angle_distance_batch,
)
from rest3d.models.scene_layout import (
    SceneLayout,
    detect_file_prefix,
    parse_scene_tree,
    split_to_fixed_movable_set,
    split_to_fixed_movable_set_based_on_parent,
    build_subtree_groups,
)

_logger = logging.getLogger("stable_scene_opt")

STAGING_Y  = 100.0
SLEEP_TIME = 0


# ===================================================================
# Isaac Gym sim setup
# ===================================================================

def set_viewer_camera_to_scene(gym, viewer, center, radius,
                               scene_forward=np.array([0, 0, -1]),
                               up=np.array([0, 1, 0]),
                               flip180=False):
    scene_forward = scene_forward / np.linalg.norm(scene_forward)
    up = up / np.linalg.norm(up)
    k = 2.5 * float(radius)
    center = np.asarray(center, dtype=np.float32)
    cam_target = gymapi.Vec3(float(center[0]), float(center[1]), float(center[2]))
    cam_pos_np = np.array([
        center[0] - k * scene_forward[0],
        center[1] + 0.35 * k,
        center[2] - k * scene_forward[2],
    ], dtype=np.float32)
    if flip180:
        cam_pos_np[0] = 2.0 * center[0] - cam_pos_np[0]
        cam_pos_np[2] = 2.0 * center[2] - cam_pos_np[2]
    cam_pos = gymapi.Vec3(float(cam_pos_np[0]), float(cam_pos_np[1]), float(cam_pos_np[2]))
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    return cam_pos, cam_target


def create_sim_and_viewer(ig, headless=False, max_gpu_contact_pairs=None,
                          viewer_width=1920, viewer_height=1080,
                          num_position_iterations=6,
                          max_depenetration_velocity=5):
    params = gymapi.SimParams()
    params.dt = 1 / 60
    params.substeps = 2
    params.up_axis = gymapi.UP_AXIS_Y
    params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)
    params.use_gpu_pipeline = True
    params.physx.use_gpu = True
    params.physx.solver_type = 1
    params.physx.num_position_iterations = num_position_iterations
    params.physx.num_velocity_iterations = 1
    params.physx.contact_offset = 0.01
    params.physx.rest_offset = 0.0
    params.physx.max_depenetration_velocity = max_depenetration_velocity
    params.physx.contact_collection = gymapi.CC_LAST_SUBSTEP
    if max_gpu_contact_pairs is None:
        max_gpu_contact_pairs = 8 * 1024 * 1024
    params.physx.max_gpu_contact_pairs = max_gpu_contact_pairs
    _mb = max_gpu_contact_pairs * 256 // 1024 // 1024
    _logger.info(f"[Sim] max_gpu_contact_pairs = {max_gpu_contact_pairs} (~{_mb} MB GPU est.)")
    sim = ig.create_sim(0, 0, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("Failed to create sim")
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 1, 0)
    plane.static_friction = 1.0
    plane.dynamic_friction = 1.0
    plane.restitution = 0.0
    ig.add_ground(sim, plane)
    viewer = None
    if not headless:
        _viewer_props = gymapi.CameraProperties()
        _viewer_props.width = int(viewer_width)
        _viewer_props.height = int(viewer_height)
        viewer = ig.create_viewer(sim, _viewer_props)
        if viewer is None:
            raise RuntimeError("Failed to create viewer")
    return sim, viewer


def load_object_assets(args, ig, sim, urdf_dir, obj_names, fixed_set, file_prefix=""):
    assets = {}
    for name in obj_names:
        urdf_fname = f"{file_prefix}{name}.urdf"
        fpath = os.path.join(urdf_dir, urdf_fname)
        if not os.path.isfile(fpath):
            urdf_fname = f"{name}.urdf"
            fpath = os.path.join(urdf_dir, urdf_fname)
        if not os.path.isfile(fpath):
            _logger.info(f"[WARN] URDF missing: {fpath}")
            continue
        is_fixed = name in fixed_set
        opt = gymapi.AssetOptions()
        opt.fix_base_link         = is_fixed or args.static
        opt.disable_gravity       = is_fixed
        opt.collapse_fixed_joints = True
        opt.override_com          = not args.no_override_com
        opt.override_inertia      = not args.no_override_com
        opt.linear_damping        = args.linear_damping
        opt.angular_damping       = args.angular_damping
        opt.vhacd_enabled         = args.vhacd_enabled
        if args.vhacd_enabled:
            n_lower = name.lower()
            if any(k in n_lower for k in ("plant", "vase", "flower", "tree", "curtain", "leaf")):
                max_hulls = 32
            elif any(k in n_lower for k in ("box", "tray", "cabinet", "shelf", "table", "desk",
                                             "chair", "sofa", "bed", "wall", "floor", "ceiling")):
                max_hulls = 8
            else:
                max_hulls = 16
            opt.vhacd_params.max_convex_hulls = max_hulls
            opt.vhacd_params.resolution = 100_000
        asset = ig.load_asset(sim, urdf_dir, urdf_fname, opt)
        assets[name] = asset
        _logger.info(f"  asset {name}  fixed={is_fixed}"
                     f"  vhacd_hulls={max_hulls if args.vhacd_enabled else 'off'}")
    return assets


def create_actors(ig, env, assets, fixed_names, movable_obj_names, base_dy, env_idx=0):
    actors = {}
    for name in fixed_names:
        if name not in assets:
            continue
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, float(base_dy), 0.0)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        actors[name] = ig.create_actor(env, assets[name], pose, name, env_idx, 0)
    for name in movable_obj_names:
        if name not in assets:
            continue
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, float(STAGING_Y), 0.0)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        actors[name] = ig.create_actor(env, assets[name], pose, name, env_idx, 0)
    return actors


# ===================================================================
# CEM Optimiser
# ===================================================================

class CEMOptimizer:
    def __init__(self, args, n_objects=1):
        self.args       = args
        self.n_objects  = n_objects
        self.act_dim    = n_objects * args.act_dim
        self.pop_size   = args.cem_pop_size
        self.n_elite    = max(1, int(args.cem_pop_size * args.cem_elite_frac))
        self.cem_iters_joint    = args.cem_iters_joint
        self.cem_iters_subtree  = args.cem_iters_subtree
        self.reward_threshold   = getattr(args, "reward_threshold", -0.01)
        self.keep_best          = getattr(args, "keep_best", False)
        self.update_use_only_best = getattr(args, "update_use_only_best", False)
        self.std_update_mode    = getattr(args, "std_update_mode", "topk_std")
        self.decay_std_rate     = getattr(args, "decay_std_rate", 0.95)
        self.cem_final_best     = getattr(args, "cem_final_best", False)
        init_mean_6 = np.array([
            getattr(args, "init_trans_x_mean",   0.0),
            getattr(args, "init_trans_y_mean",   0.0),
            getattr(args, "init_trans_z_mean",   0.0),
            getattr(args, "init_rot_roll_mean",  0.0),
            getattr(args, "init_rot_pitch_mean", 0.0),
            getattr(args, "init_rot_yaw_mean",   0.0),
        ], dtype=np.float32)
        init_std_6 = np.array([
            getattr(args, "init_trans_x_std",   0.5),
            getattr(args, "init_trans_y_std",   0.1),
            getattr(args, "init_trans_z_std",   0.5),
            getattr(args, "init_rot_roll_std",  0.5),
            getattr(args, "init_rot_pitch_std", 0.5),
            getattr(args, "init_rot_yaw_std",   0.5),
        ], dtype=np.float32)
        self._init_mean = np.tile(init_mean_6, n_objects)
        self._init_std  = np.tile(init_std_6,  n_objects)
        self._global_best_reward = -np.inf
        self._best_action = None
        self._best_reward = -np.inf
        self.reset()

    def reset(self):
        self.mean = self._init_mean.copy()
        self.std  = self._init_std.copy()
        self._last_elites = None
        self._prev_elites = None

    def warm_start(self, mode, alpha=0.8):
        prev_mean   = self.mean.copy()
        prev_std    = self.std.copy()
        prev_elites = self._last_elites
        self._last_elites = None
        if mode == "prev_mean":
            self.mean = prev_mean
            self.std  = self._init_std.copy()
            self._prev_elites = None
        elif mode == "prev_mean_std":
            self.mean = prev_mean
            self.std  = prev_std
            self._prev_elites = None
        elif mode == "momentum":
            self.mean = alpha * prev_mean + (1.0 - alpha) * self._init_mean
            self.std  = self._init_std.copy()
            self._prev_elites = None
        elif mode == "icem":
            self.mean = prev_mean
            self.std  = prev_std
            self._prev_elites = prev_elites
        elif mode == "warm_start_best":
            if self._best_action is not None:
                self.mean = self._best_action.copy()
            else:
                self.mean = prev_mean
            self.std  = self._init_std.copy()
            self._prev_elites = None

    def sample(self, n=None):
        n = n if n is not None else self.pop_size
        samples = np.random.randn(n, self.act_dim) * self.std + self.mean
        prev = self._prev_elites
        if prev is not None:
            n_carry = min(len(prev), n)
            samples[-n_carry:] = prev[-n_carry:]
            self._prev_elites = None
        return samples

    def _nes_rank_utilities(self, n):
        return (np.arange(n, dtype=np.float32) - (n - 1) * 0.5) / n

    def update(self, samples, rewards):
        idx_best = int(np.argmax(rewards))
        if rewards[idx_best] > self._best_reward:
            self._best_reward = float(rewards[idx_best])
            self._best_action = samples[idx_best].copy()
        update_mode = getattr(self.args, "cem_update_mode", "cem")
        if update_mode == "nes":
            if self.keep_best and self._best_action is not None:
                aug_samples = np.vstack([samples, self._best_action[None]])
                aug_rewards = np.append(rewards, self._best_reward)
            else:
                aug_samples = samples
                aug_rewards = rewards
            n_aug = len(aug_samples)
            sort_idx = np.argsort(aug_rewards)
            utils    = self._nes_rank_utilities(n_aug)
            u = np.empty(n_aug, dtype=np.float32)
            u[sort_idx] = utils
            d = self.act_dim
            eta_mu    = getattr(self.args, "nes_lr_mu", 1.0)
            eta_sigma = getattr(self.args, "nes_lr_sigma",
                                (3.0 + np.log(d)) / (5.0 * np.sqrt(d)))
            eps = (aug_samples - self.mean) / (self.std + 1e-8)
            grad_mu    = (u[:, None] * eps).mean(axis=0)
            grad_sigma = (u[:, None] * (eps ** 2 - 1.0)).mean(axis=0)
            self.mean = self.mean + eta_mu * self.std * grad_mu
            self.std  = np.maximum(self.std * np.exp(eta_sigma * 0.5 * grad_sigma), 1e-5)
            elite_idx = np.argpartition(rewards, -self.n_elite)[-self.n_elite:]
            self._last_elites = samples[elite_idx].copy()
        else:
            elite_idx = np.argpartition(rewards, -self.n_elite)[-self.n_elite:]
            elites = samples[elite_idx].copy()
            if self.keep_best and self._best_action is not None:
                worst = np.argmin(rewards[elite_idx])
                elites[worst] = self._best_action
            if self.update_use_only_best:
                self.mean = elites[-1].copy()
                if self.std_update_mode == "same_std":
                    pass
                elif self.std_update_mode == "decay_std":
                    self.std = self.std * self.decay_std_rate
                else:
                    self.std = elites.std(axis=0) + 1e-5
            else:
                self.mean = elites.mean(axis=0)
                self.std  = elites.std(axis=0) + 1e-5
            self._last_elites = elites.copy()

    def converged(self, rewards):
        return float(np.max(rewards)) > self.reward_threshold

    def get_final_action(self):
        if self._best_action is not None:
            return self._best_action.copy(), self.mean.copy()
        return None, self.mean.copy()


# ===================================================================
# Result saving / export
# ===================================================================

def place_all_descendants(name, pos, quat, group_child_offsets, group_root_to_gidx, poses):
    g_idx = group_root_to_gidx.get(name)
    if g_idx is None or g_idx not in group_child_offsets:
        return
    R = quat_to_rotmat(quat)
    for c_name, (c_ofs, c_quat_rel) in group_child_offsets[g_idx].items():
        c_pos        = pos + R @ c_ofs
        c_quat_world = quat_multiply(quat, c_quat_rel)
        poses[c_name] = {"pos": c_pos.tolist(), "rot": c_quat_world.tolist(), "lin_vel": [0., 0., 0.]}
        place_all_descendants(c_name, c_pos, c_quat_world,
                               group_child_offsets, group_root_to_gidx, poses)


def save_global_poses_json(env, entity_names, entity_children, pos_np, quat_np, output_path=None):
    poses = {}
    for i, name in enumerate(entity_names):
        poses[name] = {"pos": pos_np[i].tolist(), "rot": quat_np[i].tolist(), "lin_vel": [0., 0., 0.]}
        root_R = quat_to_rotmat(quat_np[i])
        for (c_name, c_ofs, c_quat) in entity_children[i]:
            c_pos_world  = pos_np[i] + root_R @ c_ofs
            c_quat_world = quat_multiply(quat_np[i], c_quat)
            poses[c_name] = {"pos": c_pos_world.tolist(), "rot": c_quat_world.tolist(),
                             "lin_vel": [0., 0., 0.]}
    data = {"base_dy": env.base_dy, "objects": poses}
    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        _logger.info(f"[global] poses -> {output_path} ({len(poses)} objects)")
    return data


def save_subtree_poses(env, names, output_path):
    poses = {}
    for name in names:
        if name not in env.actor_indices_sim[0]:
            continue
        pos, rot, lv, _ = env.get_master_state(name)
        poses[name] = {"pos": pos.tolist(), "rot": rot.tolist(), "lin_vel": lv.tolist()}
    with open(output_path, "w") as f:
        json.dump({"base_dy": env.base_dy, "objects": poses}, f, indent=2)
    _logger.info(f"[Save] poses -> {output_path} ({len(poses)} objects)")


def export_subtree_as_obj(env, names, out_dir, label=""):
    os.makedirs(out_dir, exist_ok=True)
    result = {}
    for name in names:
        obj_path = os.path.join(env.obj_dir, f"{env.file_prefix}{name}.obj")
        if not os.path.isfile(obj_path):
            continue
        if name not in env.actor_indices_sim[0]:
            continue
        mesh = load_trimesh_any(obj_path)
        pos, rot, _, _ = env.get_master_state(name)
        T = np.eye(4)
        T[:3, :3] = quat_to_rotmat(rot)
        T[:3, 3] = pos
        mesh.apply_transform(T)
        result[name] = mesh
        out_name = f"{label}_{name}.obj" if label else f"{name}.obj"
        mesh.export(os.path.join(out_dir, out_name))
    if result:
        combined = trimesh.util.concatenate(list(result.values()))
        vmin = combined.vertices.min(axis=0)
        vmax = combined.vertices.max(axis=0)
        _logger.info(f"[ExportSubtree] '{label}' -> {out_dir}/ "
                     f"({len(result)} objs)  Y range [{vmin[1]:.3f}, {vmax[1]:.3f}]")
        for name, m in result.items():
            p, r, _, _ = env.get_master_state(name)
            ymin_m = m.vertices[:, 1].min()
            ymax_m = m.vertices[:, 1].max()
            _logger.info(f"    {name[:50]:50s}  root_y={p[1]:.4f}  "
                         f"mesh_Y=[{ymin_m:.3f},{ymax_m:.3f}]")
    return result


# ===================================================================
# Multi-env CEM placement (Isaac Gym)
# ===================================================================

class MultiEnvCEMPlacement:
    def __init__(self, args):
        self.args          = args
        self.n_envs        = args.cem_pop_size
        self.scene_canon_dir      = args.scene_canon_dir
        self.urdf_dir      = getattr(args, "urdf_dir_override", None) or \
                             os.path.join(self.scene_canon_dir, "urdf_files")
        self.obj_dir       = os.path.join(self.scene_canon_dir, "obj_files")
        self.scene_tree_path = os.path.join(
            os.path.dirname(os.path.abspath(self.scene_canon_dir)), "scene_tree.json")
        self.total_settle_steps = args.total_settle_steps
        self.vel_settle_steps   = args.vel_settle_steps
        self.lambda_vel         = args.lambda_vel
        self.lambda_pose_stab         = args.lambda_pose_stab
        self.lambda_pose_layout       = args.lambda_pose_layout
        self.lambda_rot_stab          = args.lambda_rot_stab
        self.lambda_rot_layout        = args.lambda_rot_layout
        self.lambda_settled_geo_pen   = args.lambda_settled_geo_pen

        layout: SceneLayout = getattr(args, "scene_layout", None)
        if layout is not None:
            self.roots        = layout.roots
            self.children     = layout.children
            self.parent_of    = layout.parent_of
            self.node_info    = layout.node_info
            self.fixed_set    = layout.fixed_set
            self.movable_set  = layout.movable_set
            self.groups       = layout.groups
            self.file_prefix  = layout.file_prefix
            self.mesh_info    = layout.mesh_info
            self.base_dy      = layout.base_dy
            self.ref_pos      = layout.ref_pos
            self.ref_root_pos = layout.ref_root_pos
        else:
            self.roots, self.children, self.parent_of, self.node_info = \
                parse_scene_tree(self.scene_tree_path)
            self.fixed_set, self.movable_set = (
                split_to_fixed_movable_set(self.node_info, self.parent_of, self.roots)
                if args.use_fixed_type else
                split_to_fixed_movable_set_based_on_parent(
                    self.node_info, self.parent_of, self.roots))
            self.groups      = build_subtree_groups(self.children, self.movable_set, self.roots)
            self.file_prefix = detect_file_prefix(self.urdf_dir, list(self.node_info.keys()))
            _all_fixed   = [n for n in self.fixed_set
                            if n not in self.roots
                            and os.path.isfile(
                                os.path.join(self.urdf_dir, f"{self.file_prefix}{n}.urdf"))]
            _all_movable = [n for n in self.movable_set
                            if os.path.isfile(
                                os.path.join(self.urdf_dir, f"{self.file_prefix}{n}.urdf"))]
            self.mesh_info = {}
            scene_min_y = np.inf
            for name in _all_fixed + _all_movable:
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

        _logger.info(f"[Fixed set]   {self.fixed_set}")
        _logger.info(f"[Movable set] {self.movable_set}")
        _logger.info(f"[Ground] base_dy = {self.base_dy:.4f}")

        _actor_filter = getattr(args, "actor_filter", None)

        self.fixed_obj_names = [
            n for n in self.fixed_set
            if n not in self.roots
            and os.path.isfile(os.path.join(self.urdf_dir, f"{self.file_prefix}{n}.urdf"))
            and (_actor_filter is None or n in _actor_filter)
        ]
        self.movable_obj_names = [
            n for n in self.movable_set
            if os.path.isfile(os.path.join(self.urdf_dir, f"{self.file_prefix}{n}.urdf"))
            and (_actor_filter is None or n in _actor_filter)
        ]

        _logger.info(f"[RefRootPos] all (0, {self.base_dy:.4f}, 0) for "
                     f"{len(self.ref_root_pos)} objects")
        self.actor_ref_pos = self.ref_root_pos

        self.ig = gymapi.acquire_gym()
        _n_actors = len(self.fixed_obj_names) + len(self.movable_obj_names)
        _n_actor_pairs = max(1, _n_actors * (_n_actors - 1) // 2)
        _n_envs = getattr(args, "cem_pop_size", 1024)
        _pt = 64 if getattr(args, "vhacd_enabled", False) else 40
        _est = _n_envs * _n_actor_pairs * _pt
        _floor = 512 * 1024
        _cap = 8 * 1024 * 1024
        _contact_pairs = max(_floor, min(_cap, int(_est * 1.15) + 65536))
        self.sim, self.viewer = create_sim_and_viewer(
            self.ig, args.headless, max_gpu_contact_pairs=_contact_pairs,
            viewer_width=1920, viewer_height=1080,
            num_position_iterations=getattr(args, "num_position_iterations", 6),
            max_depenetration_velocity=getattr(args, "max_depenetration_velocity", 5))

        _loaded_mesh = {k: v for k, v in self.mesh_info.items()
                        if k in self.fixed_obj_names or k in self.movable_obj_names}
        if _loaded_mesh:
            all_mins = np.array([v[2] for v in _loaded_mesh.values()])
            all_maxs = np.array([v[3] for v in _loaded_mesh.values()])
            ext = all_maxs.max(axis=0) - all_mins.min(axis=0)
            self.env_half = 0.5 * max(ext[0], ext[2]) + 0.5
        else:
            self.env_half = 5.0

        all_names = self.fixed_obj_names + self.movable_obj_names
        _logger.info("\n[Assets]")
        self.assets = load_object_assets(
            args, self.ig, self.sim, self.urdf_dir,
            all_names, self.fixed_set, file_prefix=self.file_prefix)

        s = float(self.env_half)
        num_per_row = int(math.ceil(math.sqrt(self.n_envs)))
        self.envs = []
        self.all_actors = []
        for i in range(self.n_envs):
            env_i = self.ig.create_env(
                self.sim,
                gymapi.Vec3(-s, -s, -s),
                gymapi.Vec3(s, s, s),
                num_per_row)
            actors_i = create_actors(
                self.ig, env_i, self.assets,
                self.fixed_obj_names, self.movable_obj_names,
                self.base_dy, env_idx=i)
            self.envs.append(env_i)
            self.all_actors.append(actors_i)
        self.actor_names = list(self.all_actors[0].keys())
        _logger.info(f"[Sim] {self.n_envs} envs x {len(self.actor_names)} actors each")

        self.ig.prepare_sim(self.sim)
        _root_t = self.ig.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_root_t)
        self._device = self.root_states.device
        self.ig.refresh_actor_root_state_tensor(self.sim)
        _logger.info(f"[Tensor] root_states {tuple(self.root_states.shape)} on {self._device}")

        self.actor_indices_sim = []
        for i in range(self.n_envs):
            idx_map = {
                name: self.ig.get_actor_index(self.envs[i], handle, gymapi.DOMAIN_SIM)
                for name, handle in self.all_actors[i].items()
            }
            self.actor_indices_sim.append(idx_map)

        self.name_idx_long = {}
        self.name_idx_i32  = {}
        for name in self.actor_names:
            idx = torch.tensor(
                [self.actor_indices_sim[i][name] for i in range(self.n_envs)],
                dtype=torch.long, device=self._device)
            self.name_idx_long[name] = idx
            self.name_idx_i32[name]  = idx.to(torch.int32)

        _cf_t = self.ig.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(_cf_t)

        self.name_rb_idx_long = {}
        for name in self.actor_names:
            rb_indices = [
                self.ig.get_actor_rigid_body_index(
                    self.envs[i], self.all_actors[i][name], 0, gymapi.DOMAIN_SIM)
                for i in range(self.n_envs)
            ]
            self.name_rb_idx_long[name] = torch.tensor(
                rb_indices, dtype=torch.long, device=self._device)

        self.env_idx = []
        for i in range(self.n_envs):
            indices = [self.actor_indices_sim[i][n] for n in self.actor_names]
            self.env_idx.append(
                torch.tensor(indices, dtype=torch.long, device=self._device))
        self.master_idx = self.env_idx[0]

        self.ig.refresh_actor_root_state_tensor(self.sim)
        first_name = self.actor_names[0]
        ref_pos_0 = self.root_states[self.actor_indices_sim[0][first_name], 0:3].clone()
        self.env_offsets = torch.zeros(self.n_envs, 3, device=self._device)
        for i in range(1, self.n_envs):
            pos_i = self.root_states[self.actor_indices_sim[i][first_name], 0:3]
            self.env_offsets[i] = pos_i - ref_pos_0

        if self.viewer is not None and len(self.mesh_info) > 0:
            all_mins = np.array([v[2] for v in self.mesh_info.values()]) \
                        + np.array([0.0, self.base_dy, 0.0])
            all_maxs = np.array([v[3] for v in self.mesh_info.values()]) \
                        + np.array([0.0, self.base_dy, 0.0])
            center = 0.5 * (all_mins.min(axis=0) + all_maxs.max(axis=0))
            radius = float(np.linalg.norm(
                all_maxs.max(axis=0) - all_mins.min(axis=0))) * 0.5 + 0.5
            set_viewer_camera_to_scene(
                self.ig, self.viewer, center, radius,
                flip180=getattr(self.args, "flip180", False))

        self._pinned_actors = {}
        self._pin_idx_long  = None
        self._pin_idx_i32   = None
        self._pin_pos       = None
        self._pin_rot       = None
        self.placed_names   = []

        self.orig_world_pos  = {}
        self.orig_world_quat = {}
        for name in self.movable_obj_names:
            self.orig_world_pos[name]  = torch.tensor(
                self.ref_pos.get(name, [0., self.base_dy, 0.]), dtype=torch.float32)
            self.orig_world_quat[name] = IDENTITY_QUAT_T.clone()

        self.hull_verts = {}
        for name in self.movable_obj_names:
            op = os.path.join(self.obj_dir, f"{self.file_prefix}{name}.obj")
            if os.path.isfile(op):
                hull_v = load_trimesh_any(op).convex_hull.vertices.astype(np.float32)
                self.hull_verts[name] = torch.from_numpy(hull_v).to(self._device)

        self.settle_all(30)
        self.initial_master_state = self.save_master_state()

    def set_object_pose_master(self, name, pos_local, rot):
        self._write_object_master_to_tensor(name, pos_local, rot)
        self._push_root_states()

    def _write_object_master_to_tensor(self, name, pos_local, rot):
        idx = self.actor_indices_sim[0][name]
        self.root_states[idx, 0:3] = torch.tensor(pos_local, dtype=torch.float32,
                                                   device=self._device)
        self.root_states[idx, 3:7] = torch.tensor(rot, dtype=torch.float32,
                                                   device=self._device)
        self.root_states[idx, 7:13] = 0.0

    def _push_root_states(self):
        self.ig.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def settle_all(self, steps, render=False):
        for _ in range(steps):
            self.ig.simulate(self.sim)
            self.ig.fetch_results(self.sim, True)
            if self._pinned_actors:
                self._repin_actors()
            if render and self.viewer is not None:
                self.ig.step_graphics(self.sim)
                self.ig.draw_viewer(self.viewer, self.sim, True)
                time.sleep(SLEEP_TIME)

    def sim_step(self, render=False):
        self.ig.simulate(self.sim)
        self.ig.fetch_results(self.sim, True)
        if render and self.viewer is not None:
            self.ig.step_graphics(self.sim)
            self.ig.draw_viewer(self.viewer, self.sim, True)
            time.sleep(SLEEP_TIME)

    def save_master_state(self):
        self.ig.refresh_actor_root_state_tensor(self.sim)
        return self.root_states[self.master_idx].clone()

    def get_master_state(self, name):
        if name not in self.actor_indices_sim[0]:
            return np.zeros(3), IDENTITY_QUAT, np.zeros(3), np.zeros(3)
        idx = self.actor_indices_sim[0][name]
        s = self.root_states[idx]
        pos  = (s[0:3] - self.env_offsets[0]).cpu().numpy()
        quat = s[3:7].cpu().numpy()
        lv   = s[7:10].cpu().numpy()
        av   = s[10:13].cpu().numpy()
        return pos, quat, lv, av

    def freeze_actor(self, name):
        self.ig.refresh_actor_root_state_tensor(self.sim)
        idx = self.name_idx_long[name]
        self._pinned_actors[name] = (
            self.root_states[idx, 0:3].clone(),
            self.root_states[idx, 3:7].clone())
        self._rebuild_pin_cache()

    def unfreeze_actor(self, name):
        self._pinned_actors.pop(name, None)
        self._rebuild_pin_cache()

    def freeze_actors_batch(self, names):
        names = [n for n in names if n in self.name_idx_long]
        if not names:
            return
        self.ig.refresh_actor_root_state_tensor(self.sim)
        for name in names:
            idx = self.name_idx_long[name]
            self._pinned_actors[name] = (
                self.root_states[idx, 0:3].clone(),
                self.root_states[idx, 3:7].clone())
        self._rebuild_pin_cache()

    def unfreeze_actors_batch(self, names):
        for name in names:
            self._pinned_actors.pop(name, None)
        self._rebuild_pin_cache()

    def disable_simulation_batch(self, names):
        flag = gymapi.RIGID_BODY_DISABLE_SIMULATION
        for env_h, actors in zip(self.envs, self.all_actors):
            for name in names:
                if name not in actors:
                    continue
                ah = actors[name]
                props = self.ig.get_actor_rigid_body_properties(env_h, ah)
                for p in props:
                    p.flags = flag
                self.ig.set_actor_rigid_body_properties(env_h, ah, props, False)

    def enable_simulation_batch(self, names):
        for env_h, actors in zip(self.envs, self.all_actors):
            for name in names:
                if name not in actors:
                    continue
                ah = actors[name]
                props = self.ig.get_actor_rigid_body_properties(env_h, ah)
                for p in props:
                    p.flags = gymapi.RIGID_BODY_NONE
                self.ig.set_actor_rigid_body_properties(env_h, ah, props, False)

    def lock_staging_batch(self, names):
        _stg = torch.tensor([0., STAGING_Y, 0.], dtype=torch.float32)
        _id_q = IDENTITY_QUAT_T
        names_valid = [n for n in names if n in self.name_idx_long]
        for name in names_valid:
            idx = self.name_idx_long[name]
            self.root_states[idx, 0:3] = _stg.to(self._device) + self.env_offsets
            self.root_states[idx, 3:7] = _id_q.to(self._device)
            self.root_states[idx, 7:13] = 0.0
        if names_valid:
            all_idx = torch.cat([self.name_idx_long[n] for n in names_valid])
            all_idx_i32 = all_idx.to(torch.int32)
            self.ig.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(all_idx_i32), len(all_idx_i32))
        flag = gymapi.RIGID_BODY_DISABLE_GRAVITY
        for env_h, actors in zip(self.envs, self.all_actors):
            for name in names:
                if name not in actors:
                    continue
                ah = actors[name]
                props = self.ig.get_actor_rigid_body_properties(env_h, ah)
                for p in props:
                    p.flags = flag
                self.ig.set_actor_rigid_body_properties(env_h, ah, props, False)

    def unlock_staging_batch(self, names):
        for env_h, actors in zip(self.envs, self.all_actors):
            for name in names:
                if name not in actors:
                    continue
                ah = actors[name]
                props = self.ig.get_actor_rigid_body_properties(env_h, ah)
                for p in props:
                    p.flags = gymapi.RIGID_BODY_NONE
                self.ig.set_actor_rigid_body_properties(env_h, ah, props, False)

    def _rebuild_pin_cache(self):
        if not self._pinned_actors:
            self._pin_idx_long = self._pin_idx_i32 = None
            self._pin_pos = self._pin_rot = None
            return
        idx_parts, pos_parts, rot_parts = [], [], []
        for nm, (pos, rot) in self._pinned_actors.items():
            idx_parts.append(self.name_idx_long[nm])
            pos_parts.append(pos)
            rot_parts.append(rot)
        self._pin_idx_long = torch.cat(idx_parts)
        self._pin_idx_i32  = self._pin_idx_long.to(torch.int32)
        self._pin_pos      = torch.cat(pos_parts)
        self._pin_rot      = torch.cat(rot_parts)

    def _repin_actors(self):
        if self._pin_idx_long is None:
            return
        self.ig.refresh_actor_root_state_tensor(self.sim)
        self.root_states[self._pin_idx_long, 0:3]  = self._pin_pos
        self.root_states[self._pin_idx_long, 3:7]  = self._pin_rot
        self.root_states[self._pin_idx_long, 7:13] = 0.0
        self.ig.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(self._pin_idx_i32),
            len(self._pin_idx_i32))

    def place_all_at_ref(self):
        _id_rot = torch.tensor([0., 0., 0., 1.], dtype=torch.float32)
        for name in self.movable_obj_names:
            if name not in self.name_idx_long:
                continue
            ref_p = torch.tensor(
                self.actor_ref_pos.get(name, [0., self.base_dy, 0.]),
                dtype=torch.float32)
            pos = ref_p.unsqueeze(0).expand(self.n_envs, -1)
            rot = _id_rot.unsqueeze(0).expand(self.n_envs, -1)
            idx = self.name_idx_long[name]
            self.root_states[idx, 0:3] = pos.to(self._device) + self.env_offsets
            self.root_states[idx, 3:7] = rot.to(self._device)
            self.root_states[idx, 7:13] = 0.0
        self._push_root_states()

    def set_actor_poses_all_envs(self, name, positions, quats):
        if name not in self.name_idx_long:
            return
        idx = self.name_idx_long[name]
        self.root_states[idx, 0:3] = positions.to(self._device) + self.env_offsets
        self.root_states[idx, 3:7] = quats.to(self._device)
        self.root_states[idx, 7:13] = 0.0

    def reset_all_envs(self, stage_fixed=False):
        _stg = torch.tensor([0., STAGING_Y, 0.], dtype=torch.float32)
        _id_q = IDENTITY_QUAT_T
        for name in self.actor_names:
            if name not in self.name_idx_long:
                continue
            idx = self.name_idx_long[name]
            place_at_ref = (name in self.fixed_set) and not stage_fixed
            p = (torch.tensor(self.actor_ref_pos.get(name, [0., self.base_dy, 0.]),
                               dtype=torch.float32)
                 if place_at_ref else _stg)
            pos = p.unsqueeze(0).expand(self.n_envs, -1)
            rot = _id_q.unsqueeze(0).expand(self.n_envs, -1)
            self.root_states[idx, 0:3] = pos.to(self._device) + self.env_offsets
            self.root_states[idx, 3:7] = rot.to(self._device)
            self.root_states[idx, 7:13] = 0.0
        self._push_root_states()

    def show_group(self, keep_names):
        self._write_group_staging(keep_names)
        self._push_root_states()

    def render(self):
        if self.viewer is not None:
            self.ig.step_graphics(self.sim)
            self.ig.draw_viewer(self.viewer, self.sim, True)

    def viewer_running(self):
        if self.viewer is None:
            return False
        return not self.ig.query_viewer_has_closed(self.viewer)

    def close(self):
        if self.viewer is not None:
            self.ig.destroy_viewer(self.viewer)
        self.ig.destroy_sim(self.sim)

    def _write_group_staging(self, keep_names, staging_pos_dict=None, stage_fixed=False):
        keep = set(keep_names)
        _stg_default = torch.tensor([0., STAGING_Y, 0.], dtype=torch.float32)
        _id_q = IDENTITY_QUAT_T
        for name in self.actor_names:
            if name not in self.name_idx_long:
                continue
            at_ref = (name in keep) or ((name in self.fixed_set) and not stage_fixed)
            if at_ref:
                p = torch.tensor(self.actor_ref_pos.get(name, [0., self.base_dy, 0.]),
                                 dtype=torch.float32)
            elif staging_pos_dict is not None and name in staging_pos_dict:
                p = torch.tensor(staging_pos_dict[name], dtype=torch.float32)
            else:
                p = _stg_default
            pos = p.unsqueeze(0).expand(self.n_envs, -1)
            rot = _id_q.unsqueeze(0).expand(self.n_envs, -1)
            idx = self.name_idx_long[name]
            self.root_states[idx, 0:3] = pos.to(self._device) + self.env_offsets
            self.root_states[idx, 3:7] = rot.to(self._device)
            self.root_states[idx, 7:13] = 0.0
