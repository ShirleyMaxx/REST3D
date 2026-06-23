import argparse
import glob
import io
import json
import logging
import os
import shutil
import subprocess
import sys

from rest3d.models.cem_opt import (
    CEMOptimizer,
    MultiEnvCEMPlacement,
    place_all_descendants,
    export_subtree_as_obj,
    save_global_poses_json,
    save_subtree_poses,
)

import numpy as np
import torch

import wandb

from rest3d.config.stable_scene_cfg import StableSceneCfg
from rest3d.models.scene_layout import SceneLayout, local_group_movable_support_ancestors
from rest3d.utils.coll_det import compute_geo_pen_batch, quat_to_rotmat_batch
from rest3d.utils.log import attach_phase_log
from rest3d.utils.mesh import load_trimesh_any
from rest3d.utils.scene_io import load_local_group_from_dir, load_global_from_dir
from rest3d.utils import postprocess
from rest3d.utils.quat import (
    IDENTITY_QUAT,
    axis_angle_to_quat_batch,
    local_group_child_offset_parent_frame,
    quat_angle_distance_batch,
    quat_multiply,
    quat_multiply_batch,
    quat_to_rotmat,
)



_logger = logging.getLogger("stable_scene_opt")


# ===================================================================
# local_group
# ===================================================================


def local_group_optimize(args, layout):
    _logger.info("\n[local_group] Subtree CEM optimization")
    _log_dir = os.path.join(args.output_dir, "log")
    os.makedirs(_log_dir, exist_ok=True)
    _ph = attach_phase_log(_logger, _log_dir, "local_group")
    group_child_offsets = {}
    settled_pos_registry  = {}
    settled_quat_registry = {}

    for g_idx, group in reversed(list(enumerate(layout.groups))):
        root_name   = group[0]
        child_names = group[1:]
        n_children  = len(child_names)
        _logger.info(f"\n[local_group | G{g_idx}]  Optimizing subtree  root={root_name}  children={list(child_names)}")

        if n_children == 0:
            group_child_offsets[g_idx] = {}
            continue

        _support_mov = local_group_movable_support_ancestors(layout, root_name)
        args.actor_filter = set(group) | set(_support_mov)
        if _support_mov:
            _logger.info(f"  [local_group | G{g_idx}] actor_filter adds support movables: {_support_mov}")
        args.scene_layout = layout
        env = MultiEnvCEMPlacement(args)
        render = (env.viewer is not None)

        _base_ref = np.array([0., env.base_dy, 0.], dtype=np.float32)
        root_frozen_pos  = np.array(settled_pos_registry.get(root_name, _base_ref), dtype=np.float32)
        root_frozen_quat = np.array(settled_quat_registry.get(root_name, IDENTITY_QUAT), dtype=np.float32)
        child_body_centroid_ofs = np.array(
            [env.ref_pos.get(c, [0., env.base_dy, 0.]) for c in child_names],
            dtype=np.float32) - _base_ref

        if root_name in settled_pos_registry:
            child_ref_pos     = np.stack([root_frozen_pos  for _ in child_names], axis=0)
            child_ref_quat_np = np.stack([root_frozen_quat for _ in child_names], axis=0)
        else:
            child_ref_pos     = np.array(
                [env.ref_root_pos.get(c, [0., env.base_dy, 0.]) for c in child_names],
                dtype=np.float32)
            child_ref_quat_np = np.stack([IDENTITY_QUAT for _ in child_names], axis=0)
        child_centroid_ofs = child_body_centroid_ofs

        _R_root = quat_to_rotmat(root_frozen_quat)
        orig_world_rows, orig_quat_rows = [], []
        for j, c in enumerate(child_names):
            if root_name in settled_pos_registry:
                v = child_body_centroid_ofs[j]
                orig_world_rows.append(root_frozen_pos + (_R_root @ v).astype(np.float32))
                orig_quat_rows.append(root_frozen_quat.copy())
            else:
                orig_world_rows.append(
                    np.array(env.ref_pos.get(c, [0., env.base_dy, 0.]), dtype=np.float32))
                orig_quat_rows.append(IDENTITY_QUAT.copy())
        orig_world_np = np.stack(orig_world_rows, axis=0)
        orig_quat_np  = np.stack(orig_quat_rows, axis=0)

        child_ref_t       = torch.from_numpy(child_ref_pos).float()
        child_ref_quat_t  = torch.from_numpy(child_ref_quat_np).float()
        centroid_ofs_dev  = torch.from_numpy(child_centroid_ofs).float().to(env._device)
        child_hull_dev    = [env.hull_verts.get(c) for c in child_names]
        root_hull_dev     = env.hull_verts.get(root_name)
        child_flat_idx    = torch.stack(
            [env.name_idx_long[c] for c in child_names], dim=1).view(-1)
        orig_world_dev      = torch.from_numpy(orig_world_np).float().to(env._device)
        orig_world_quat_dev = torch.from_numpy(orig_quat_np).float().to(env._device)

        _rref_t   = torch.tensor(root_frozen_pos,  dtype=torch.float32)
        _rqroot_t = torch.tensor(root_frozen_quat, dtype=torch.float32)
        env.set_actor_poses_all_envs(
            root_name,
            _rref_t.unsqueeze(0).expand(env.n_envs, -1),
            _rqroot_t.unsqueeze(0).expand(env.n_envs, -1))
        env._push_root_states()
        env.freeze_actor(root_name)

        warm_start_mode = getattr(args, "cem_warm_start", None)
        cem = CEMOptimizer(args, n_objects=n_children)

        _best_iter_num      = 0
        _best_iter_reward   = -np.inf
        _best_iter_rewards  = {}
        _best_iter_children = []
        _best_env_idx       = -1
        _best_child_raw     = None
        _best_root_raw      = None
        _saved_best_reward  = -np.inf
        _best_offsets       = None
        _best_per_child_results = None
        output_obj_dir = os.path.join(args.output_dir, "local_groups")
        _local_obj_dir = os.path.join(output_obj_dir, "obj_files")
        os.makedirs(_local_obj_dir, exist_ok=True)

        for episode in range(args.cem_episodes):
            if episode > 0:
                if warm_start_mode is None:
                    cem.reset()
                else:
                    alpha = getattr(args, "cem_warm_start_alpha", 0.8)
                    cem.warm_start(mode=warm_start_mode, alpha=alpha)
                    _logger.info(f"  [local_group | G{g_idx}] warm-start={warm_start_mode}"
                                 f"  mean_norm={np.linalg.norm(cem.mean):.4f}"
                                 f"  std_mean={cem.std.mean():.4f}")

            _logger.info(f"\n  [local_group | G{g_idx}] Episode {episode+1}/{args.cem_episodes}"
                         f"  CEM init: mean={cem.mean[:6]}  std={cem.std[:6]}")

            for cem_iter in range(args.cem_iters_subtree):
                _pop_decay = getattr(args, "cem_pop_size_decay", 0.0)
                if _pop_decay > 0.0:
                    _n_active = min(max(int(args.cem_pop_size / (_pop_decay ** cem_iter)),
                                        2 * cem.n_elite), args.cem_pop_size)
                else:
                    _n_active = args.cem_pop_size

                samples   = cem.sample()
                samples_t = torch.from_numpy(samples).float()
                s3d       = samples_t.view(env.n_envs, n_children, 6)
                delta_pos = s3d[:, :, 0:3]
                delta_aa  = s3d[:, :, 3:6]

                placed_pos_all  = child_ref_t.unsqueeze(0) + delta_pos
                _q_delta = axis_angle_to_quat_batch(delta_aa.reshape(-1, 3).to(env._device))
                _q_ref   = (child_ref_quat_t.unsqueeze(0)
                            .expand(env.n_envs, n_children, 4)
                            .reshape(-1, 4).to(env._device))
                placed_quat_all = quat_multiply_batch(_q_ref, _q_delta).view(
                    env.n_envs, n_children, 4)

                for j, c in enumerate(child_names):
                    env.set_actor_poses_all_envs(c, placed_pos_all[:, j, :], placed_quat_all[:, j, :])
                env._push_root_states()

                env.ig.refresh_actor_root_state_tensor(env.sim)
                placed_root       = (env.root_states[child_flat_idx, 0:3]
                                     .view(env.n_envs, n_children, 3)
                                     - env.env_offsets.unsqueeze(1))
                placed_world_quat = (env.root_states[child_flat_idx, 3:7]
                                     .view(env.n_envs, n_children, 4))

                _lambda_pgp = getattr(args, "lambda_place_geo_pen", 0.0)
                if _lambda_pgp > 0.0:
                    _root_flat_pre = env.name_idx_long[root_name]
                    _rpos_pre  = (env.root_states[_root_flat_pre, 0:3] - env.env_offsets).unsqueeze(1)
                    _rquat_pre = env.root_states[_root_flat_pre, 3:7].unsqueeze(1)
                    r_place_geo_pen = compute_geo_pen_batch(
                        torch.cat([_rpos_pre,  placed_root],        dim=1),
                        torch.cat([_rquat_pre, placed_world_quat],  dim=1),
                        [root_hull_dev] + child_hull_dev, env._device)
                else:
                    r_place_geo_pen = torch.zeros(env.n_envs, device=env._device)

                env.settle_all(args.vel_settle_steps, render=render)
                env.ig.refresh_actor_root_state_tensor(env.sim)
                vel_early = (env.root_states[child_flat_idx, 7:10]
                             .view(env.n_envs, n_children, 3))

                env.settle_all(args.total_settle_steps - args.vel_settle_steps, render=render)
                env.ig.refresh_actor_root_state_tensor(env.sim)

                settled_root = (env.root_states[child_flat_idx, 0:3]
                                .view(env.n_envs, n_children, 3)
                                - env.env_offsets.unsqueeze(1))
                settled_quat = (env.root_states[child_flat_idx, 3:7]
                                .view(env.n_envs, n_children, 4))
                _cofs = centroid_ofs_dev.view(1, n_children, 3, 1).expand(env.n_envs, -1, -1, -1)
                _R_placed  = quat_to_rotmat_batch(placed_world_quat.reshape(-1, 4)).view(
                    env.n_envs, n_children, 3, 3)
                _R_settled = quat_to_rotmat_batch(settled_quat.reshape(-1, 4)).view(
                    env.n_envs, n_children, 3, 3)
                placed_world  = placed_root  + torch.matmul(_R_placed,  _cofs).squeeze(-1)
                settled_world = settled_root + torch.matmul(_R_settled, _cofs).squeeze(-1)

                r_pos_stab = torch.zeros(env.n_envs, device=env._device)
                r_rot_stab = torch.zeros(env.n_envs, device=env._device)
                r_pos_layout  = torch.zeros(env.n_envs, device=env._device)
                r_rot_layout  = torch.zeros(env.n_envs, device=env._device)
                r_vel       = torch.zeros(env.n_envs, device=env._device)
                for j in range(n_children):
                    sw = settled_world[:, j, :]; sq = settled_quat[:, j, :]
                    pw = placed_world[:, j, :];  pq = placed_world_quat[:, j, :]
                    ow = orig_world_dev[j];       oq = orig_world_quat_dev[j]
                    r_pos_stab += torch.norm(sw - pw, dim=1)
                    r_rot_stab += quat_angle_distance_batch(sq, pq)
                    r_pos_layout  += torch.norm(sw - ow.unsqueeze(0), dim=1)
                    r_rot_layout  += quat_angle_distance_batch(sq, oq)
                    r_vel       += torch.norm(vel_early[:, j, :], dim=1)

                root_flat_idx = env.name_idx_long[root_name]
                root_pos_b    = (env.root_states[root_flat_idx, 0:3] - env.env_offsets).unsqueeze(1)
                root_quat_b   = env.root_states[root_flat_idx, 3:7].unsqueeze(1)
                all_settled_pos  = torch.cat([root_pos_b,  settled_root], dim=1)
                all_settled_quat = torch.cat([root_quat_b, settled_quat], dim=1)
                r_settled_geo_pen = compute_geo_pen_batch(
                    all_settled_pos, all_settled_quat,
                    [root_hull_dev] + child_hull_dev, env._device)

                r_stab   = args.lambda_pose_stab   * r_pos_stab   + args.lambda_rot_stab   * r_rot_stab
                r_layout = args.lambda_pose_layout * r_pos_layout + args.lambda_rot_layout * r_rot_layout
                r_pen    = args.lambda_settled_geo_pen * r_settled_geo_pen + args.lambda_place_geo_pen * r_place_geo_pen
                rewards  = -(r_stab + r_layout + args.lambda_vel * r_vel + r_pen)

                rewards_np = rewards.cpu().numpy()
                cem.update(samples[:_n_active], rewards_np[:_n_active])
                best_i = int(np.argmax(rewards_np[:_n_active]))
                best_r = float(rewards_np[best_i])

                if best_r > _best_iter_reward:
                    _best_iter_reward  = best_r
                    _best_iter_num     = cem_iter + 1
                    _best_iter_rewards = {
                        "stab":   float(r_stab[best_i]),
                        "layout": float(r_layout[best_i]),
                        "vel":    float(r_vel[best_i]),
                        "pen":    float(r_pen[best_i]),
                    }
                    _best_iter_children = [
                        {
                            "name":         c,
                            "aa":           s3d[best_i, j, 3:6].numpy().copy(),
                            "placed_origin":  placed_pos_all[best_i, j, :].numpy().copy(),
                            "placed_world":   placed_world[best_i, j, :].cpu().numpy().copy(),
                            "settled_origin": settled_root[best_i, j, :].cpu().numpy().copy(),
                            "settled_world":  settled_world[best_i, j, :].cpu().numpy().copy(),
                        }
                        for j, c in enumerate(child_names)
                    ]
                    _best_env_idx   = best_i
                    _best_child_raw = [
                        env.root_states[child_flat_idx[best_i * n_children + j].item(), :].clone()
                        for j in range(n_children)
                    ]
                    _best_root_raw  = env.root_states[
                        env.actor_indices_sim[best_i][root_name], :].clone()

                _logger.debug(
                    f"  Episode {episode+1}/{args.cem_episodes}"
                    f"  iter {cem_iter+1:3d}/{args.cem_iters_subtree}"
                    f"  best_reward={cem._best_reward:.4f}  cur_best={best_r:.4f}"
                    f"  stab={r_stab[best_i]:.3f}"
                    f"  layout={r_layout[best_i]:.3f}"
                    f"  vel={r_vel[best_i]:.3f}"
                    f"  pen={r_pen[best_i]:.0f}")

                if args.use_wandb:
                    _p1_wb_step = cem_iter + episode * args.cem_iters_subtree
                    wandb.log({
                        f"local_group{g_idx}/cem_iter":        _p1_wb_step,
                        f"local_group{g_idx}/best_reward":     cem._best_reward,
                        f"local_group{g_idx}/cur_best_reward": best_r,
                        f"local_group{g_idx}/best_stab":   float(r_stab[best_i]),
                        f"local_group{g_idx}/best_layout": float(r_layout[best_i]),
                        f"local_group{g_idx}/best_vel":    float(r_vel[best_i]),
                        f"local_group{g_idx}/best_pen":    float(r_pen[best_i]),
                    })

                if cem.converged(rewards_np):
                    _logger.info("    converged.")
                    break

            if cem._best_reward > _saved_best_reward:
                _saved_best_reward = cem._best_reward
                _best_per_child_results = None
                _best_offs = None
                if _best_child_raw is not None:
                    _ofs_bi = env.env_offsets[_best_env_idx]
                    _ofs_0  = env.env_offsets[0]
                    for j, c in enumerate(child_names):
                        _c_idx0 = env.actor_indices_sim[0][c]
                        _cs = _best_child_raw[j].clone()
                        _cs[0:3] = _cs[0:3] - _ofs_bi + _ofs_0
                        env.root_states[_c_idx0] = _cs
                    _r_idx0 = env.actor_indices_sim[0][root_name]
                    _rs = _best_root_raw.clone()
                    _rs[0:3] = _rs[0:3] - _ofs_bi + _ofs_0
                    env.root_states[_r_idx0] = _rs
                    env._push_root_states()
                    _best_offs = {}
                    _best_per_child_results = []
                    for j, c in enumerate(child_names):
                        _cs = _best_child_raw[j]
                        fin_pos  = (_cs[0:3] - _ofs_bi).cpu().numpy()
                        fin_quat = _cs[3:7].cpu().numpy()
                        _ofs_l, _qrel = local_group_child_offset_parent_frame(
                            root_frozen_quat, root_frozen_pos, fin_pos, fin_quat)
                        _best_offs[c] = (_ofs_l, _qrel)
                        settled_pos_registry[c]  = fin_pos
                        settled_quat_registry[c] = fin_quat.copy()
                        sl = fin_pos.copy()
                        sw = sl + quat_to_rotmat(fin_quat) @ child_centroid_ofs[j]
                        _best_per_child_results.append({
                            "name":         c,
                            "aa":           _best_iter_children[j]["aa"],
                            "placed_origin":  _best_iter_children[j]["placed_origin"],
                            "placed_world":   _best_iter_children[j]["placed_world"],
                            "settled_origin": sl.copy(),
                            "settled_world":  sw.copy(),
                        })
                    _logger.info(f"  [local_group | G{g_idx}] best: copied env-{_best_env_idx} -> env-0")
                    save_subtree_poses(env, list(group), os.path.join(
                        output_obj_dir, f"local_group_{root_name}.json"))
                    if args.debug_save_local_group_objs:
                        export_subtree_as_obj(
                            env, list(group), _local_obj_dir, label=f"local_group_g{g_idx}")
                _best_offsets = _best_offs
                _logger.info(f"  [local_group | G{g_idx}] episode {episode+1} new best saved"
                             f"  (best_reward={_saved_best_reward:.4f})")
            else:
                _logger.info(f"  [local_group | G{g_idx}] episode {episode+1} no improvement, skip save.")

        r = _best_iter_rewards
        _logger.info(f"[local_group | G{g_idx}] ALL-TIME BEST after {args.cem_episodes} episodes"
                     f"  iter={_best_iter_num}  reward={_best_iter_reward:.4f}"
                     f"  stab={r.get('stab', 0.):.3f}"
                     f"  layout={r.get('layout', 0.):.3f}"
                     f"  vel={r.get('vel', 0.):.3f}"
                     f"  pen={r.get('pen', 0.):.0f}")

        env.unfreeze_actor(root_name)
        group_child_offsets[g_idx] = _best_offsets or {}
        _logger.info(f"[local_group | G{g_idx}] Local group optimization done"
                     f" ({len(group_child_offsets[g_idx])} children)."
                     f" Saved to {output_obj_dir}/local_group_{root_name}.json")

        env.close()
        del env
        args.actor_filter = None
        torch.cuda.empty_cache()

    _logger.removeHandler(_ph); _ph.close()
    return group_child_offsets


# ===================================================================
# global
# ===================================================================

def global_scene_optimize(args, layout, group_child_offsets):
    n_cem_iters = getattr(args, "cem_iters_joint", args.cem_iters_subtree)
    _logger.info("\n[global] Joint CEM optimization")
    _log_dir = os.path.join(args.output_dir, "log")
    os.makedirs(_log_dir, exist_ok=True)
    _ph = attach_phase_log(_logger, _log_dir, "global_scene")
    global_scene_out_dir     = os.path.join(args.output_dir, "global_scene")
    _global_scene_obj_dir  = os.path.join(global_scene_out_dir, "obj_files")
    _global_scene_urdf_dir = os.path.join(global_scene_out_dir, "urdf_files")
    os.makedirs(_global_scene_obj_dir,  exist_ok=True)
    os.makedirs(_global_scene_urdf_dir, exist_ok=True)

    _all_grp_members = set(n for g in layout.groups for n in g)
    _p2_roots   = [g[0] for g in layout.groups]
    _p2_singles = [n for n in layout.all_movable_names if n not in _all_grp_members]
    args.actor_filter = set(_p2_roots) | set(_p2_singles)
    args.scene_layout = layout
    env = MultiEnvCEMPlacement(args)
    render = (env.viewer is not None)

    group_root_to_gidx = {group[0]: g_idx for g_idx, group in enumerate(env.groups)}
    entity_names = list(env.movable_obj_names)
    n_entities   = len(entity_names)

    if n_entities == 0:
        _logger.info("[global] No entities to optimise — skipping.")
        return {}

    group_roots_in_env = [n for n in entity_names if n in group_root_to_gidx]
    _logger.info(f"[global] Entities ({n_entities}): {entity_names}")
    _logger.info(f"[global] Group roots ({len(group_roots_in_env)}): {group_roots_in_env}")

    ref_pos_np = np.array(
        [env.ref_root_pos.get(n, [0., env.base_dy, 0.]) for n in entity_names],
        dtype=np.float32)
    ref_pos_t  = torch.from_numpy(ref_pos_np).float()
    centroid_ofs_np = np.array(
        [env.ref_pos.get(n, [0., env.base_dy, 0.]) for n in entity_names],
        dtype=np.float32) - ref_pos_np
    centroid_ofs_dev = torch.from_numpy(centroid_ofs_np).float().to(env._device)
    orig_world_dev      = torch.stack(
        [env.orig_world_pos[n]  for n in entity_names]).to(env._device)
    orig_world_quat_dev = torch.stack(
        [env.orig_world_quat[n] for n in entity_names]).to(env._device)
    entity_children = [
        [(c, d[0], d[1]) for c, d in group_child_offsets.get(group_root_to_gidx[n], {}).items()]
        if n in group_root_to_gidx else []
        for n in entity_names
    ]
    entity_flat_idx = torch.stack(
        [env.name_idx_long[n] for n in entity_names], dim=1).view(-1)
    entity_hull_dev = [env.hull_verts.get(n) for n in entity_names]

    env.reset_all_envs()
    env._push_root_states()

    warm_start_mode = getattr(args, "cem_warm_start", None)
    cem = CEMOptimizer(args, n_objects=n_entities)

    _best_iter_reward  = -np.inf
    _best_iter_num     = 0
    _best_iter_rewards = {}
    _best_action       = None
    _saved_best_reward = -np.inf
    os.makedirs(global_scene_out_dir, exist_ok=True)

    for episode in range(args.cem_episodes):
        if episode > 0:
            if warm_start_mode is None:
                cem.reset()
            else:
                alpha = getattr(args, "cem_warm_start_alpha", 0.8)
                cem.warm_start(mode=warm_start_mode, alpha=alpha)
                _logger.info(f"  [global] warm-start={warm_start_mode}"
                             f"  mean_norm={np.linalg.norm(cem.mean):.4f}")

        _logger.info(f"\n  [global] Episode {episode+1}/{args.cem_episodes}"
                     f"  CEM init: mean={cem.mean[:6]}  std={cem.std[:6]}")

        for cem_iter in range(n_cem_iters):
            _lambda_settled_geo_pen_eff       = float(args.lambda_settled_geo_pen)
            _lambda_pgp_eff       = float(args.lambda_place_geo_pen)
            _pop_decay = getattr(args, "cem_pop_size_decay", 0.0)
            _n_active  = args.cem_pop_size if _pop_decay == 0.0 else \
                min(args.cem_pop_size,
                    max(int(args.cem_pop_size / (_pop_decay ** cem_iter)), 2 * cem.n_elite))

            samples   = cem.sample()
            samples_t = torch.from_numpy(samples).float()
            s3d       = samples_t.view(env.n_envs, n_entities, 6)
            delta_pos = s3d[:, :, 0:3]
            delta_aa  = s3d[:, :, 3:6]

            placed_pos_all  = ref_pos_t.unsqueeze(0) + delta_pos
            placed_quat_all = axis_angle_to_quat_batch(
                delta_aa.reshape(-1, 3)).view(env.n_envs, n_entities, 4)

            for i, name in enumerate(entity_names):
                env.set_actor_poses_all_envs(
                    name, placed_pos_all[:, i, :], placed_quat_all[:, i, :])
            env._push_root_states()

            env.ig.refresh_actor_root_state_tensor(env.sim)
            placed_root    = (env.root_states[entity_flat_idx, 0:3]
                              .view(env.n_envs, n_entities, 3)
                              - env.env_offsets.unsqueeze(1))
            placed_quat_ig = (env.root_states[entity_flat_idx, 3:7]
                              .view(env.n_envs, n_entities, 4))

            env.settle_all(args.vel_settle_steps, render=render)
            env.ig.refresh_actor_root_state_tensor(env.sim)
            vel_early = (env.root_states[entity_flat_idx, 7:10]
                         .view(env.n_envs, n_entities, 3))
            env.settle_all(args.total_settle_steps - args.vel_settle_steps, render=render)
            env.ig.refresh_actor_root_state_tensor(env.sim)

            settled_root = (env.root_states[entity_flat_idx, 0:3]
                            .view(env.n_envs, n_entities, 3)
                            - env.env_offsets.unsqueeze(1))
            settled_quat = (env.root_states[entity_flat_idx, 3:7]
                            .view(env.n_envs, n_entities, 4))
            _cofs2     = centroid_ofs_dev.view(1, n_entities, 3, 1).expand(env.n_envs, -1, -1, -1)
            _R2_placed  = quat_to_rotmat_batch(placed_quat_ig.reshape(-1, 4)).view(
                env.n_envs, n_entities, 3, 3)
            _R2_settled = quat_to_rotmat_batch(settled_quat.reshape(-1, 4)).view(
                env.n_envs, n_entities, 3, 3)
            placed_world  = placed_root  + torch.matmul(_R2_placed,  _cofs2).squeeze(-1)
            settled_world = settled_root + torch.matmul(_R2_settled, _cofs2).squeeze(-1)

            r_pos_stab = torch.zeros(env.n_envs, device=env._device)
            r_rot_stab = torch.zeros(env.n_envs, device=env._device)
            r_pos_layout  = torch.zeros(env.n_envs, device=env._device)
            r_rot_layout  = torch.zeros(env.n_envs, device=env._device)
            r_vel       = torch.zeros(env.n_envs, device=env._device)
            for i in range(n_entities):
                sw = settled_world[:, i, :]; sq = settled_quat[:, i, :]
                pw = placed_world[:, i, :];  pq = placed_quat_ig[:, i, :]
                ow = orig_world_dev[i];       oq = orig_world_quat_dev[i]
                r_pos_stab += torch.norm(sw - pw, dim=1)
                r_rot_stab += quat_angle_distance_batch(sq, pq)
                r_pos_layout  += torch.norm(sw - ow.unsqueeze(0), dim=1)
                r_rot_layout  += quat_angle_distance_batch(sq, oq)
                r_vel       += torch.norm(vel_early[:, i, :], dim=1)

            if _lambda_pgp_eff > 0.0:
                r_place_geo_pen = compute_geo_pen_batch(
                    placed_root, placed_quat_ig, entity_hull_dev, env._device)
            else:
                r_place_geo_pen = torch.zeros(env.n_envs, device=env._device)
            r_settled_geo_pen = compute_geo_pen_batch(
                settled_root, settled_quat, entity_hull_dev, env._device)

            r_stab   = args.lambda_pose_stab   * r_pos_stab   + args.lambda_rot_stab   * r_rot_stab
            r_layout = args.lambda_pose_layout * r_pos_layout + args.lambda_rot_layout * r_rot_layout
            r_pen    = _lambda_settled_geo_pen_eff * r_settled_geo_pen + _lambda_pgp_eff * r_place_geo_pen
            rewards  = -(r_stab + r_layout + args.lambda_vel * r_vel + r_pen)

            rewards_np = rewards.cpu().numpy()
            cem.update(samples[:_n_active], rewards_np[:_n_active])
            best_i = int(np.argmax(rewards_np[:_n_active]))
            best_r = float(rewards_np[best_i])

            if best_r > _best_iter_reward:
                _best_iter_reward  = best_r
                _best_iter_num     = cem_iter + 1
                _best_action       = samples[best_i].copy()
                _best_iter_rewards = {
                    "stab":   float(r_stab[best_i]),
                    "layout": float(r_layout[best_i]),
                    "vel":    float(r_vel[best_i]),
                    "pen":    float(r_pen[best_i]),
                }

            _logger.debug(f"  [global] ep {episode+1} iter {cem_iter+1:3d}/{n_cem_iters}"
                          f"  best_reward={cem._best_reward:.4f}  cur_best={best_r:.4f}"
                          f"  stab={r_stab[best_i]:.3f}  layout={r_layout[best_i]:.3f}"
                          f"  vel={r_vel[best_i]:.3f}  pen={r_pen[best_i]:.0f}")

            if args.use_wandb:
                _p2_wb_step = cem_iter + episode * n_cem_iters
                wandb.log({
                    "global/cem_iter":        _p2_wb_step,
                    "global/best_reward":     cem._best_reward,
                    "global/cur_best_reward": best_r,
                    "global/best_stab":   float(r_stab[best_i]),
                    "global/best_layout": float(r_layout[best_i]),
                    "global/best_vel":    float(r_vel[best_i]),
                    "global/best_pen":    float(r_pen[best_i]),
                })

            if cem.converged(rewards_np):
                _logger.info("    [global] converged.")
                break

        if cem._best_reward > _saved_best_reward:
            _saved_best_reward = cem._best_reward
            _ep_act_best, _ep_act_mean = cem.get_final_action()
            if getattr(args, "save_mean_result", False) and _ep_act_mean is not None:
                _s3d = torch.from_numpy(_ep_act_mean).float().view(n_entities, 6)
                _dp  = _s3d[:, 0:3].numpy()
                _qnp = axis_angle_to_quat_batch(_s3d[:, 3:6].reshape(-1, 3)).numpy().reshape(n_entities, 4)
                save_global_poses_json(
                    env, entity_names, entity_children, ref_pos_np + _dp, _qnp,
                    os.path.join(global_scene_out_dir, "global_mean_poses.json"))
            _logger.info(f"  [global] episode {episode+1} new best"
                         f"  (best_reward={_saved_best_reward:.4f})")
        else:
            _logger.info(f"  [global] episode {episode+1} no improvement, skip save.")

    r = _best_iter_rewards
    _logger.info(f"[global] ALL-TIME BEST  iter={_best_iter_num}"
                 f"  reward={_best_iter_reward:.4f}"
                 f"  stab={r.get('stab', 0.):.3f}"
                 f"  layout={r.get('layout', 0.):.3f}"
                 f"  vel={r.get('vel', 0.):.3f}"
                 f"  pen={r.get('pen', 0.):.0f}")

    if _best_action is None:
        _logger.info("[global] No valid action found.")
        return {}, None

    best_s3d       = torch.from_numpy(_best_action).float().view(n_entities, 6)
    best_delta_pos = best_s3d[:, 0:3].numpy()
    best_aa        = best_s3d[:, 3:6]
    best_quat_np   = axis_angle_to_quat_batch(
        best_aa.reshape(-1, 3)).numpy().reshape(n_entities, 4)
    best_pos_final = ref_pos_np + best_delta_pos

    group_residuals = {}
    for i, name in enumerate(entity_names):
        if name not in group_root_to_gidx:
            continue
        g_idx    = group_root_to_gidx[name]
        residual = best_delta_pos[i]
        group_residuals[g_idx] = residual
        _logger.info(f"  [global] G{g_idx} root={name}"
                     f"  residual=({residual[0]:+.4f},{residual[1]:+.4f},{residual[2]:+.4f})")

    output_obj_dir = global_scene_out_dir
    os.makedirs(output_obj_dir, exist_ok=True)

    _p2_structural = {"wall", "ceiling", "floor-wall", "floor"}
    _p2_added = 0
    _p2_data = save_global_poses_json(
        env, entity_names, entity_children, best_pos_final, best_quat_np)
    local_group_out_dir = os.path.join(args.output_dir, "local_groups")
    for _bp_path in sorted(glob.glob(os.path.join(local_group_out_dir, "local_group_*.json"))):
        _bp = json.load(open(_bp_path))
        for _bname, _bpose in _bp.get("objects", {}).items():
            if _bname not in _p2_data["objects"]:
                _par = layout.parent_of.get(_bname)
                if _par not in _p2_structural:
                    _p2_data["objects"][_bname] = _bpose
                    _p2_added += 1
    if _p2_added:
        _logger.info(f"[global] Added {_p2_added} group children to final JSON")

    # Ensure every scene node appears in the JSON (wall/ceiling children may have been missed).
    # Use a default pose; postprocess will overwrite with wall-fitted poses for w_walls variant.
    _base_dy = float(layout.base_dy)
    _missing_nodes = [n for n in layout.node_info if n not in _p2_data["objects"]]
    if _missing_nodes:
        for _n in _missing_nodes:
            _p2_data["objects"][_n] = {
                "pos": [0.0, _base_dy, 0.0],
                "rot": [0.0, 0.0, 0.0, 1.0],
                "lin_vel": [0.0, 0.0, 0.0],
            }
        _logger.info(f"[global] Added {len(_missing_nodes)} missing scene nodes to JSON: {_missing_nodes}")

    for i, name in enumerate(entity_names):
        pos_all  = torch.from_numpy(best_pos_final[i]).unsqueeze(0).expand(env.n_envs, -1)
        quat_all = torch.from_numpy(best_quat_np[i]).unsqueeze(0).expand(env.n_envs, -1)
        env.set_actor_poses_all_envs(name, pos_all, quat_all)
    env._push_root_states()
    export_subtree_as_obj(env, list(entity_names), _global_scene_obj_dir, label="")

    for i, name in enumerate(entity_names):
        if not entity_children[i]:
            continue
        rp     = best_pos_final[i]
        root_R = quat_to_rotmat(best_quat_np[i])
        for (c_name, c_ofs, c_quat) in entity_children[i]:
            c_pos        = rp + root_R @ c_ofs
            c_quat_world = quat_multiply(best_quat_np[i], c_quat)
            obj_path = os.path.join(env.obj_dir, f"{env.file_prefix}{c_name}.obj")
            if not os.path.isfile(obj_path):
                continue
            mesh = load_trimesh_any(obj_path)
            T = np.eye(4)
            T[:3, :3] = quat_to_rotmat(c_quat_world)
            T[:3, 3]  = c_pos
            mesh.apply_transform(T)
            mesh.export(os.path.join(_global_scene_obj_dir, f"{c_name}.obj"))

    for _fname, _fpose in _p2_data.get("objects", {}).items():
        _dst = os.path.join(_global_scene_obj_dir, f"{_fname}.obj")
        if os.path.exists(_dst):
            continue
        _base_obj = os.path.join(env.obj_dir, f"{env.file_prefix}{_fname}.obj")
        if not os.path.isfile(_base_obj):
            continue
        _mesh = load_trimesh_any(_base_obj)
        _pos  = np.array(_fpose["pos"], dtype=np.float64)
        _quat = np.array(_fpose["rot"], dtype=np.float64)
        _T = np.eye(4)
        _T[:3, :3] = quat_to_rotmat(_quat)
        _T[:3,  3] = _pos
        _mesh.apply_transform(_T)
        _mesh.export(_dst)

    # Only copy objects that appear in the scene tree (node_info keys).
    # Files like the_floor.obj/urdf are not scene nodes and are excluded this way.
    _scene_node_names = set(layout.node_info.keys())

    # copy any remaining stage2 OBJs (fixed objects not yet in global_scene/obj_files/)
    for _fname in os.listdir(env.obj_dir):
        if not _fname.endswith(".obj"):
            continue
        _stem = _fname[:-4]  # strip .obj
        _name = _stem[len(env.file_prefix):] if (env.file_prefix and _stem.startswith(env.file_prefix)) else _stem
        if _name not in _scene_node_names:
            continue
        _dst = os.path.join(_global_scene_obj_dir, f"{_name}.obj")
        if not os.path.exists(_dst):
            shutil.copy2(os.path.join(env.obj_dir, _fname), _dst)

    # copy URDFs from scene_canon to global_scene/urdf_files/
    _src_urdf_dir = os.path.join(args.scene_canon_dir, "urdf_files")
    if os.path.isdir(_src_urdf_dir):
        for _fname in os.listdir(_src_urdf_dir):
            if not _fname.endswith(".urdf"):
                continue
            _name = _fname[len(env.file_prefix):-5] if (env.file_prefix and _fname.startswith(env.file_prefix)) else _fname[:-5]
            if _name not in _scene_node_names:
                continue
            with open(os.path.join(_src_urdf_dir, _fname)) as _uf:
                _uc = _uf.read()
            if env.file_prefix:
                _uc = _uc.replace(f"../obj_files/{env.file_prefix}", "../obj_files/")
            with open(os.path.join(_global_scene_urdf_dir, f"{_name}.urdf"), "w") as _uf:
                _uf.write(_uc)

    env.close()
    args.actor_filter = None
    _logger.info(f"[global] Global scene optimization done. Saved to {global_scene_out_dir}")
    _logger.removeHandler(_ph); _ph.close()
    return group_residuals, _p2_data


# ===================================================================
# postprocess
# ===================================================================

def postprocess_scene(args, layout, poses_data=None):
    _log_dir = os.path.join(args.output_dir, "log")
    os.makedirs(_log_dir, exist_ok=True)
    _ph = attach_phase_log(_logger, _log_dir, "global_scene_w_walls")

    global_scene_out_dir          = os.path.join(args.output_dir, "global_scene")
    _global_scene_obj_dir      = os.path.join(global_scene_out_dir, "obj_files")
    local_group_out_dir          = os.path.join(args.output_dir, "local_groups")
    _local_groups_obj_dir      = os.path.join(local_group_out_dir, "obj_files")
    global_scene_w_walls_out_dir                    = os.path.join(args.output_dir, "global_scene_w_walls")
    global_scene_w_walls_out_obj_dir                = os.path.join(global_scene_w_walls_out_dir, "obj_files")
    global_scene_w_walls_out_urdf_dir               = os.path.join(global_scene_w_walls_out_dir, "urdf_files")
    global_json                = os.path.join(global_scene_out_dir, "global_scene_poses.json")
    os.makedirs(global_scene_w_walls_out_obj_dir,  exist_ok=True)
    os.makedirs(global_scene_w_walls_out_urdf_dir, exist_ok=True)

    if poses_data is None and not os.path.isfile(global_json):
        _logger.warning(f"[postprocess] global_scene_poses.json not found at {global_json} — skipping.")
        _logger.removeHandler(_ph); _ph.close()
        return None

    wall_names    = [n for n, p in layout.parent_of.items() if p == "wall"]
    ceiling_names = [n for n, p in layout.parent_of.items() if p == "ceiling"]

    if not wall_names and not ceiling_names:
        _logger.info("[postprocess] No wall or ceiling objects found — skipping postprocess.")
        _logger.removeHandler(_ph); _ph.close()
        return None

    if postprocess is None:
        _logger.warning("[postprocess] rest3d.utils.postprocess not importable — skipping postprocess.")
        _logger.removeHandler(_ph); _ph.close()
        return None

    _logger.info(f"\n[postprocess] Placing wall/ceiling objects "
                 f"(wall: {wall_names}, ceiling: {ceiling_names})")

    _pw_buf = io.StringIO()
    _pw_old_stdout = sys.stdout
    sys.stdout = _pw_buf
    try:
        postprocess.run(
            output_dir       = global_scene_w_walls_out_dir,
            root_dir         = args.scene_canon_dir,
            scene_tree_name  = "scene_tree.json",
            margin           = postprocess.WALL_MARGIN,
            out_json_name    = None,
            debug_save_obj   = False,
            wall_align_max_yaw_deg=float(getattr(args, "postprocess_wall_yaw_align_deg", 25.0)),
            force_three_walls=bool(getattr(args, "fit_three_walls", False)),
            poses_data       = poses_data,
            poses_path       = global_json if poses_data is None else None,
            local_group_dir  = local_group_out_dir,
            wall_obj_dir     = global_scene_w_walls_out_obj_dir,
            wall_urdf_dir    = global_scene_w_walls_out_urdf_dir,
        )
    finally:
        sys.stdout = _pw_old_stdout
    for _pw_line in _pw_buf.getvalue().splitlines():
        if _pw_line.strip():
            _logger.info(f"[postprocess] {_pw_line}")

    # copy existing global_scene OBJs → global_scene_w_walls/obj_files/
    # Do NOT overwrite files already written by postprocess.run() (postprocess result wins).
    if os.path.isdir(_global_scene_obj_dir):
        for _fn in os.listdir(_global_scene_obj_dir):
            if _fn.endswith(".obj"):
                _dst = os.path.join(global_scene_w_walls_out_obj_dir, _fn)
                if not os.path.exists(_dst):
                    shutil.copy2(os.path.join(_global_scene_obj_dir, _fn), _dst)

    # promote local_group OBJs not yet in global_scene
    if os.path.isdir(_local_groups_obj_dir):
        for _bfname in sorted(os.listdir(_local_groups_obj_dir)):
            if not (_bfname.startswith("local_group_g") and _bfname.endswith(".obj")):
                continue
            _parts = _bfname.split("_", 3)   # ["local", "group", "g{n}", "{cname}.obj"]
            if len(_parts) < 4:
                continue
            _cname = _parts[3][:-4]
            _dst   = os.path.join(global_scene_w_walls_out_obj_dir, f"{_cname}.obj")
            if not os.path.exists(_dst):
                shutil.copy2(os.path.join(_local_groups_obj_dir, _bfname), _dst)

    # Copy scene_canon OBJs for any scene-tree node not yet in global_scene_w_walls/obj_files/.
    _scene_node_names_pp = set(layout.node_info.keys())
    obj_src_dir = os.path.join(args.scene_canon_dir, "obj_files")
    file_prefix = layout.file_prefix
    if os.path.isdir(obj_src_dir):
        for fname in sorted(os.listdir(obj_src_dir)):
            if not fname.lower().endswith(".obj"):
                continue
            if file_prefix and not fname.startswith(file_prefix):
                continue
            name = fname[len(file_prefix):-4]
            if name not in _scene_node_names_pp:
                continue
            dst = os.path.join(global_scene_w_walls_out_obj_dir, f"{name}.obj")
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(obj_src_dir, fname), dst)

    # copy URDFs scene_canon → global_scene_w_walls/urdf_files/ (only scene nodes)
    _src_urdf_dir = os.path.join(args.scene_canon_dir, "urdf_files")
    if os.path.isdir(_src_urdf_dir):
        for _fname in os.listdir(_src_urdf_dir):
            if not _fname.endswith(".urdf"):
                continue
            _name = _fname[len(file_prefix):-5] if (file_prefix and _fname.startswith(file_prefix)) else _fname[:-5]
            if _name not in _scene_node_names_pp:
                continue
            with open(os.path.join(_src_urdf_dir, _fname)) as _uf:
                _uc = _uf.read()
            if file_prefix:
                _uc = _uc.replace(f"../obj_files/{file_prefix}", "../obj_files/")
            with open(os.path.join(global_scene_w_walls_out_urdf_dir, f"{_name}.urdf"), "w") as _uf:
                _uf.write(_uc)

    _logger.info(f"[postprocess] Wall/ceiling placement done. Saved to {global_scene_w_walls_out_dir}")
    _logger.removeHandler(_ph); _ph.close()


# ===================================================================
# Load helpers
# ===================================================================

def local_group_optimize_in_subprocess(args, layout):
    if os.environ.get("_STABLE_SCENE_LOCAL_GROUP_WORKER") == "1":
        local_group_optimize(args, layout)
        _logger.info("[local_group] subprocess worker done, exiting.")
        sys.exit(0)

    script = os.path.abspath(__file__)
    cmd    = [sys.executable, script] + sys.argv[1:]
    env    = {**os.environ, "_STABLE_SCENE_LOCAL_GROUP_WORKER": "1"}
    _groups_with_children = [(g_idx, g) for g_idx, g in enumerate(layout.groups) if len(g) > 1]
    _logger.info(f"[local_group] {len(_groups_with_children)} group(s) to optimize:")
    for _gi, _g in _groups_with_children:
        _logger.info(f"  G{_gi}  root={_g[0]}  children={list(_g[1:])}")
    _logger.info("[local_group] Spawning subprocess for VRAM isolation ...")
    _logger.info(f"  cmd: {' '.join(cmd)}")

    proc = subprocess.run(cmd, env=env)

    objs_dir = os.path.join(args.output_dir, "local_groups")
    missing  = []
    for group in args.scene_layout.groups:
        root_name = group[0]
        if len(group) <= 1:
            continue
        expected = os.path.join(objs_dir, f"local_group_{root_name}.json")
        if not os.path.isfile(expected):
            missing.append(expected)

    if proc.returncode != 0:
        if missing:
            raise RuntimeError(
                f"[local_group] Subprocess exited with code {proc.returncode} "
                f"and {len(missing)} pose file(s) missing:\n"
                + "\n".join(f"  {p}" for p in missing))
        else:
            _logger.warning(
                f"[local_group] Subprocess exited with code {proc.returncode} "
                f"(likely Isaac Gym cleanup crash), but all pose files present — continuing.")
    elif missing:
        raise FileNotFoundError(
            f"[local_group] Subprocess rc=0 but {len(missing)} pose file(s) missing:\n"
            + "\n".join(f"  {p}" for p in missing))

    _logger.info(f"[local_group] Subprocess done. All pose files verified in {objs_dir}/")


# ===================================================================
# Orchestration
# ===================================================================

def stabilize_scene(args, layout):
    if args.use_wandb:
        for _gi, _grp in enumerate(layout.groups):
            wandb.define_metric(f"local_group{_gi}/cem_iter")
            wandb.define_metric(f"local_group{_gi}/*", step_metric=f"local_group{_gi}/cem_iter")
        wandb.define_metric("global/cem_iter")
        wandb.define_metric("global/*", step_metric="global/cem_iter")

    local_group_load_dir = getattr(args, "local_group_load_dir", None)
    global_load_dir = getattr(args, "global_load_dir", None)
    has_children    = any(len(g) > 1 for g in layout.groups)

    _SEP = "=" * 60

    _logger.info(f"\n{_SEP}\n[Step 1] Local Group Optimization\n{_SEP}")
    if local_group_load_dir:
        group_child_offsets = load_local_group_from_dir(layout, local_group_load_dir)
    elif not has_children:
        _logger.info("[local_group] No groups with children — skipping local_group.")
        group_child_offsets = {g_idx: {} for g_idx, g in enumerate(layout.groups)}
    else:
        local_group_optimize_in_subprocess(args, layout)
        local_group_out_dir = os.path.join(args.output_dir, "local_groups")
        group_child_offsets = load_local_group_from_dir(layout, local_group_out_dir)

    _logger.info(f"\n{_SEP}\n[Step 2] Global Scene Optimization\n{_SEP}")
    if global_load_dir:
        load_global_from_dir(args, layout, group_child_offsets, global_load_dir)
        poses_data = None  # load path writes global_scene_poses.json; postprocess reads from file
    else:
        _, poses_data = global_scene_optimize(args, layout, group_child_offsets)

    _logger.info(f"\n{_SEP}\n[Step 3] Postprocess (Wall / Ceiling Placement)\n{_SEP}")
    postprocess_scene(args, layout, poses_data)


# ===================================================================
# Entry point
# ===================================================================

def config_args():
    ap = argparse.ArgumentParser(description="Physics-based scene stabilization (Stage 3)")
    ap.add_argument("--scene_dir", type=str, required=True,
                    help="Path to stage2 output directory (contains scene_canon/, scene_tree.json)")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Output directory for stage3 results")
    cli = ap.parse_args()

    cfg = StableSceneCfg()
    args = argparse.Namespace(**vars(cfg))

    args.output_dir = cli.output_dir
    args.scene_canon_dir = os.path.join(cli.scene_dir, "scene_canon")
    args.wandb_run_name = os.path.basename(os.path.normpath(cli.output_dir))

    return args


def main():
    args = config_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if os.environ.get("_STABLE_SCENE_LOCAL_GROUP_WORKER") == "1":
        args.use_wandb = False
    _logger.setLevel(logging.DEBUG if args.debug_log else logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
               for h in _logger.handlers):
        _sh = logging.StreamHandler(sys.stdout)
        _sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
        _logger.addHandler(_sh)
    _logger.debug(f"Command: {' '.join(sys.argv)}")
    for k, v in sorted(vars(args).items()):
        _logger.debug(f"  {k} = {v}")

    layout = SceneLayout(args)
    args.scene_layout = layout

    _logger.info(f"[Groups] {len(layout.groups)} groups:")
    for g_idx, group in enumerate(layout.groups):
        _logger.info(f"  G{g_idx}: {group}")

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args),
                   dir=os.path.join(args.output_dir, "log"))

    stabilize_scene(args, layout)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    # Isaac Gym P4 segfaults during Python interpreter teardown (known upstream bug).
    # sys.exit(0) terminates before the C++ module destructors fire.
    sys.exit(0)
