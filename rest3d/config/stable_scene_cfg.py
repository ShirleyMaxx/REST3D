"""Default configuration for the stable_scene physics placement pipeline."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StableSceneCfg:
    # ----------------------------------------------------------------
    # Physics
    # ----------------------------------------------------------------
    vhacd_enabled: bool = True
    ground_scene: bool = True
    total_settle_steps: int = 60
    vel_settle_steps: int = 15
    linear_damping: float = 0.3
    angular_damping: float = 0.3
    no_override_com: bool = False
    static: bool = False
    headless: bool = True
    num_position_iterations: int = 6
    max_depenetration_velocity: float = 5.0

    # ----------------------------------------------------------------
    # Output control
    # ----------------------------------------------------------------
    debug_save_local_group_objs: bool = True
    debug_save_global_objs: bool = True
    debug_save_postprocess_objs: bool = False
    save_mean_result: bool = False
    place_fixed_last: bool = True
    use_fixed_type: bool = False
    fit_three_walls: bool = False
    postprocess_wall_yaw_align_deg: float = 25.0
    debug_log: bool = False

    # ----------------------------------------------------------------
    # CEM
    # ----------------------------------------------------------------
    act_dim: int = 6
    cem_pop_size: int = 2048
    cem_episodes: int = 2
    cem_iters_subtree: int = 15
    cem_iters_joint: int = 15
    cem_elite_frac: float = 0.025
    cem_final_best: bool = True
    keep_best: bool = True
    update_use_only_best: bool = False
    std_update_mode: str = "topk_std"
    cem_update_mode: str = "cem"
    nes_lr_mu: float = 1.0
    nes_lr_sigma: Optional[float] = None
    decay_std_rate: float = 0.95
    cem_warm_start: str = "prev_mean_std"
    cem_warm_start_alpha: float = 0.8
    cem_pop_size_decay: float = 0.0
    reward_threshold: float = -0.01

    # ----------------------------------------------------------------
    # CEM init distribution
    # ----------------------------------------------------------------
    init_trans_x_mean: float = 0.0
    init_trans_y_mean: float = 0.0
    init_trans_z_mean: float = 0.0
    init_rot_roll_mean: float = 0.0
    init_rot_pitch_mean: float = 0.0
    init_rot_yaw_mean: float = 0.0
    init_trans_x_std: float = 0.05
    init_trans_y_std: float = 0.005
    init_trans_z_std: float = 0.05
    init_rot_roll_std: float = 0.005
    init_rot_pitch_std: float = 0.05
    init_rot_yaw_std: float = 0.005

    # ----------------------------------------------------------------
    # Energyweights
    # ----------------------------------------------------------------
    lambda_vel: float = 1.0
    lambda_pose_stab: float = 1.0
    lambda_pose_layout: float = 6.0
    lambda_rot_stab: float = 1.0
    lambda_rot_layout: float = 1.0
    lambda_settled_geo_pen: float = 0.5
    lambda_place_geo_pen: float = 0.5

    # ----------------------------------------------------------------
    # Wandb (disabled by default)
    # ----------------------------------------------------------------
    use_wandb: bool = True
    wandb_project: str = "stable_scene"
    wandb_run_name: str = ""

    # ----------------------------------------------------------------
    # Internal state (not user-facing)
    # ----------------------------------------------------------------
    track_objects: List = field(default_factory=list)
    actor_filter: Optional[object] = None
    scene_layout: Optional[object] = None
    urdf_dir_override: Optional[str] = None
    local_group_load_dir: Optional[str] = None
    global_load_dir: Optional[str] = None
