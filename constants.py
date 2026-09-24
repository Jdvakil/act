import pathlib

# Repo root = the parent of submodules/act. Everything below is derived from it so
# the file keeps working if the checkout moves.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ACT_DATA_DIR = REPO_ROOT / 'act_style_data'

### Task parameters
# Upstream ACT bimanual sim tasks. Unused by this project -- kept only because
# record_sim_episodes.py, scripted_policy.py and imitate_episodes.py import the name.
DATA_DIR = '/home/jaydv/code/proximity_learning/data'
SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top'],
    },
    'sim_transfer_cube_human': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top'],
    },
    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top'],
    },
    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top'],
    },
}

# This project's tasks. All three are the one-env fumehood obstacle pick; they differ
# only in whether the dataset carries proximity and whether the hazard bar is visible
# to the RGB cameras. Dataset paths are produced by scripts/convert_obstacle_to_act.py.
TASK_CONFIGS = {
    'obstacle_baseline': {
        # VANILLA ACT BASELINE (rgb + qpos, NO proximity) on the one-env obstacle
        # pick: red cup in the fumehood, hazard bar present ~75% of episodes.
        # 100 successful episodes (5 houses x 25 trajs, dropping fail[-1]
        # trajectories). qpos=9 (arm7 + 2 fingers), action=8 (arm7 + 1 gripper cmd).
        # Source episode T: median 84, max 167.
        #
        # NOT ON DISK -- deleted 2026-08-16 to reclaim space. The trained checkpoints
        # under ckpts/act_obstacle_baseline_v1/ still exist, so published numbers are
        # reproducible; you only need this dataset to retrain. Rebuild with:
        #   python -m scripts.convert_obstacle_to_act \
        #       --runs assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855 \
        #       --out act_style_data/obstacle_v1
        'dataset_dir': str(ACT_DATA_DIR / 'obstacle_v1'),
        'num_episodes': 100,
        'episode_len': 169,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'obstacle_pact': {
        # SAME obstacle pick as obstacle_baseline, but the dataset ALSO carries
        # /observations/proximity (T,40,8,8) so it can train BOTH arms of the PACT
        # comparison: vanilla ACT (ignores proximity) and P+ACT (--use_proximity).
        # qpos=9, action=8.
        #
        # NOT ON DISK -- deleted 2026-08-16 alongside obstacle_v1. Checkpoints under
        # ckpts/obstacle_pact/ survive. Rebuild with:
        #   python -m scripts.convert_obstacle_to_act --with_proximity \
        #       --runs assets/datagen/hybrid_obstacle_v1/FrankaSkinHybridObstacleConfig/20260612_183855 \
        #       --out act_style_data/obstacle_prox_v1
        'dataset_dir': str(ACT_DATA_DIR / 'obstacle_prox_v1'),
        'num_episodes': 100,
        'episode_len': 168,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'obstacle_pact_v2': {
        # CURRENT TRAINING SET. Invisible-bar collection (causal forcing): hazard bar
        # hidden from every RGB camera (MuJoCo geom group 4) but visible to the skin
        # depth renderer and fully present in physics. Cells: visible-bar /
        # invisible-bar / free at 0.375 / 0.375 / 0.25 (OBSTACLE_P=0.75, INVIS_P=0.5);
        # object placement decoupled from bar presence so vision carries NO bar cue in
        # the invisible cell. Source = scripts/convert_obstacle_to_act.py
        # --with_proximity over hybrid_invis_obstacle_v1 20260703_095653 (5 of 8 houses
        # survived an OOM; 105 successful episodes).
        'dataset_dir': str(ACT_DATA_DIR / 'obstacle_prox_v2'),
        'num_episodes': 105,
        'episode_len': 185,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'obstacle_pact_avoid_v1': {
        # Collision-aware reconvert of hybrid_obstacle_v1 20260612_183855
        # (2026-08-23): drop inbound scrapes except deflect grazes, min-pool
        # skin substeps, 3× upsample bows. 125 source trajs − 25 fail − 13
        # non-deflect collisions = 87 unique (32 deflect × 3 + 55 free = 151).
        # RESULT (2026-08-24, n=50 invisible): PACT 30% vs vanilla 40% collisions,
        # p≈0.40, success down — failed the ≥15pt/p<0.05 bar. Superseded by
        # obstacle_gate_v1: the bows here were learnable from vision alone.
        'dataset_dir': str(ACT_DATA_DIR / 'obstacle_prox_avoid_v1'),
        'num_episodes': 151,
        'episode_len': 140,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'obstacle_gate_v1': {
        # GATE-BAR dataset (v3.1, 2026-08-24): 44 cm pole snapped onto the live
        # TCP line (~18 cm bow), bow sign = wall coin-flip, INVIS_P=1.0 so no
        # training episode ever renders the pole. Cup y is independent of bar
        # fields; cameras see where the line is, not which way is open.
        # Source: FrankaSkinHybridGateBarConfig -> assets/datagen/hybrid_gate_bar_v1,
        # converted by scripts/convert_obstacle_to_act.py --with_proximity
        # --prox_pool min --skip_approach_collision (no upsample; every bar ep bows).
        # num_episodes / episode_len are placeholders until convert prints the real
        # counts — paste them here (convert refuses nothing; training with 0 would
        # see an empty set).
        'dataset_dir': str(ACT_DATA_DIR / 'obstacle_gate_v1'),
        'num_episodes': 0,     # <- paste from convert output
        'episode_len': 0,      # <- paste from convert output
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'pact_place_corridor_v5': {
        # Coauthor recovered pick-and-place corridor (HF Lundii/pact_place_corridor_v5).
        # 152 clean-success demos. Wrist RGB only (no exo). 40-sensor skin in
        # /observations/proximity. Scene XML is pact_place_corridor_v2; HF name is
        # the v5 recovery schema. Convert:
        #   python -m scripts.convert_pact_place_to_act \
        #       --src data/pact_place_corridor_v5 \
        #       --dst act_style_data/pact_place_corridor_v5 \
        #       --with_proximity --prox_pool min --image_h 240 --image_w 320
        # Converted 2026-08-25: 152/152 clean, max T=634, sides left=72 right=80.
        'dataset_dir': str(ACT_DATA_DIR / 'pact_place_corridor_v5'),
        'num_episodes': 152,
        'episode_len': 636,
        'camera_names': ['wrist_camera'],
    },
    'pact_pick_n_place_v2': {
        # v10.11d four-object place (data/pact_pick_n_place_v2/data/v1011d).
        # 200 accepted demos. Table RGB is exo_camera_1 + wrist. 40-sensor skin
        # in /observations/proximity. Env pact_place_corridor_v10_11d. Convert:
        #   python -m scripts.convert_pact_place_to_act \
        #       --src data/pact_pick_n_place_v2/data/v1011d \
        #       --dst act_style_data/pact_pick_n_place_v2/data/v1011d \
        #       --with_proximity --prox_pool min --image_h 240 --image_w 320 \
        #       --task_name pact_pick_n_place_v2
        # Converted 2026-09-03: 200/200, max T=559, sides left=100 right=100.
        # hdf5 keys: exo_camera_1 + wrist_camera. Franka dims: qpos 9, action 8.
        # Eval: eval_act_pact_pick_n_place.py (V1010 sampler, v10_7 XMLs,
        # origin/main worktree). Not eval_act_place_corridor.py.
        'dataset_dir': str(ACT_DATA_DIR / 'pact_pick_n_place_v2/data/v1011d'),
        'num_episodes': 200,
        'episode_len': 561,
        'state_dim': 9,
        'action_dim': 8,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'pact_place_corridor_v1010': {
        # Four-object v10.10 (data/pact_place_corridor/data/v1010/accepted).
        # Convert 2026-09-11: 215/215, max T=633, table_camera + wrist.
        # Not exo_camera_1. Closed-loop eval not wired.
        'dataset_dir': str(ACT_DATA_DIR / 'pact_place_corridor/data/v1010/accepted'),
        'num_episodes': 215,
        'episode_len': 635,
        'state_dim': 9,
        'action_dim': 8,
        'camera_names': ['table_camera', 'wrist_camera'],
    },
    'pact_place_corridor_v107_spaced': {
        # v10.6 spaced pendant (data/pact_place_corridor/data/v107_spaced/accepted).
        # Convert 2026-09-11: 210/210, max T=615, table_camera + wrist.
        # Closed-loop: eval_act_v107spaced.py. Not eval_act.py --task.
        'dataset_dir': str(ACT_DATA_DIR / 'pact_place_corridor/data/v107_spaced/accepted'),
        'num_episodes': 210,
        'episode_len': 617,
        'state_dim': 9,
        'action_dim': 8,
        'camera_names': ['table_camera', 'wrist_camera'],
    },
    'pact_place_corridor_v10_11c_100': {
        # Mixed v10.11c clutter geometry (taller primitives).
        # Convert 2026-09-11: 99/99, max T=546, exo + wrist. Not v1011d.
        # Closed-loop eval not wired.
        'dataset_dir': str(ACT_DATA_DIR / 'mixed_v1011_clutter_geometry/pact_place_corridor_v10_11c_100'),
        'num_episodes': 99,
        'episode_len': 548,
        'state_dim': 9,
        'action_dim': 8,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'pact_pick_n_place_v2_v1011d': {
        # v10.11d (data/pact_pick_n_place_v2/data/v1011d).
        # Convert 2026-09-03: 200/200, max T=559, exo + wrist.
        # Closed-loop eval: repo-root eval_act_v1011d.py.
        'dataset_dir': str(ACT_DATA_DIR / 'pact_pick_n_place_v2/data/v1011d'),
        'num_episodes': 200,
        'episode_len': 561,
        'state_dim': 9,
        'action_dim': 8,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'pact_pick_n_place_v2_v12': {
        # v12.0 (data/pact_pick_n_place_v2/data/v12).
        # Convert 2026-09-11: 165/165, max T=581, exo + wrist.
        # Closed-loop eval: repo-root eval_act_place.py --env v12.
        'dataset_dir': str(ACT_DATA_DIR / 'pact_pick_n_place_v2/data/v12'),
        'num_episodes': 165,
        'episode_len': 583,
        'state_dim': 9,
        'action_dim': 8,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'pact_pick_n_place_v2_v6': {
        # v6 (data/pact_pick_n_place_v2/data/v6): V10.10 two-object, only the route
        # bottles Soap_Bottle_30 (slot 01) and Soap_Bottle_11 (slot 06) live.
        # Convert 2026-09-22: 200/200, max T=622, exo + wrist, sides 100/100.
        # Closed-loop eval: repo-root eval_act_place.py --env v6.
        'dataset_dir': str(ACT_DATA_DIR / 'pact_pick_n_place_v2/data/v6'),
        'num_episodes': 200,
        'episode_len': 624,
        'state_dim': 9,
        'action_dim': 8,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'pact_pick_n_place_v2_v107_spaced': {
        # v107_spaced hub (data/pact_pick_n_place_v2/data/v107_spaced): 200 eps, exo + wrist.
        # Not the 210-ep batman table_camera set 'pact_place_corridor_v107_spaced'.
        # Convert 2026-09-22: 200/200, max T=620, sides 100/100.
        # Closed-loop eval: repo-root eval_act_place.py --env v107_spaced.
        'dataset_dir': str(ACT_DATA_DIR / 'pact_pick_n_place_v2/data/v107_spaced'),
        'num_episodes': 200,
        'episode_len': 622,
        'state_dim': 9,
        'action_dim': 8,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'pact_place_corridor_v5_ext': {
        # v5 ext (data/pact_place_corridor/data/v5/pick_and_place/accepted): 193 eps on the
        # hallway corridor-v2 scene; episode IDs disjoint from the 152 Lundii v5 rows.
        # Convert 2026-09-22: 193/193, max T=650, sides L101/R92. The hdf5 also carries
        # table_camera (batman review render, pose not recorded); trained wrist-only so it
        # runs on the hallway protocol. Closed-loop eval: eval_act.py --task hallway.
        'dataset_dir': str(ACT_DATA_DIR / 'pact_place_corridor/data/v5/pick_and_place/accepted'),
        'num_episodes': 193,
        'episode_len': 652,
        'state_dim': 9,
        'action_dim': 8,
        'camera_names': ['wrist_camera'],
    },
    'pact_place_corridor_v107': {
        # v107 (data/pact_place_corridor/data/v107/pick_and_place/accepted): V10.7 asymmetric
        # pendant + V9.5 clutter, 48 eps. Convert 2026-09-22 with --hold_half_rate_video
        # (25/48 table mp4s recorded on even steps only), max T=609, sides L28/R20.
        # Wrist-only (table_camera pose not recorded). No closed-loop evaluator.
        'dataset_dir': str(ACT_DATA_DIR / 'pact_place_corridor/data/v107/pick_and_place/accepted'),
        'num_episodes': 48,
        'episode_len': 611,
        'state_dim': 9,
        'action_dim': 8,
        'camera_names': ['wrist_camera'],
    },
}

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
