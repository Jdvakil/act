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
        # Collision-aware reconvert: drop inbound scrapes (keep deflect grazes),
        # min-pool skin substeps, upsample real bows. Paste num_episodes /
        # episode_len from convert_meta.json after:
        #   python -m scripts.convert_obstacle_to_act --with_proximity --prox_pool min \
        #       --skip_approach_collision --keep_deflect_collisions --upsample_deflect 3 \
        #       --src assets/datagen/hybrid_invis_obstacle_v1/FrankaSkinHybridInvisObstacleConfig/20260703_095653 \
        #       --dst act_style_data/obstacle_prox_avoid_v1
        'dataset_dir': str(ACT_DATA_DIR / 'obstacle_prox_avoid_v1'),
        'num_episodes': 0,
        'episode_len': 185,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
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
