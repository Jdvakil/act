import pathlib

### Task parameters
DATA_DIR = '/home/jaydv/code/proximity_learning/data'
SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },
}

TASK_CONFIGS = {
    'test': {
        'dataset_dir': '/home/jaydv/code/proximity_learning/act_episodes',
        'num_episodes': 200,  # Your dataset has 200 episodes
        'episode_len': 50,    # Maximum episode length (for padding consistency)
        'camera_names': ['top']
    },
    
    'proximity_learning': {
        'dataset_dir': '/home/jaydv/code/proximity_learning/proximity_learning_dataset_episodes',
        'num_episodes': 200,  # Your dataset has 200 episodes
        'episode_len': 50,    # Maximum episode length (for padding consistency)
        'camera_names': ['top']  # Images saved as 'top' in HDF5
    },

    'pla_house1_mug': {
        # PLA house_1 mug pickup, 250 episodes. qpos=9 (arm7+2 fingers),
        # action=8 (arm7+gripper_cmd1). Episodes are 261 frames each.
        'dataset_dir': '/home/jaydv/code/prox_learning/act_style_data/pla_house1_mug_v1',
        'num_episodes': 250,
        'episode_len': 261,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'pla_smoke': {
        # PLA smoke set: 10 houses × ~3-4 trajs = 36 episodes. Episode lengths
        # vary (223-301, mean 261). Used for the proximity-residual experiment.
        'dataset_dir': '/home/jaydv/code/prox_learning/act_style_data/smoke_v1',
        'num_episodes': 36,
        'episode_len': 301,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'pla_house1_mug_random': {
        # PLA house_1 mug pickup, randomized everything: 356 episodes,
        # T 220-301 (median 291). Source = mug_house_1_random_everything datagen.
        'dataset_dir': '/home/jaydv/code/prox_learning/act_style_data/mug_house1_random_everything',
        'num_episodes': 356,
        'episode_len': 301,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'pla_house3_mug_random': {
        # PLA house_3 mug pickup, randomized everything: 132 episodes,
        # T 180-301 (median 269). Source = mug_house_3_random_everything
        # datagen (collected 2026-05-24, 3 parallel run_data.sh chunks).
        'dataset_dir': '/home/jaydv/code/prox_learning/act_style_data/mug_house3_random_everything',
        'num_episodes': 132,
        'episode_len': 301,
        'camera_names': ['exo_camera_1', 'wrist_camera'],
    },
    'pla_houses_1_3_mug_random': {
        # PLA combined house_1 + house_3 mug pickup, randomized everything:
        # 488 episodes (356 from h1, 132 from h3 appended at indices 356..487
        # via symlinks; see scripts/build_combined_h1_h3.py).
        'dataset_dir': '/home/jaydv/code/prox_learning/act_style_data/mug_houses_1_3_random_everything',
        'num_episodes': 488,
        'episode_len': 301,
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
