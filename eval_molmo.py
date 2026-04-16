import sys
from types import SimpleNamespace
from pathlib import Path

# ===== paths =====
ACT_ROOT = "/home/qinzhengfangli/act"
MOLMO_ROOT = "/home/qinzhengfangli/molmo_test/molmospaces"

sys.argv = ["ignore"]
sys.path.append(ACT_ROOT)
sys.path.append(f"{ACT_ROOT}/detr")
sys.path.append(MOLMO_ROOT)


import types
fake_module = types.ModuleType("molmospaces_resources")

class Dummy:
    def __init__(self, *args, **kwargs):
        pass

def dummy_fn(*args, **kwargs):
    return None

fake_module.HFRemoteStorage = Dummy
fake_module.LocalStorage = Dummy
fake_module.ResourceManager = Dummy
fake_module.R2RemoteStorage = Dummy
fake_module.S3RemoteStorage = Dummy
fake_module.GCSRemoteStorage = Dummy
fake_module.FileSystemStorage = Dummy
fake_module.PickleLMDBMap = Dummy

fake_module.str2bool = dummy_fn
fake_module.setup_resource_manager = dummy_fn
fake_module.get_resource_manager = dummy_fn
fake_module.get_scenes = dummy_fn
fake_module.split_query_tokens = dummy_fn

sys.modules["molmospaces_resources"] = fake_module

import numpy as np
import torch

from detr.main import build_ACT_model
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler


# =========================
# 1. build model
# =========================
class DummyArgs:
    hidden_dim = 256
    dim_feedforward = 1024
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    num_queries = 10
    backbone = "resnet18"
    lr_backbone = 1e-5
    masks = False
    dilation = False
    position_embedding = "sine"
    dropout = 0.1
    pre_norm = False
    camera_names = ["main"]
    state_dim = 9


def load_model(ckpt_path: str):
    print("Building model...")
    model = build_ACT_model(DummyArgs())

    print("Loading weights...")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[len("model."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.cuda()
    model.eval()
    print(" Model loaded successfully")
    return model


# =========================
# 2. sampler config
# =========================
def make_sampler():
    task_sampler_config = SimpleNamespace(
        house_inds=[0],
        pickup_types=None,
    )

    task_config = SimpleNamespace(
        model_copy=lambda deep=True: SimpleNamespace()
    )

    config = SimpleNamespace(
        seed=0,
        num_envs=1,
        data_split="train",
        scene_dataset="procthor-10k",
        task_sampler_config=task_sampler_config,
        task_config_preset_exp=None,
        task_config=task_config,
    )

    sampler = PickTaskSampler(config)
    return sampler


# =========================
# 3. obs helpers
# =========================
def extract_image_and_qpos(obs):

    if isinstance(obs, dict):
        print("obs keys:", list(obs.keys()))

        # image
        if "image" in obs:
            image = obs["image"]
        elif "images" in obs:
            # 常见情况：images 是 dict
            if isinstance(obs["images"], dict):
                first_key = list(obs["images"].keys())[0]
                image = obs["images"][first_key]
            else:
                image = obs["images"]
        else:
            raise KeyError("Could not find image in observation")

        # qpos
        if "qpos" in obs:
            qpos = obs["qpos"]
        elif "state" in obs:
            qpos = obs["state"]
        elif "robot_state" in obs:
            qpos = obs["robot_state"]
        else:
            raise KeyError("Could not find qpos/state in observation")

        return image, qpos

    raise TypeError(f"Unsupported obs type: {type(obs)}")


def image_to_tensor(image_np):
    image_np = np.asarray(image_np)

    # HWC -> CHW
    if image_np.ndim == 3:
        image_np = np.transpose(image_np, (2, 0, 1))
        image_np = np.expand_dims(image_np, axis=0)   # num_cam = 1
    elif image_np.ndim == 4:
        # already camera-first maybe
        pass
    else:
        raise ValueError(f"Unexpected image shape: {image_np.shape}")

    image = torch.tensor(image_np, dtype=torch.float32, device="cuda")

    # [num_cam, C, H, W] -> [B, num_cam, C, H, W]
    image = image.unsqueeze(0)

    # normalize if image looks like uint8 range
    if image.max() > 1.0:
        image = image / 255.0

    return image


def qpos_to_tensor(qpos_np):
    qpos_np = np.asarray(qpos_np, dtype=np.float32).reshape(-1)
    qpos_np = qpos_np[:9]  # ACT state_dim = 9
    return torch.tensor(qpos_np, dtype=torch.float32, device="cuda").unsqueeze(0)


def action_from_model_output(model_out):
    if isinstance(model_out, tuple):
        action_pred = model_out[0]
    else:
        action_pred = model_out

    action = action_pred[:, 0, :]
    action = action.squeeze(0).detach().cpu().numpy()
    return action


def is_success(info, reward):
    if isinstance(info, dict):
        if "success" in info:
            return bool(info["success"])
        if "is_success" in info:
            return bool(info["is_success"])
    return bool(reward is not None and reward > 0)


# =========================
# 4. main eval loop
# =========================
def main():
    ckpt_path = f"{ACT_ROOT}/ckpt/policy_best.ckpt"
    model = load_model(ckpt_path)

    print(" Creating task sampler...")
    sampler = make_sampler()

    num_episodes = 10
    max_steps = 100
    success_count = 0

    print(" Running evaluation...")

    for ep in range(num_episodes):
        print(f"\n===== Episode {ep} =====")

        task = sampler.sample_task()

        env = task.make_env()

        obs = env.reset()
        done = False
        info = {}
        reward = 0.0
        step = 0

        while not done and step < max_steps:
            image_np, qpos_np = extract_image_and_qpos(obs)

            image = image_to_tensor(image_np)
            qpos = qpos_to_tensor(qpos_np)

            with torch.no_grad():
                model_out = model(qpos, image, None, None, None)

            action = action_from_model_output(model_out)
            print("action shape:", action.shape)

            obs, reward, done, info = env.step(action)
            step += 1

        ep_success = is_success(info, reward)
        success_count += int(ep_success)

        print(f"Episode {ep} done | success={ep_success} | reward={reward} | steps={step}")

    success_rate = success_count / num_episodes
    print(f"\n Success rate: {success_rate:.3f} ({success_count}/{num_episodes})")


if __name__ == "__main__":
    main()

