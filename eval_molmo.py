
import sys
import numpy as np
import torch

# ===== PATHS =====
ACT_ROOT = "/home/qinzhengfangli/act"
MOLMO_ROOT = "/home/qinzhengfangli/molmo_test/molmospaces"

sys.argv = ["ignore"]
sys.path.append(ACT_ROOT)
sys.path.append(f"{ACT_ROOT}/detr")
sys.path.append(MOLMO_ROOT)

# ===== IMPORTS =====
from detr.main import build_ACT_model
from molmo_spaces.evaluation.configs.evaluation_configs import DummyPickPlaceEvalConfig
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.configs.camera_configs import CameraSystemConfig
# =========================
# 1. MODEL
# =========================
class DummyArgs:
    hidden_dim = 512
    dim_feedforward = 3200
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    num_queries = 20
    backbone = "resnet18"
    lr_backbone = 1e-5
    masks = False
    dilation = False
    position_embedding = "sine"
    dropout = 0.1
    pre_norm = False
    camera_names = ["wrist"]
    state_dim = 9


def load_model():
    print("Building model...")
    model = build_ACT_model(DummyArgs())

    print("Loading weights...")
    state_dict = torch.load(f"{ACT_ROOT}/ckpt_molmo_chunk20_bs64_100ep_gripperfix/policy_best.ckpt", map_location="cpu")

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[len("model."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.cuda()
    model.eval()

    print("✅ Model loaded successfully")
    return model


# =========================
# 2. SAMPLER (正确版本)
# =========================
def make_sampler():
    config = DummyPickPlaceEvalConfig()
    config.camera_config = CameraSystemConfig(
        name="default",
        img_resolution=(224,224)
    )

    config.robot_config = FrankaRobotConfig()
    config.task_sampler_config.samples_per_house = 1
    config.task_sampler_config.house_inds = [0]   # 先用一个scene
    config.task_sampler_config.max_tasks = 1

    sampler = config.task_sampler_config.task_sampler_class(config)
    return sampler




def to_tensor(image_np, qpos_np):
    image_np = np.asarray(image_np)

    # HWC → CHW
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0)

    image = torch.tensor(image_np, dtype=torch.float32).cuda()
    image = image.unsqueeze(0)

    if image.max() > 1:
        image = image / 255.0

    qpos_np = np.asarray(qpos_np).reshape(-1)[:9]
    qpos = torch.tensor(qpos_np, dtype=torch.float32).cuda().unsqueeze(0)

    return image, qpos


# =========================
# 4. ACTION
# =========================
def get_action(model, qpos, image):
    with torch.no_grad():
        out = model(qpos, image, None, None, None)

    if isinstance(out, tuple):
        out = out[0]

    print("model out shape:", out.shape)
    return out.action.squeeze(0).cpu().numpy()


def check_success(info, reward):
    if isinstance(info, dict):
        if "success" in info:
            return info["success"]
        if "is_success" in info:
            return info["is_success"]
    return reward > 0


# =========================
# 5. MAIN LOOP
# =========================
def main():
    model = load_model()
    sampler = make_sampler()

    num_episodes = 10
    max_steps = 100
    success_count = 0

    print("🔥 Running evaluation...")

    for ep in range(num_episodes):
        success = False
        step = 0
        print(f"\n===== Episode {ep} =====")

        #task = sampler.sample_task()
        max_tries = 20
        task = None
        for i in range(max_tries):
            task = sampler.sample_task()
            if task is not None:
                break
        print(f"retry {i+1}...")
        if task is None:
            print(" skip episode (sampling failed)")
            continue





        env = task.env
        if env is None:
            raise RuntimeError("Cannot find env")

    # run episode
        action_chunk = None
        for t in range(max_steps):

            #env.step(1)
            if t % 20 == 0:

                image_np = env._renderer.render()
                qpos_np = env.mj_datas[0].qpos.copy()
                print("image shape:", image_np.shape)
                image, qpos = to_tensor(image_np, qpos_np)
                action_chunk = get_action_chunk(model, qpos, image)

            action = action_chunk[t%20]
            action = action[:8]
            env.mj_datas[0].ctrl[:] = action
            env.step()

            try:
                if env.is_success():
                   success = True
                   break
            except:
                  pass

            step += 1

        if success:
            success_count +=1
        print(f"Episode {ep} | success={success} | steps={step}")

    print("\n🔥 SUCCESS RATE:", success_count / num_episodes)


if __name__ == "__main__":
    main()

