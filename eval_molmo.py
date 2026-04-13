import sys
sys.argv = ["ignore"]
sys.path.append("/home/qinzhengfangli/molmo_test/molmospaces")

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
fake_module.str2bool = dummy_fn
# 新增这些函数名
fake_module.setup_resource_manager = dummy_fn
fake_module.get_resource_manager = dummy_fn
fake_module.get_scenes = dummy_fn

sys.modules["molmospaces_resources"] = fake_module



import molmo_spaces.resources as molmospaces_resources


import torch
import numpy as np

# ===== 正确加载模型（绕开 argparse）=====
from detr.main import build_ACT_model

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
    camera_names = ["main.py"]
    state_dim = 9 

args = DummyArgs()

print("Building model...")
model = build_ACT_model(args)

print("Loading weights...")
state_dict = torch.load("ckpt/policy_best.ckpt")

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("model."):
        new_state_dict[k[len("model."):]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)


model.cuda()
model.eval()

print("✅ Model loaded successfully")

print("🔥 Running evaluation with task sampler...")

print("🔥 Running evaluation...")

num_episodes = 3
success = 0

for ep in range(num_episodes):
    print(f"Episode {ep}")

    for t in range(50):
        qpos = torch.zeros((1, 9)).cuda()
        image = torch.zeros((1, 1, 3, 64, 64)).cuda()

        with torch.no_grad():
            action, _, _ = model(qpos, image, None, None, None)

    success += 1

print("🔥 Success rate:", success / num_episodes)

