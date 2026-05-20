"""ACT eval in the env matching the dup250 training data.

The dup250 dataset (assets/datagen/pick_and_place_one_house_mug_dup250_v1/
FrankaSkinPickAndPlaceOneHouseMugDup250Config/) was a 250-episode collection
on house_1 + mug where every episode is pixel-identical (1 demo replicated
250 times — see PLA memory dataset_dup250_duplicate_demos). The original
Dup250 config class no longer lives in the molmospaces tree, so this script
reconstructs the closest matching env from FrankaSkinPickAndPlaceOneHouseMugConfig
with ALL randomization disabled, so the eval scenarios are drawn from the
same (deterministic) distribution the policy was trained on.

Run from the repo root:

    cd /home/jaydv/code/prox_learning/submodules/act
    PYTHONPATH="$PWD:$PYTHONPATH" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    python eval_act_house1_dup250.py \
        --ckpt_dir /home/jaydv/code/prox_learning/submodules/act/ckpts/act_house1_mug_v1 \
        --num_rollouts 10 \
        --output_dir /home/jaydv/code/prox_learning/eval_output/act_house1_mug_v1_dup250env_10ep
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.pop("DISPLAY", None)

import argparse
from pathlib import Path

# Reuse the existing wrapper / config / CLI plumbing.
from eval_act_house1 import (
    ACTHouse1MugEvalConfig,
    ACTInferencePolicy,
    ACTPolicyConfig,
)
from utils import set_seed

from molmo_spaces.configs.robot_configs import ActionNoiseConfig
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner


class ACTHouse1MugDup250EvalConfig(ACTHouse1MugEvalConfig):
    """Eval env matching the dup250 training distribution.

    Disables every per-attempt randomization knob the parent enables so the
    eval scenario is deterministic — mirroring the dup250 dataset, which is
    1 demo replicated 250 times (pixel-identical frames across episodes).

    Knobs we turn off vs FrankaSkinPickAndPlaceOneHouseMugConfig:
      - randomize_textures / randomize_robot_textures   -> False
      - randomize_lighting                              -> False
      - randomize_dynamics                              -> False
      - robot_config.init_qpos_noise_range              -> None
      - robot_config.action_noise_config.enabled        -> False
    """

    def model_post_init(self, _context):  # type: ignore[override]
        super().model_post_init(_context)
        ts = self.task_sampler_config
        ts.randomize_textures = False
        ts.randomize_robot_textures = False
        ts.randomize_lighting = False
        ts.randomize_dynamics = False
        self.robot_config.init_qpos_noise_range = None
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--ckpt_name", default="policy_best.ckpt")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_rollouts", type=int, default=10)
    p.add_argument("--task_horizon", type=int, default=300)
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--kl_weight", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--image_h", type=int, default=240)
    p.add_argument("--image_w", type=int, default=320)
    p.add_argument("--temp_agg_off", action="store_true")
    p.add_argument("--temp_agg_m", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=2026)
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[act-eval-dup250] ignoring extra args: {unknown}")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = ACTHouse1MugDup250EvalConfig()
    cfg.task_horizon = args.task_horizon
    cfg.task_sampler_config.samples_per_house = args.num_rollouts
    cfg.output_dir = Path(args.output_dir).resolve()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.seed = args.seed

    pc = cfg.policy_config
    pc.ckpt_dir = str(Path(args.ckpt_dir).resolve())
    pc.ckpt_name = args.ckpt_name
    pc.chunk_size = args.chunk_size
    pc.kl_weight = args.kl_weight
    pc.hidden_dim = args.hidden_dim
    pc.dim_feedforward = args.dim_feedforward
    pc.image_h = args.image_h
    pc.image_w = args.image_w
    pc.temp_agg_off = args.temp_agg_off
    pc.temp_agg_m = args.temp_agg_m

    cfg.save_config()
    print(f"[act-eval-dup250] writing rollouts to {cfg.output_dir}")
    print(f"[act-eval-dup250] ckpt: {pc.ckpt_dir}/{pc.ckpt_name}")

    runner = ParallelRolloutRunner(cfg)
    success, total = runner.run()
    print(f"[act-eval-dup250] success {success}/{total}")
    summary = cfg.output_dir / "summary.txt"
    summary.write_text(
        f"ckpt_dir: {pc.ckpt_dir}\n"
        f"ckpt_name: {pc.ckpt_name}\n"
        f"num_rollouts: {args.num_rollouts}\n"
        f"task_horizon: {args.task_horizon}\n"
        f"seed: {args.seed}\n"
        f"success: {success}\n"
        f"total: {total}\n"
        f"success_rate: {(success/total) if total else 0.0:.4f}\n"
    )


if __name__ == "__main__":
    main()
