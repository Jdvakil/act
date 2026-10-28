"""Diffusion Policy baseline (cameras + qpos only, no skin), 2026-10-17.

A standard CNN-based Diffusion Policy (Chi et al., RSS 2023), written to sit next to ACTPolicy
with the same call contract so imitate_episodes.py trains it and eval_act_v1011d.py runs it
without a second evaluator:

    policy(qpos, image, actions, is_pad)       -> loss dict          (training / validation)
    policy(qpos, image)                         -> (B, horizon, A)    (inference, z-scored actions)

* Inputs: one observation step (n_obs_steps = 1: the evaluator renders cameras only on query
  steps), every camera (exo + wrist) through its own ResNet18 (ImageNet init, frozen BatchNorm:
  the same visual backbone and init as ACT's DETR backbone), spatial-softmax keypoints (32 per
  camera) -> 64-d per camera, concatenated with the z-scored 9-d qpos as the global condition.
* Images: 0-1 floats (B, n_cam, 3, H, W) like ACT; ImageNet normalisation inside. Random crop
  (crop_frac, default 0.9 -> 216 x 288 of 240 x 320) at training, centre crop at eval.
* Denoiser: 1D temporal conditional U-Net (down dims 256/512/1024, kernel 5, GroupNorm 8, FiLM).
* Noise: DDPM, 100 training steps, squaredcos_cap_v2 betas, epsilon prediction, x0 clipped to
  [-1, 1]. Inference: deterministic DDIM (eta = 0), 10 steps.
* Actions: the data loader hands z-scored actions (dataset_stats.pkl, as for ACT). Inside, they
  are mapped to raw and then min-max scaled to [-1, 1] (the DP convention, gripper 0 / 255 ->
  -1 / 1). The output is mapped back to z-scores so the evaluator's post-processing is unchanged.
* Horizon: predicts `horizon` actions (default 56, divisible by 4 for the U-Net); the evaluator
  executes the first `num_queries` (50, ACT's chunk) and re-plans (receding horizon). Padded tail
  steps (episode end) repeat the last real action, as DP's sequence sampler does.

EMA weights (DP's power-0.75 warmup EMA) are kept by `EMA` below; imitate_episodes saves the EMA
model as policy_best / policy_last.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DP_DEFAULTS = {
    "horizon": 56,
    "num_queries": 50,
    "down_dims": (256, 512, 1024),
    "kernel_size": 5,
    "n_groups": 8,
    "diffusion_step_embed_dim": 256,
    "num_train_timesteps": 100,
    "num_inference_steps": 10,
    "num_kp": 32,
    "crop_frac": 0.9,
    "image_hw": (240, 320),
    "weight_decay": 1e-6,
    "warmup_steps": 500,
    "ema_power": 0.75,
    "ema_max_decay": 0.9999,
}


# ----------------------------------------------------------------------------- vision encoder
class SpatialSoftmax(nn.Module):
    """1x1 conv to K keypoint maps, softmax over space, expected (x, y) per keypoint -> 2K."""

    def __init__(self, in_channels: int, h: int, w: int, num_kp: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_kp, kernel_size=1)
        ys, xs = torch.meshgrid(torch.linspace(-1.0, 1.0, h), torch.linspace(-1.0, 1.0, w), indexing="ij")
        self.register_buffer("pos", torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1), persistent=False)
        self.num_kp = num_kp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        att = F.softmax(self.conv(x).flatten(2), dim=-1)  # (B, K, HW)
        return (att @ self.pos).reshape(b, self.num_kp * 2)


class CameraEncoder(nn.Module):
    """ResNet18 trunk (frozen BN, as ACT) -> spatial softmax -> Linear -> ReLU (64-d)."""

    def __init__(self, crop_hw: tuple[int, int], num_kp: int, pretrained: bool) -> None:
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.trunk = nn.Sequential(*list(net.children())[:-2])
        with torch.no_grad():
            fh, fw = self.trunk(torch.zeros(1, 3, *crop_hw)).shape[-2:]
        self.pool = SpatialSoftmax(512, int(fh), int(fw), num_kp)
        self.out = nn.Linear(num_kp * 2, num_kp * 2)
        self.feature_dim = num_kp * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.out(self.pool(self.trunk(x))))


# ----------------------------------------------------------------------------- 1D U-Net
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        emb = t.float()[:, None] * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, n_groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(n_groups, cout), nn.Mish())

    def forward(self, x):
        return self.block(x)


class CondResBlock1D(nn.Module):
    """Two conv blocks with FiLM (scale, bias) from the condition vector in between."""

    def __init__(self, cin: int, cout: int, cond_dim: int, k: int, n_groups: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Conv1dBlock(cin, cout, k, n_groups), Conv1dBlock(cout, cout, k, n_groups)])
        self.cout = cout
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cout * 2))
        self.residual = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        scale, bias = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.cout, 1).unbind(1)
        out = self.blocks[1](scale * out + bias)
        return out + self.residual(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim: int, global_cond_dim: int, down_dims=(256, 512, 1024),
                 kernel_size: int = 5, n_groups: int = 8, dsed: int = 256) -> None:
        super().__init__()
        self.step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed))
        cond_dim = dsed + global_cond_dim
        dims = [input_dim] + list(down_dims)
        in_out = list(zip(dims[:-1], dims[1:]))

        def res(a, b):
            return CondResBlock1D(a, b, cond_dim, kernel_size, n_groups)

        self.down = nn.ModuleList()
        for i, (a, b) in enumerate(in_out):
            last = i >= len(in_out) - 1
            self.down.append(nn.ModuleList([res(a, b), res(b, b),
                                            nn.Conv1d(b, b, 3, 2, 1) if not last else nn.Identity()]))
        mid = down_dims[-1]
        self.mid = nn.ModuleList([res(mid, mid), res(mid, mid)])
        self.up = nn.ModuleList()
        for a, b in reversed(in_out[1:]):
            self.up.append(nn.ModuleList([res(b * 2, a), res(a, a), nn.ConvTranspose1d(a, a, 4, 2, 1)]))
        self.final = nn.Sequential(Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
                                   nn.Conv1d(down_dims[0], input_dim, 1))

    def forward(self, sample, timestep, global_cond):
        """sample (B, T, A), timestep (B,), global_cond (B, G) -> (B, T, A)."""
        x = sample.transpose(1, 2)
        cond = torch.cat([self.step_encoder(timestep), global_cond], dim=-1)
        h = []
        for r1, r2, down in self.down:
            x = r2(r1(x, cond), cond)
            h.append(x)
            x = down(x)
        for m in self.mid:
            x = m(x, cond)
        for r1, r2, up in self.up:
            x = torch.cat((x, h.pop()), dim=1)
            x = up(r2(r1(x, cond), cond))
        return self.final(x).transpose(1, 2)


# ----------------------------------------------------------------------------- noise schedule
def squaredcos_cap_v2_betas(n: int, max_beta: float = 0.999) -> torch.Tensor:
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    return torch.tensor([min(1 - alpha_bar((i + 1) / n) / alpha_bar(i / n), max_beta) for i in range(n)],
                        dtype=torch.float32)


# ----------------------------------------------------------------------------- policy
class DPModel(nn.Module):
    def __init__(self, cfg: dict, pretrained: bool) -> None:
        super().__init__()
        h, w = cfg["image_hw"]
        crop = (int(round(h * cfg["crop_frac"])), int(round(w * cfg["crop_frac"])))
        n_cam = len(cfg["camera_names"])
        # Named `backbones.<i>` so eval_act_v1011d._n_backbones counts one per camera, as for ACT.
        self.backbones = nn.ModuleList([CameraEncoder(crop, cfg["num_kp"], pretrained) for _ in range(n_cam)])
        gdim = sum(b.feature_dim for b in self.backbones) + int(cfg["state_dim"])
        self.unet = ConditionalUnet1D(int(cfg["action_dim"]), gdim, tuple(cfg["down_dims"]),
                                      int(cfg["kernel_size"]), int(cfg["n_groups"]),
                                      int(cfg["diffusion_step_embed_dim"]))
        self.crop_hw = crop


class DiffusionPolicy(nn.Module):
    """ACTPolicy-shaped wrapper: same __call__ / configure_optimizers contract."""

    def __init__(self, args_override: dict, pretrained: bool = True) -> None:
        super().__init__()
        cfg = dict(DP_DEFAULTS)
        cfg.update({k: v for k, v in args_override.items() if v is not None})
        if cfg["horizon"] % 4 != 0:
            raise ValueError(f"DP horizon {cfg['horizon']} must be divisible by 4 (two U-Net downsamples)")
        if cfg["num_queries"] > cfg["horizon"]:
            raise ValueError(f"DP num_queries {cfg['num_queries']} > horizon {cfg['horizon']}")
        self.cfg = cfg
        self.model = DPModel(cfg, pretrained)
        a = int(cfg["action_dim"])
        # Action normalisation (filled by set_action_stats before training; saved in the state dict).
        for name in ("z_mean", "a_min"):
            self.register_buffer(name, torch.zeros(a))
        for name in ("z_std", "a_max"):
            self.register_buffer(name, torch.ones(a))
        betas = squaredcos_cap_v2_betas(int(cfg["num_train_timesteps"]))
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0), persistent=False)
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.horizon = int(cfg["horizon"])
        self.num_queries = int(cfg["num_queries"])
        self._optimizer = None

    # ---- normalisation
    def set_action_stats(self, z_mean, z_std, a_min, a_max) -> None:
        for name, v in (("z_mean", z_mean), ("z_std", z_std), ("a_min", a_min), ("a_max", a_max)):
            getattr(self, name).copy_(torch.as_tensor(v, dtype=torch.float32).reshape(-1))

    def _z_to_unit(self, z):
        raw = z * self.z_std + self.z_mean
        rng = (self.a_max - self.a_min).clamp(min=1e-2)
        return 2.0 * (raw - self.a_min) / rng - 1.0

    def _unit_to_z(self, u):
        rng = (self.a_max - self.a_min).clamp(min=1e-2)
        raw = (u + 1.0) / 2.0 * rng + self.a_min
        return (raw - self.z_mean) / self.z_std

    # ---- observation encoding
    def _crop(self, x):
        h, w = x.shape[-2:]
        ch, cw = self.model.crop_hw
        if self.training:
            top = int(torch.randint(0, h - ch + 1, (1,)))
            left = int(torch.randint(0, w - cw + 1, (1,)))
        else:
            top, left = (h - ch) // 2, (w - cw) // 2
        return x[..., top:top + ch, left:left + cw]

    def _global_cond(self, qpos, image):
        feats = []
        for i, enc in enumerate(self.model.backbones):
            img = (self._crop(image[:, i]) - self.img_mean) / self.img_std
            feats.append(enc(img))
        return torch.cat(feats + [qpos], dim=-1)

    # ---- diffusion
    def _sample(self, gcond, generator=None):
        b = gcond.shape[0]
        a = int(self.cfg["action_dim"])
        x = torch.randn((b, self.horizon, a), device=gcond.device, generator=generator)
        n_train = int(self.cfg["num_train_timesteps"])
        n_inf = int(self.cfg["num_inference_steps"])
        ratio = n_train // n_inf
        ac = self.alphas_cumprod
        for t in range(n_inf - 1, -1, -1):
            t_cur = t * ratio  # "leading" DDIM spacing: 90, 80, ..., 0
            t_prev = t_cur - ratio
            a_t = ac[t_cur]
            a_prev = ac[t_prev] if t_prev >= 0 else torch.ones((), device=ac.device)  # set_alpha_to_one
            eps = self.model.unet(x, torch.full((b,), t_cur, device=x.device, dtype=torch.long), gcond)
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1.0, 1.0)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x

    def __call__(self, qpos, image, actions=None, is_pad=None, proximity_positions=None, image_dropped=None):
        if proximity_positions is not None:
            raise ValueError("DiffusionPolicy is a cameras-only baseline: no proximity input")
        gcond = self._global_cond(qpos, image)
        if actions is None:  # inference: z-scored (B, horizon, A), like ACT's a_hat
            return self._unit_to_z(self._sample(gcond))
        actions = actions[:, :self.horizon]
        is_pad = is_pad[:, :self.horizon]
        # Padded tail = repeat the last real action (DP's sequence sampler pads by edge replicate).
        n_real = (~is_pad).sum(dim=1).clamp(min=1)
        idx = torch.arange(self.horizon, device=actions.device).unsqueeze(0)
        idx = torch.minimum(idx, (n_real - 1).unsqueeze(1))
        actions = torch.gather(actions, 1, idx.unsqueeze(-1).expand_as(actions))
        x0 = self._z_to_unit(actions)
        noise = torch.randn_like(x0)
        t = torch.randint(0, int(self.cfg["num_train_timesteps"]), (x0.shape[0],), device=x0.device)
        a_t = self.alphas_cumprod[t].view(-1, 1, 1)
        noisy = a_t.sqrt() * x0 + (1 - a_t).sqrt() * noise
        mse = F.mse_loss(self.model.unet(noisy, t, gcond), noise)
        out = {"mse": mse, "loss": mse}
        if not self.training:
            # Validation: L1 of the DDIM-sampled chunk vs the demo on the executed steps (z-scored,
            # real steps only), the checkpoint-selection metric (parallel to ACT's l1).
            pred = self._unit_to_z(self._sample(gcond))
            k = self.num_queries
            mask = (~is_pad[:, :k]).unsqueeze(-1).float()
            out["l1_sample"] = (F.l1_loss(pred[:, :k], actions[:, :k], reduction="none") * mask).sum() / (
                mask.sum() * actions.shape[-1]).clamp(min=1.0)
        return out

    # ---- optimisation
    def configure_optimizers(self):
        if self._optimizer is None:
            bb = [p for n, p in self.named_parameters() if n.startswith("model.backbones.") and ".trunk." in n
                  and p.requires_grad]
            bb_ids = {id(p) for p in bb}
            rest = [p for p in self.parameters() if p.requires_grad and id(p) not in bb_ids]
            self._optimizer = torch.optim.AdamW(
                [{"params": rest, "lr": float(self.cfg["lr"])},
                 {"params": bb, "lr": float(self.cfg["lr_backbone"])}],
                lr=float(self.cfg["lr"]), betas=(0.95, 0.999), weight_decay=float(self.cfg["weight_decay"]))
        return self._optimizer


def cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def f(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class EMA:
    """DP's EMAModel: decay = 1 - (1 + step) ** -power, capped at max_decay; buffers copied."""

    def __init__(self, model: nn.Module, power: float = 0.75, max_decay: float = 0.9999) -> None:
        opt, model._optimizer = model._optimizer, None  # never deep-copy the optimizer
        self.averaged_model = copy.deepcopy(model)
        model._optimizer = opt
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)
        self.power, self.max_decay, self.step_count = power, max_decay, 0

    @torch.no_grad()
    def step(self, model: nn.Module) -> None:
        decay = min(self.max_decay, max(0.0, 1.0 - (1.0 + self.step_count) ** -self.power))
        self.step_count += 1
        for ema_p, p in zip(self.averaged_model.parameters(), model.parameters()):
            if p.requires_grad:
                ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
            else:
                ema_p.copy_(p.detach())
        for ema_b, b in zip(self.averaged_model.buffers(), model.buffers()):
            ema_b.copy_(b)
