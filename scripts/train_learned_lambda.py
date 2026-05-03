# ============================================================================
# Train LearnedLambdaSchedule (PostHoc-A v2)
#
# 论文对应: §5.6 plug-and-play 升级 — 自适应 modulation
#
# 设计:
#   1. 加载 frozen Plain backbone (FoundationScore from dit_plain ckpt)
#   2. 加载 frozen shock detector (固定算法 argmax|∇D_x|, 无参数)
#   3. 训 LearnedLambdaSchedule (~8K params)
#   4. 训练流程 (与 train_posthoc_bv 类似的 DSM proxy):
#      - 拿 batch (x_target, IC, σ ~ ViscosityMatchedSchedule.sample_sigma)
#      - x_noisy = x_target + σ·ε
#      - 走 Plain backbone → D_plain
#      - 算 BV correction = λ(σ, IC) · (κ_local/2) · tanh(d/2σ²) · sign(d)
#         — 其中 λ(σ, IC) 来自 LearnedLambdaSchedule (待训)
#      - s_total = (D_plain - x_noisy)/σ² + BV_correction
#      - D_total = x_noisy + σ²·s_total
#      - L = ||D_total - x_target||² (DSM)
#   5. Adam optimize 仅 lambda_schedule params
#
# 用法:
#   python scripts/train_learned_lambda.py --config configs/foundation/posthoc_a_v2.yaml
# ============================================================================

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.data.mixed_pde_dataset import MixedPDEDataset
from src.models.foundation_score import FoundationScore
from src.models.learned_lambda_schedule import LearnedLambdaSchedule, n_params
from src.diffusion.schedules import ViscosityMatchedSchedule
from src.diffusion.posthoc_bv_sampler import detect_shock_loc_and_jump, signed_distance_to_shock


def build_plain_backbone_from_ckpt(plain_ckpt_path: Path, n_pde_types: int, device) -> FoundationScore:
    """加载 dit_plain ckpt 重建 FoundationScore."""
    state = torch.load(str(plain_ckpt_path), map_location=device, weights_only=False)
    cfg = state["config"]
    model_cfg = cfg["model"]
    if model_cfg["type"] != "dit_plain":
        raise ValueError(f"plain_ckpt 必须是 dit_plain (got {model_cfg['type']})")

    dit_kwargs = dict(model_cfg["dit"])
    dit_kwargs["n_pde_types"] = n_pde_types
    in_channels = int(model_cfg.get("in_channels", 2))
    Nx = int(dit_kwargs.get("Nx", 128))
    nx_for_dit = dit_kwargs.pop("Nx", Nx)

    backbone = FoundationScore(in_channels=in_channels, Nx=Nx, dit_kwargs=dit_kwargs)
    backbone.load_state_dict(state["model"])
    backbone = backbone.to(device)
    backbone.eval()
    # 冻结
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def compute_dsm_loss_with_learned_lambda(
    plain_backbone: FoundationScore,
    lambda_schedule: LearnedLambdaSchedule,
    x_target: torch.Tensor,
    cond: torch.Tensor,
    sigma: torch.Tensor,
    pde_id: torch.Tensor,
    device,
) -> torch.Tensor:
    """
    DSM proxy loss (训练 LearnedLambdaSchedule 用).

    流程:
      1. x_noisy = x_target + σ·ε
      2. Plain backbone → D_plain (no_grad)
      3. 检测 shock + κ_local + signed distance (无参数, 但要 grad-friendly)
      4. tanh + sign → bv_term (常数张量)
      5. λ = LearnedLambdaSchedule(σ, IC) — 唯一可训
      6. D_total = x_noisy + σ²·(s_plain + λ·bv_term)
      7. L = ||D_total - x_target||²
    """
    B = x_target.shape[0]
    Nx = x_target.shape[-1]

    # x_noisy
    noise = torch.randn_like(x_target)
    sigma_b = sigma.view(B, 1, 1)
    x_noisy = x_target + sigma_b * noise

    # Plain forward (no grad)
    x_input = torch.cat([x_noisy, cond], dim=1) if cond is not None else x_noisy
    with torch.no_grad():
        D_plain = plain_backbone(x_input, sigma, pde_id=pde_id)  # (B, 1, Nx)

    # s_plain (no grad)
    s_plain = (D_plain - x_noisy) / (sigma_b ** 2 + 1e-8)

    # 检测 shock (no grad needed, but compute always)
    with torch.no_grad():
        shock_idx, kappa_local = detect_shock_loc_and_jump(D_plain)
        d_shock = signed_distance_to_shock(shock_idx, Nx, device, dtype=D_plain.dtype)

    # tanh 剖面 (per sample, σ 不同)
    sigma_sq = (sigma_b ** 2).clamp(min=1e-6)
    tanh_factor = torch.tanh(d_shock / (2.0 * sigma_sq))           # (B, 1, Nx)
    sign_d = torch.sign(d_shock)                                   # (B, 1, Nx)
    kappa_b = kappa_local.view(B, 1, 1)
    bv_term = (kappa_b / 2.0) * tanh_factor * sign_d               # (B, 1, Nx)
    # 上面全部 no-grad (来自 D_plain 和固定 shock 算法)

    # λ(σ, IC) — 唯一可训
    # IC 用 cond (IC-conditioning) 作 IC features 输入
    ic_for_lambda = cond if cond is not None else x_noisy  # fallback
    lam_per_sample = lambda_schedule(sigma, ic_for_lambda)  # (B,)
    lam = lam_per_sample.view(B, 1, 1)

    # s_total
    s_total = s_plain + lam * bv_term  # (B, 1, Nx) — 仅 lam 有梯度

    # D_total
    D_total = x_noisy + sigma_sq * s_total  # (B, 1, Nx)

    # L_DSM
    loss = ((D_total - x_target) ** 2).mean()
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="random seed (for multi-seed training)")
    args = parser.parse_args()

    # 设置 seed
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    device = torch.device(env.default_device)
    config_path = (PROJECT_ROOT / args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    if model_cfg["type"] != "learned_lambda":
        raise ValueError(f"This script handles model.type=learned_lambda, got {model_cfg['type']}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = exp_cfg["name"]
    seed_tag = f"_s{args.seed}" if args.seed != 42 else ""
    output_dir = env.output_dir / (exp_name + seed_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(exp_cfg.get("epochs", 50))
    lr = float(exp_cfg.get("learning_rate", 1e-3))
    batch_size = int(exp_cfg.get("batch_size", 64))
    nu = float(exp_cfg.get("nu", 1.0))
    tau_max = float(exp_cfg.get("tau_max", 1.0))
    ckpt_every = int(exp_cfg.get("ckpt_every", 10))
    warm_start = bool(model_cfg.get("warm_start_to_v1", True))

    print(f"[train_learned_lambda] config: {config_path}, seed={args.seed}")
    print(f"  output_dir: {output_dir}")
    print(f"  epochs={epochs}  lr={lr}  batch={batch_size}")

    # ---- 数据 ----
    pdes = data_cfg["pdes"]
    train_dataset = MixedPDEDataset(
        pdes_config=pdes, data_dir=env.data_dir,
        mode="train", conditioning_type=data_cfg.get("conditioning_type", "ic"),
        mix_strategy=data_cfg.get("mix_strategy", "uniform"),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=env.num_workers, collate_fn=MixedPDEDataset.collate_fn,
    )
    n_pde_types = train_dataset.num_pde_types

    # ---- Plain backbone (frozen) ----
    plain_ckpt_path = (PROJECT_ROOT / model_cfg["plain_ckpt"]).resolve()
    if not plain_ckpt_path.exists():
        raise FileNotFoundError(f"plain_ckpt 不存在: {plain_ckpt_path}")
    plain_backbone = build_plain_backbone_from_ckpt(plain_ckpt_path, n_pde_types, device)
    print(f"[plain] frozen, params={sum(p.numel() for p in plain_backbone.parameters())/1e6:.2f}M")

    # ---- LearnedLambdaSchedule ----
    hidden = int(model_cfg.get("hidden", 64))
    freq_dim = int(model_cfg.get("freq_dim", 64))
    lam_max = float(model_cfg.get("lam_max", 5.0))
    lambda_schedule = LearnedLambdaSchedule(
        hidden=hidden, freq_dim=freq_dim, lam_max=lam_max, zero_init=True,
    ).to(device)
    print(f"[lambda] LearnedLambdaSchedule  trainable={n_params(lambda_schedule)} (~{n_params(lambda_schedule)/1e3:.1f}K)")

    # 可选: warm-start to v1 behavior (让初始 λ ≈ exp_decay)
    if warm_start:
        print("[warm-start] pre-fit λ to v1 exp_decay behavior...")
        lambda_schedule.init_to_v1_behavior()

    optimizer = optim.Adam(lambda_schedule.parameters(), lr=lr)
    schedule = ViscosityMatchedSchedule(nu=nu, tau_max=tau_max)

    # ---- Resume ----
    start_epoch = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(args.resume)
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        lambda_schedule.load_state_dict(state["lambda_schedule"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0))
        print(f"[resume] from epoch {start_epoch}")

    # ---- 日志 ----
    log_path = output_dir / f"train_log_{exp_name}{seed_tag}_{run_timestamp}.txt"
    log_fp = open(str(log_path), "w", encoding="utf-8")
    log_fp.write(f"# LearnedLambdaSchedule Training\n")
    log_fp.write(f"# config: {config_path}\n")
    log_fp.write(f"# seed: {args.seed}\n")
    log_fp.write(f"# trainable: {n_params(lambda_schedule)}\n")
    log_fp.write(f"# epoch  loss_dsm\n")
    log_fp.flush()

    # ---- 训练循环 ----
    print(f"[train] start (epoch 1 → {epochs})")
    grad_clip_norm = float(exp_cfg.get("grad_clip_norm", 1.0))
    for epoch in range(start_epoch, epochs):
        lambda_schedule.train()
        plain_backbone.eval()  # 始终冻结

        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x_target = batch["x_target"].to(device)
            cond = train_dataset.get_conditioning(batch)
            cond = cond.to(device) if cond is not None else None
            pde_id = batch["pde_id"].to(device)

            B = x_target.shape[0]
            sigmas = schedule.sample_sigma(B, device)

            loss = compute_dsm_loss_with_learned_lambda(
                plain_backbone, lambda_schedule,
                x_target, cond, sigmas, pde_id, device,
            )

            if not torch.isfinite(loss):
                print(f"  [warn] NaN loss at epoch {epoch+1}, skip batch")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lambda_schedule.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        line = f"Epoch {epoch+1}/{epochs} | DSM: {avg:.6f}"
        print(line)
        log_fp.write(f"{epoch+1:4d}  {avg:.8f}\n")
        log_fp.flush()

        if (epoch + 1) % ckpt_every == 0 or (epoch + 1) == epochs:
            ckpt_file = output_dir / f"learned_lambda_{exp_name}{seed_tag}_{run_timestamp}_ep{epoch+1}.pt"
            torch.save({
                "lambda_schedule": lambda_schedule.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "config": cfg,
                "seed": args.seed,
                "plain_ckpt_path": str(plain_ckpt_path),
                "pde_names": train_dataset.pde_names,
            }, str(ckpt_file))
            print(f"  [ckpt] {ckpt_file}")

    log_fp.close()
    print(f"[train] done. log: {log_path}")


if __name__ == "__main__":
    main()
