# ============================================================================
# 方案 B 训练: PostHocBVAwareScore (冻结 plain backbone + 训 phi_sh + kappa)
#
# 用法:
#   python scripts/train_posthoc_bv.py --config configs/foundation/posthoc_b.yaml
#
# 设计:
#   - 加载 dit_plain ckpt → frozen FoundationScore as backbone
#   - 包装为 PostHocBVAwareScore (新增 phi_sh_net + kappa_net)
#   - 仅 phi_sh_net + kappa_net 参与梯度
#   - L_DSM 单 loss
#   - 50 epoch 即可
# ============================================================================

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.data.mixed_pde_dataset import MixedPDEDataset
from src.models.foundation_score import FoundationScore
from src.models.posthoc_bv_score import PostHocBVAwareScore
from src.diffusion.schedules import ViscosityMatchedSchedule
from src.diffusion.losses import get_dsm_loss


def build_plain_backbone_from_ckpt(plain_ckpt_path: Path, n_pde_types: int, device) -> FoundationScore:
    """加载 plain ckpt + 重建 FoundationScore."""
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
    print(f"[plain backbone] loaded from {plain_ckpt_path.name}, params={sum(p.numel() for p in backbone.parameters())/1e6:.2f}M")
    return backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="random seed (用于多 seed 训练)")
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
    aug_cfg = exp_cfg.get("augmentation", None)   # ← W5 ext: data augmentation

    if model_cfg["type"] != "posthoc_bv":
        raise ValueError(f"This script only handles model.type=posthoc_bv, got {model_cfg['type']}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = exp_cfg["name"]
    seed_tag = f"_s{args.seed}" if args.seed != 42 else ""   # ← W5 ext: seed-suffix dir
    output_dir = env.output_dir / (exp_name + seed_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(exp_cfg.get("epochs", 50))
    lr = float(exp_cfg.get("learning_rate", 5e-4))
    base_batch_size = int(exp_cfg.get("batch_size", 64))
    nu = float(exp_cfg.get("nu", 1.0))
    tau_max = float(exp_cfg.get("tau_max", 1.0))
    ckpt_every = int(exp_cfg.get("ckpt_every", 10))

    print(f"[train_posthoc_bv] config: {config_path}")
    print(f"  output_dir: {output_dir}")
    print(f"  epochs={epochs}  lr={lr}  batch={base_batch_size}")

    # ---- 数据 ----
    pdes = data_cfg["pdes"]
    train_dataset = MixedPDEDataset(
        pdes_config=pdes, data_dir=env.data_dir,
        mode="train", conditioning_type=data_cfg.get("conditioning_type", "ic"),
        mix_strategy=data_cfg.get("mix_strategy", "uniform"),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=base_batch_size, shuffle=True,
        num_workers=env.num_workers, collate_fn=MixedPDEDataset.collate_fn,
    )
    n_pde_types = train_dataset.num_pde_types

    # ---- 加载 plain backbone (frozen) ----
    plain_ckpt_path = (PROJECT_ROOT / model_cfg["plain_ckpt"]).resolve()
    if not plain_ckpt_path.exists():
        raise FileNotFoundError(f"plain_ckpt 不存在: {plain_ckpt_path}")
    plain_backbone = build_plain_backbone_from_ckpt(plain_ckpt_path, n_pde_types, device)

    # ---- 包装为 PostHocBVAwareScore ----
    in_channels = int(model_cfg.get("in_channels", 2))
    # W5 ext: 支持调大 phi_sh / kappa 网络
    phi_sh_dim = int(model_cfg.get("phi_sh_dim", 32))
    kappa_dim = int(model_cfg.get("kappa_dim", 16))
    depth = int(model_cfg.get("depth", 2))
    model = PostHocBVAwareScore(
        plain_backbone=plain_backbone,
        in_channels=in_channels,
        sigma_data=0.5,
        phi_sh_dim=phi_sh_dim,
        kappa_dim=kappa_dim,
        depth=depth,
    )
    model = model.to(device)

    # 仅训 phi_sh + kappa
    trainable_params = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable={n_trainable:,} ({n_trainable/1e3:.1f}K) / total={n_total:,} ({n_total/1e6:.2f}M)")
    print(f"[model] frozen ratio = {(1 - n_trainable/n_total)*100:.1f}%")

    optimizer = optim.Adam(trainable_params, lr=lr)
    schedule = ViscosityMatchedSchedule(nu=nu, tau_max=tau_max)

    # ---- W5 ext: EMA shadow weights (默认开启, 可在 yaml 关) ----
    use_ema = bool(exp_cfg.get("use_ema", True))
    ema_decay = float(exp_cfg.get("ema_decay", 0.9995))
    if use_ema:
        import copy
        ema_phi_sh = copy.deepcopy(model.phi_sh_net).to(device)
        ema_kappa = copy.deepcopy(model.kappa_net).to(device)
        for p in ema_phi_sh.parameters():
            p.requires_grad_(False)
        for p in ema_kappa.parameters():
            p.requires_grad_(False)
        print(f"[ema] enabled, decay={ema_decay}")
    else:
        ema_phi_sh = None
        ema_kappa = None

    # ---- W5 ext: data augmentation 配置 ----
    if aug_cfg is not None:
        ic_scale_min = float(aug_cfg.get("ic_scale_min", 1.0))
        ic_scale_max = float(aug_cfg.get("ic_scale_max", 1.0))
        ic_noise_sigma = float(aug_cfg.get("ic_noise_sigma", 0.0))
        target_scale_with_ic = bool(aug_cfg.get("target_scale_with_ic", True))
        print(f"[aug] enabled: ic_scale=[{ic_scale_min},{ic_scale_max}], noise={ic_noise_sigma}, target_scale={target_scale_with_ic}")
    else:
        ic_scale_min = ic_scale_max = 1.0
        ic_noise_sigma = 0.0
        target_scale_with_ic = False
        print("[aug] disabled")

    # ---- 日志 ----
    log_path = output_dir / f"train_log_{exp_name}_{run_timestamp}.txt"
    log_fp = open(str(log_path), "w", encoding="utf-8")
    log_fp.write(f"# PostHoc BV-aware Training Log\n")
    log_fp.write(f"# config: {config_path}\n")
    log_fp.write(f"# trainable: {n_trainable} ({n_trainable/1e3:.1f}K)\n")
    log_fp.write(f"# epoch  loss_dsm\n")
    log_fp.flush()

    # ---- 训练循环 ----
    print(f"[train] start (epoch 1 → {epochs})")
    for epoch in range(epochs):
        model.train()  # phi_sh + kappa 训练模式; plain backbone 已 eval()
        # 但 PostHocBVAwareScore.__init__ 已把 plain_backbone.eval() 设过, 不会被外层 .train() 翻
        # 重新强制
        model.plain_backbone.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x_target = batch["x_target"].to(device)
            cond = train_dataset.get_conditioning(batch).to(device) if train_dataset.get_conditioning(batch) is not None else None
            pde_id = batch["pde_id"].to(device)

            B = x_target.shape[0]
            sigmas = schedule.sample_sigma(B, device)

            # ---- W5 ext: data augmentation (per-batch) ----
            if aug_cfg is not None and (ic_scale_min < 1.0 or ic_scale_max > 1.0 or ic_noise_sigma > 0):
                if cond is not None:
                    scale = torch.empty(B, 1, 1, device=device).uniform_(ic_scale_min, ic_scale_max)
                    cond = cond * scale
                    if target_scale_with_ic:
                        # Burgers 非线性使该缩放不严格保解, 但 PostHoc-B 是修正小网络, OK
                        x_target = x_target * scale
                    if ic_noise_sigma > 0:
                        cond = cond + ic_noise_sigma * torch.randn_like(cond)
                else:
                    if ic_noise_sigma > 0:
                        x_target = x_target + ic_noise_sigma * torch.randn_like(x_target)

            # L_DSM: 用 PostHocBVAwareScore 当 model
            loss = get_dsm_loss(model, x_target, sigmas, conditioning=cond, pde_id=pde_id)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # ---- W5 ext: EMA update ----
            if use_ema:
                with torch.no_grad():
                    for ema_p, p in zip(ema_phi_sh.parameters(), model.phi_sh_net.parameters()):
                        ema_p.mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)
                    for ema_p, p in zip(ema_kappa.parameters(), model.kappa_net.parameters()):
                        ema_p.mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        line = f"Epoch {epoch+1}/{epochs} | DSM: {avg:.6f}"
        print(line)
        log_fp.write(f"{epoch+1:4d}  {avg:.8f}\n")
        log_fp.flush()

        if (epoch + 1) % ckpt_every == 0 or (epoch + 1) == epochs:
            ckpt_file = output_dir / f"posthoc_b_{exp_name}{seed_tag}_{run_timestamp}_ep{epoch+1}.pt"
            ckpt_dict = {
                "phi_sh_net": model.phi_sh_net.state_dict(),
                "kappa_net": model.kappa_net.state_dict(),
                "epoch": epoch + 1,
                "config": cfg,
                "seed": args.seed,
                "pde_names": train_dataset.pde_names,
                "plain_ckpt_path": str(plain_ckpt_path),
            }
            # ---- W5 ext: 同时保存 EMA shadow weights ----
            if use_ema:
                ckpt_dict["phi_sh_net_ema"] = ema_phi_sh.state_dict()
                ckpt_dict["kappa_net_ema"] = ema_kappa.state_dict()
                ckpt_dict["ema_decay"] = ema_decay
            torch.save(ckpt_dict, str(ckpt_file))
            print(f"  [ckpt] {ckpt_file}")

    log_fp.close()
    print(f"[train] done. Log: {log_path}")


if __name__ == "__main__":
    main()
