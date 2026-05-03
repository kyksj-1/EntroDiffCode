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
    args = parser.parse_args()

    device = torch.device(env.default_device)
    config_path = (PROJECT_ROOT / args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    if model_cfg["type"] != "posthoc_bv":
        raise ValueError(f"This script only handles model.type=posthoc_bv, got {model_cfg['type']}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = exp_cfg["name"]
    output_dir = env.output_dir / exp_name
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

            # L_DSM: 用 PostHocBVAwareScore 当 model
            loss = get_dsm_loss(model, x_target, sigmas, conditioning=cond, pde_id=pde_id)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        line = f"Epoch {epoch+1}/{epochs} | DSM: {avg:.6f}"
        print(line)
        log_fp.write(f"{epoch+1:4d}  {avg:.8f}\n")
        log_fp.flush()

        if (epoch + 1) % ckpt_every == 0 or (epoch + 1) == epochs:
            ckpt_file = output_dir / f"posthoc_b_{exp_name}_{run_timestamp}_ep{epoch+1}.pt"
            # 仅保存 trainable + cfg + plain_ckpt 路径 (避免 ckpt 冗余, 加载时再装载 plain)
            torch.save({
                "phi_sh_net": model.phi_sh_net.state_dict(),
                "kappa_net": model.kappa_net.state_dict(),
                "epoch": epoch + 1,
                "config": cfg,
                "pde_names": train_dataset.pde_names,
                "plain_ckpt_path": str(plain_ckpt_path),
            }, str(ckpt_file))
            print(f"  [ckpt] {ckpt_file}")

    log_fp.close()
    print(f"[train] done. Log: {log_path}")


if __name__ == "__main__":
    main()
