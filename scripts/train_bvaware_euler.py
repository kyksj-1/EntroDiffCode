# ============================================================================
# E3 · 1D Euler Sod BVAware 训练脚本 (W5-SA2)
# 论文对应: §5.4 + A2.2
#
# 与 train_bvaware.py 的差异:
#   1. 数据集: EulerDataset (3 通道守恒变量) 而非 BurgersDataset
#   2. 模型: BVAwareScore(in_channels=6, out_channels=3) — 3 noisy + 3 IC, 输出 3 通道
#   3. x_target: trajectory[:, -1, :, :] 末帧 3 通道 (而非 batch[:, -1, :].unsqueeze(1))
#   4. cond: 3 通道 IC
#
# 现有 train_bvaware.py 不变 (向后兼容铁律).
# ============================================================================
import os
import sys
import re
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.data.euler_dataset import EulerDataset
from src.models.score_param import BVAwareScore
from src.diffusion.schedules import ViscosityMatchedSchedule
from src.diffusion.losses import get_dsm_loss, get_bv_loss


def train_euler_bvaware() -> None:
    """E3 Euler 训练主流程."""
    # ========== 0. 命令行参数 ==========
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment/e3_euler.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()

    # ========== 1. 加载配置 ==========
    device = torch.device(env.default_device)
    batch_size = env._config["hardware"].get("max_batch_size", 32)   # 3 通道, 减小默认
    num_workers = env.num_workers
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)["experiment"]

    data_path = env.data_dir / exp_cfg.get("data_file", "euler_sod_1d_N5000_Nx128.npy")

    exp_name = exp_cfg.get("name", "e3_euler_run")
    output_dir = env.output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(exp_cfg.get("epochs", 200))
    lr = float(exp_cfg.get("learning_rate", 2e-4))
    nu = float(exp_cfg.get("nu", 1.0))
    tau_max = float(exp_cfg.get("tau_max", 1.0))
    lambda_bv = float(exp_cfg.get("lambda_bv", 0.1))
    lambda_dsm = float(exp_cfg.get("lambda_dsm", 1.0))

    in_channels = int(exp_cfg.get("in_channels", 6))      # 3 noisy + 3 IC
    out_channels = int(exp_cfg.get("out_channels", 3))    # 3 守恒变量
    n_components = int(exp_cfg.get("n_components", 3))
    use_ic = bool(exp_cfg.get("use_ic", True))

    if not data_path.exists():
        print(f"数据不存在: {data_path}, 请先运行 generate_euler_data.py")
        return

    # ========== 2. 数据加载 ==========
    print(f"加载数据: {data_path}")
    cond_type = "ic" if use_ic else "none"
    train_dataset = EulerDataset(
        data_path, mode="train",
        conditioning_type=cond_type,
        n_components=n_components,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
    )

    # ========== 3. 模型初始化 ==========
    # BVAwareScore: 系统级 (out_channels=3 for Euler)
    print(
        f"初始化 BVAwareScore (dim={args.dim}, in_channels={in_channels}, "
        f"out_channels={out_channels}, nu={nu}, λ_bv={lambda_bv})"
    )
    model = BVAwareScore(
        in_channels=in_channels,
        dim=args.dim,
        return_denoiser=True,
        out_channels=out_channels,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    schedule = ViscosityMatchedSchedule(nu=nu, tau_max=tau_max)

    # Resume
    start_epoch = 0
    log_path = output_dir / f"train_log_{exp_name}_{run_timestamp}.txt"
    if args.resume:
        ckpt_path = Path(args.resume)
        print(f"[resume] 加载: {ckpt_path}")
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        m = re.search(r"_ep(\d+)", ckpt_path.name)
        start_epoch = int(m.group(1)) if m else 0

    log_fp = open(str(log_path), "a" if args.resume else "w", encoding="utf-8")
    if not args.resume:
        log_fp.write(f"# E3 Euler BVAware Training Log\n")
        log_fp.write(
            f"# Model: BVAwareScore dim={args.dim} in={in_channels} out={out_channels}\n"
        )
        log_fp.write(f"# epoch  loss_total  loss_dsm  loss_bv\n")
    log_fp.flush()
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 4. 训练主循环 ==========
    print(f"开始训练 (epoch {start_epoch+1} → {epochs}, device={device})...")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, total_dsm, total_bv = 0.0, 0.0, 0.0

        for batch in train_loader:
            # batch: (B, N_time, 3, Nx)
            cond = train_dataset.get_conditioning(batch)   # (B, 3, Nx) or None
            if cond is not None:
                cond = cond.to(device)
            # 末帧 3 通道作为目标
            x_target = batch[:, -1, :, :].to(device)        # (B, 3, Nx)

            optimizer.zero_grad()
            sigmas = schedule.sample_sigma(x_target.shape[0], device)

            # L_DSM + L_BV (复用现有, conditioning 现支持任意通道)
            loss_dsm = get_dsm_loss(model, x_target, sigmas, conditioning=cond)
            loss_bv = get_bv_loss(model, x_target, sigmas, conditioning=cond)

            loss = lambda_dsm * loss_dsm + lambda_bv * loss_bv
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dsm += loss_dsm.item()
            total_bv += loss_bv.item()

        n = len(train_loader)
        avg_loss, avg_dsm, avg_bv = total_loss / n, total_dsm / n, total_bv / n
        line = (
            f"Epoch {epoch+1}/{epochs} | L: {avg_loss:.6f} "
            f"DSM: {avg_dsm:.6f}  BV: {avg_bv:.6f}"
        )
        print(line)
        log_fp.write(
            f"{epoch+1:4d}  {avg_loss:.8f}  {avg_dsm:.8f}  {avg_bv:.8f}\n"
        )
        log_fp.flush()

        if (epoch + 1) % 5 == 0:
            ckpt_file = (
                output_dir
                / f"entrodiff_{exp_name}_{run_timestamp}_ep{epoch+1}.pt"
            )
            torch.save(model.state_dict(), ckpt_file)
            print(f"  → ckpt: {ckpt_file}")

    log_fp.write(f"# 训练完成. final_epoch={epochs}\n")
    log_fp.close()
    print(f"训练完成. 日志: {log_path}")


if __name__ == "__main__":
    train_euler_bvaware()
