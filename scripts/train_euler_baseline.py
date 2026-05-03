import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from copy import deepcopy

# 将 PROJECT/black/ 加入 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.data.euler_dataset import EulerDataset
from src.models.score_param import StandardScore
from src.diffusion.schedules import BaselineSchedule  # 纯 EDM log-normal sigma 采样
from src.diffusion.losses import get_dsm_loss           # 仅用到 Denoising Score Matching 损失

import yaml
import argparse


def train_euler_baseline():
    """
    E3 Euler Sod 纯 EDM Baseline 训练脚本.

    目的: 为 Euler Sod 三组分系统 (ρ, ρu, E) 训练一个无建筑先验的纯 EDM baseline,
          用于对比验证 BVAwareScore 对 shock 和组分间耦合的保护效果.

    与 train_baseline.py (Burgers) 的关键区别:
      - 数据: EulerDataset 返回 [B, N_t, 3, Nx] (3 守恒变量)
      - 模型: StandardScore(in_channels=6, out_channels=3)
              输入 = cat([noisy_u(3ch), IC(3ch)], dim=1) → 6 in; 输出 3 ch
      - 无 BV loss, 无 viscosity-matched schedule → 纯 EDM
    """
    parser = argparse.ArgumentParser(description="E3 Euler Sod EDM Baseline Training")
    parser.add_argument("--config", type=str,
                        default="configs/experiment/e3_euler.yaml",
                        help="实验配置文件路径 (YAML)")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的 checkpoint 路径 (会加载 model/optim/scheduler/ema/epoch)")
    args = parser.parse_args()

    # ---- 设备与环境 ----
    device = torch.device(env.default_device)
    batch_size = env._config["hardware"].get("max_batch_size", 64)  # 默认 64, 系统级可能需调小
    num_workers = env.num_workers

    # ---- 加载实验配置 ----
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)["experiment"]

    data_path = env.data_dir / exp_cfg.get("data_file", "euler_sod_1d_N5000_Nx128.npy")

    # 实验名强制覆盖为 "e3_baseline" (与主实验 e3_euler_run 区分)
    exp_name = "e3_baseline"
    output_dir = env.output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 训练超参 ----
    epochs = exp_cfg.get("epochs", 10)
    lr = float(exp_cfg.get("learning_rate", 2e-4))
    lambda_dsm = float(exp_cfg.get("lambda_dsm", 1.0))
    # n_components 从 config 取 (默认 3: ρ, ρu, E)
    n_components = exp_cfg.get("n_components", 3)
    # 梯度裁剪阈值
    grad_clip_norm = float(exp_cfg.get("grad_clip_norm", 1.0))
    # EMA 衰减系数
    ema_decay = float(exp_cfg.get("ema_decay", 0.999))
    # ckpt 保存周期
    ckpt_every = exp_cfg.get("ckpt_every", 5)

    if not data_path.exists():
        print(f"[ERROR] 数据文件不存在: {data_path}")
        print("  请先运行 generate_euler_data.py 生成数据.")
        return

    # ---- 数据加载 ----
    print(f"[Data] 加载 Euler 数据集: {data_path}")
    train_dataset = EulerDataset(data_path, mode='train', n_components=n_components)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # ---- 模型初始化 ----
    # StandardScore  输入: cat([noisy(3ch), IC(3ch)]) = 6 ch
    #                 输出: 3 ch (ρ, ρu, E 去噪预测)
    print(f"[Model] 初始化 StandardScore (in_channels=6, out_channels={n_components})")
    model = StandardScore(in_channels=2 * n_components, out_channels=n_components).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {param_count:,}")

    # ---- 优化器 ----
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---- Cosine 学习率调度器 ----
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ---- Schedule: 纯 EDM log-normal sigma 采样 ----
    schedule = BaselineSchedule()

    # ---- 手动 EMA (Exponential Moving Average) ----
    ema_model = deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # ---- 恢复训练 ----
    start_epoch = 0
    if args.resume:
        print(f"[Resume] 从 checkpoint 恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        ema_model.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt["epoch"]  # checkpoint 保存的 epoch 已完成, 从下一个 epoch 继续
        run_timestamp = ckpt.get("run_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
        print(f"  恢复至 epoch {start_epoch}, 将继续训练 epoch {start_epoch+1}~{epochs}")
    else:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- 训练日志文件 ----
    log_path = output_dir / f"train_log_{exp_name}_{run_timestamp}.txt"
    log_fp = open(str(log_path), "w", encoding="utf-8")
    log_fp.write(f"# EntroDiff Euler E3 Baseline (Pure EDM) Training Log\n")
    log_fp.write(f"# Time: {run_timestamp}\n")
    log_fp.write(f"# Config: {config_path}\n")
    log_fp.write(f"# Device: {device}  Batch: {batch_size}  Workers: {num_workers}\n")
    log_fp.write(f"# Params: epochs={epochs} lr={lr} lambda_dsm={lambda_dsm}\n")
    log_fp.write(f"# Model: StandardScore in=6 out=3  Params: {param_count:,}\n")
    log_fp.write(f"# Schedule: BaselineSchedule (log-normal sigma, P_mean=-1.2, P_std=1.2)\n")
    log_fp.write(f"# Loss: L_DSM only (no BV, no time)\n")
    log_fp.write(f"# EMA: decay={ema_decay}  GradClip: max_norm={grad_clip_norm}\n")
    log_fp.write(f"# LR Scheduler: CosineAnnealingLR(T_max={epochs})\n")
    log_fp.write(f"# Data: {data_path}\n")
    log_fp.write(f"# Output: {output_dir}\n")
    if args.resume:
        log_fp.write(f"# Resume from: {args.resume}  start_epoch={start_epoch}\n")
    log_fp.write(f"#\n")
    log_fp.write(f"# epoch    loss_dsm    lr\n")
    log_fp.flush()

    # ---- 训练循环 ----
    print(f"\n[Train] 开始 Euler E3 EDM Baseline 训练 (epoch {start_epoch+1}~{epochs})")
    print(f"  Device: {device}  Batch: {batch_size}  Epochs: {epochs}  LR: {lr}")
    print(f"  Output: {output_dir}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            # batch shape: [B, N_t, 3, N_x]
            batch = batch.to(device)

            # 提取目标 (末帧) 和初始条件 (首帧)
            #   x_target: [B, 3, N_x]  — 末帧 3 守恒变量 (ρ, ρu, E)
            #   x_IC:     [B, 3, N_x]  — 首帧 3 守恒变量 (IC)
            x_target = batch[:, -1, :, :]   # 末帧
            x_IC = batch[:, 0, :, :]        # 首帧

            optimizer.zero_grad()

            # 使用 BaselineSchedule 获取标准 EDM sigmas (log-normal sampling)
            # 论文对照: EDM §3.1, ln(σ) ~ N(P_mean, P_std)
            sigmas = schedule.sample_sigma(x_target.shape[0], device)

            # 计算 L_DSM (仅去噪分数匹配, 无 BV / 时间损失)
            # get_dsm_loss 内部: x_noisy = x_target + σ·noise;
            #   cat([x_noisy, x_IC], dim=1) → [B, 6, Nx] 送入模型;
            #   模型输出 [B, 3, Nx] → MSE 对 x_target
            loss_dsm = get_dsm_loss(model, x_target, sigmas, conditioning=x_IC)
            loss = lambda_dsm * loss_dsm

            loss.backward()

            # 梯度裁剪: 防止 NaN (W5-NaN 防御)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            # ---- 手动 EMA 更新 ----
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1.0 - ema_decay)

            total_loss += loss.item()

        # 每个 epoch 后步进 Cosine 学习率调度器
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        avg_loss = total_loss / len(train_loader)

        # 输出到 stdout 和日志
        line = f"Epoch {epoch+1:3d}/{epochs} | Loss (DSM only): {avg_loss:.6f} | LR: {current_lr:.2e}"
        print(line)
        log_fp.write(f"{epoch+1:4d}  {avg_loss:.8f}  {current_lr:.6e}\n")
        log_fp.flush()

        # ---- Checkpoint 保存 ----
        if (epoch + 1) % ckpt_every == 0:
            # 常规 checkpoint
            ckpt_path = output_dir / f"euler_baseline_{run_timestamp}_ep{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,                           # 已完成 epoch 编号
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "run_timestamp": run_timestamp,
                "config_path": str(config_path),
                "exp_name": exp_name,
            }, ckpt_path)
            print(f"  Saved regular ckpt: {ckpt_path}")

            # EMA checkpoint (独立保存, 仅模型权重, 评估时直接加载)
            ema_ckpt_path = output_dir / f"euler_baseline_ema_{run_timestamp}_ep{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": ema_model.state_dict(),
                "run_timestamp": run_timestamp,
                "exp_name": exp_name,
            }, ema_ckpt_path)
            print(f"  Saved EMA ckpt:    {ema_ckpt_path}")

    # ---- 训练结束: 关闭日志 ----
    log_fp.write(f"# Training completed.\n")
    log_fp.write(f"# Final epoch: {epochs}\n")
    log_fp.write(f"# Checkpoints: {output_dir}\n")
    log_fp.close()

    print(f"\n[Done] 训练完成.")
    print(f"  Log:  {log_path}")
    print(f"  Ckpt: {output_dir}")


if __name__ == "__main__":
    train_euler_baseline()
