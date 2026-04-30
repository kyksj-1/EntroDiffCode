import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

# 将 PROJECT/black/ 加入 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.data.burgers_dataset import BurgersDataset
from src.models.score_param import StandardScore
from src.diffusion.schedules import BaselineSchedule  # 纯 EDM Schedule (log-normal sigma sampling)
from src.diffusion.losses import get_dsm_loss         # 仅用到 Denoising Score Matching 损失

import yaml
import argparse

def train_baseline():
    """
    纯 EDM baseline 训练脚本。
    目的: 提供一个无 viscosity-matched schedule 和 BV loss 的对比线，
          用于验证 Entropy-aware diffusion (EntroDiff) 对 shock 的保护效果。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment/mvp_burgers.yaml")
    args = parser.parse_args()
    
    device = torch.device(env.default_device)
    batch_size = env._config["hardware"].get("max_batch_size", 64)
    num_workers = env.num_workers
    
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)["experiment"]

    data_path = env.data_dir / exp_cfg.get("data_file", "burgers_1d_N5000_Nx128.npy")

    # 实验命名: 专门使用 baseline 目录
    exp_name = "mvp_baseline"
    output_dir = env.output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = exp_cfg.get("epochs", 10)
    lr = float(exp_cfg.get("learning_rate", 2e-4))
    lambda_dsm = float(exp_cfg.get("lambda_dsm", 1.0))

    if not data_path.exists():
        print(f"Data not found at {data_path} for baseline. Run generate_data.py first.")
        return

    print("Loading Dataset for Baseline...")
    train_dataset = BurgersDataset(data_path, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print(f"Initializing EDM Baseline Model... (lr={lr})")
    # Backbone 使用同样的 1D U-Net，但损失和 schedule 为普通 EDM
    model = StandardScore(in_channels=2).to(device)  # IC-conditioned
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 使用 BaselineSchedule: EDM 标准 log-normal sigma 采样 (无视物理ν, τ)
    # 取 EDM 论文推荐参数 P_mean=-1.2, P_std=1.2
    schedule = BaselineSchedule()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- 训练日志文件 ----
    # 每次运行独立日志, 文件名带时间戳不覆盖
    log_path = output_dir / f"train_log_{exp_name}_{run_timestamp}.txt"
    log_fp = open(str(log_path), "w", encoding="utf-8")
    # 写日志头: 运行元信息
    log_fp.write(f"# EntroDiff Baseline (Pure EDM) Training Log\n")
    log_fp.write(f"# Time: {run_timestamp}\n")
    log_fp.write(f"# Config: {config_path}\n")
    log_fp.write(f"# Device: {device}  Batch: {batch_size}  Workers: {num_workers}\n")
    log_fp.write(f"# Params: epochs={epochs} lr={lr} lambda_dsm={lambda_dsm}\n")
    log_fp.write(f"# Schedule: BaselineSchedule (log-normal sigma, P_mean=-1.2, P_std=1.2)\n")
    log_fp.write(f"# Data: {data_path}\n")
    log_fp.write(f"# Output: {output_dir}\n")
    log_fp.write(f"#\n")
    log_fp.write(f"# epoch  loss_dsm\n")
    log_fp.flush()  # 立即刷新, 防止中断丢失日志

    print(f"Starting EDM Baseline Training on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            ic = batch[:, 0, :].unsqueeze(1).to(device)          # [B, 1, Nx]
            x_target = batch[:, -1, :].unsqueeze(1).to(device)    # [B, 1, Nx]
            optimizer.zero_grad()
            
            # 使用 BaselineSchedule 获取标准 EDM sigmas
            sigmas = schedule.sample_sigma(x_target.shape[0], device)
            
            # 仅使用 L_DSM (删去了 lambda_bv * L_BV), 含 IC 条件
            loss_dsm = get_dsm_loss(model, x_target, sigmas, ic=ic)
            loss = lambda_dsm * loss_dsm
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        # 同时输出到 stdout 和 log 文件
        line = f"Epoch {epoch+1}/{epochs} | Loss (DSM only): {avg_loss:.6f}"
        print(line)
        # 写入日志文件: epoch loss_dsm (机器可读格式)
        log_fp.write(f"{epoch+1:4d}  {avg_loss:.8f}\n")
        log_fp.flush()  # 每 epoch 刷新, 实时可读

        # ---- Checkpoint 保存 ----
        # 格式必须严格对齐
        if (epoch + 1) % 5 == 0:
            ckpt_path = output_dir / f"entrodiff_{exp_name}_{run_timestamp}_ep{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved baseline checkpoint to {ckpt_path}")

    # ---- 训练结束: 关闭日志 ----
    log_fp.write(f"# Training completed.\n")
    log_fp.write(f"# Final epoch: {epochs}  Final ckpt: {output_dir}\n")
    log_fp.close()
    print(f"Training complete. Log saved to {log_path}")
    print(f"Checkpoints saved to {output_dir}")

if __name__ == "__main__":
    train_baseline()
