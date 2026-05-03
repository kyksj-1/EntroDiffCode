# ============================================================================
# E3 · 1D Euler Sod StandardScore 训练脚本 (W5-SA2)
# 论文对应: §5.4 + A2.2
#
# 与 train_bvaware_euler.py 的差异:
#   1. 使用 StandardScore (纯 EDM 参数化) 而非 BVAwareScore (建筑先验)
#   2. 多了 EMA (指数移动平均) 以稳定系统级 PDE 训练
#   3. CosineAnnealingLR 调度器 (替代恒定 lr)
#   4. 支持 L_time (Godunov 时间一致性) 损失项
#   5. 输出目录: e3_standard_run (与 bvaware 分离, 便于消融对比)
#
# 现有 train_bvaware_euler.py 不变 (向后兼容铁律).
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

# 将 PROJECT/black/ 加入 sys.path, 确保 src/ 下所有模块可被 import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.data.euler_dataset import EulerDataset
from src.models.score_param import StandardScore  # 纯 EDM 参数化 D_θ(x, σ)
from src.diffusion.schedules import ViscosityMatchedSchedule  # §3.1: σ²(τ)=2ντ
from src.diffusion.losses import get_dsm_loss, get_bv_loss, get_godunov_time_loss  # §3.2/§3.3


def train_euler_standard() -> None:
    """E3 Euler StandardScore 训练主流程."""
    # ========== 0. 命令行参数 ==========
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment/e3_euler.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    parser.add_argument("--dim", type=int, default=128,
                        help="UNet 基础通道数 (PC=64, 服务器=128~256)")
    args = parser.parse_args()

    # ========== 1. 加载配置 ==========
    # 设备: 从 env_config.yaml 读取, 含 CUDA→CPU fallback
    device = torch.device(env.default_device)
    # 批大小: 3 通道守恒变量, 减小默认 (PC 端 RTX 4060 8GB 显存受限)
    batch_size = env._config["hardware"].get("max_batch_size", 32)
    num_workers = env.num_workers  # Windows 下默认 0, 避免 multiprocessing 报错
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 实验超参数 YAML 配置
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)["experiment"]

    # 数据路径: 默认读取 env.data_dir 下的 euler_sod 数据
    data_path = env.data_dir / exp_cfg.get("data_file", "euler_sod_1d_N5000_Nx128.npy")

    # 实验命名与输出目录
    exp_name = exp_cfg.get("name", "e3_standard_run")
    output_dir = env.output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 训练超参数 (均从 YAML 读取, 含默认值兜底) ----
    epochs = int(exp_cfg.get("epochs", 200))
    lr = float(exp_cfg.get("learning_rate", 2e-4))
    # ν (nu): 物理粘度系数, 控制扩散调度 §3.1 方程: σ²(τ) = 2ντ
    nu = float(exp_cfg.get("nu", 1.0))
    # τ_max: 最大扩散时间 τ ∈ [0, τ_max]
    tau_max = float(exp_cfg.get("tau_max", 1.0))
    # λ_dsm: DSM 损失权重 (通常固定为 1.0)
    lambda_dsm = float(exp_cfg.get("lambda_dsm", 1.0))
    # λ_bv: BV/TV 正则化权重 §3.3
    lambda_bv = float(exp_cfg.get("lambda_bv", 0.1))
    # λ_time: Godunov 时间一致性损失权重 (0 = 关闭; 论文 §3.4)
    lambda_time = float(exp_cfg.get("lambda_time", 0.0))
    # EMA 衰减率 (0 = 不启用 EMA)
    ema_decay = float(exp_cfg.get("ema_decay", 0.9999))
    # Grad clipping max_norm
    grad_clip = float(exp_cfg.get("grad_clip", 1.0))

    # ---- 模型结构参数 ----
    in_channels = int(exp_cfg.get("in_channels", 6))      # 3 noisy + 3 IC
    out_channels = int(exp_cfg.get("out_channels", 3))    # 3 守恒变量 (ρ, ρu, E)
    n_components = int(exp_cfg.get("n_components", 3))
    use_ic = bool(exp_cfg.get("use_ic", True))

    # 数据文件缺失时提前退出, 提示先运行 generate_euler_data.py
    if not data_path.exists():
        print(f"数据不存在: {data_path}, 请先运行 generate_euler_data.py")
        return

    # ========== 2. 数据加载 ==========
    # EulerDataset: 封装 1D Euler 系统守恒变量 (ρ, ρu, E)
    #   __getitem__ 返回 (N_time, n_components, Nx) — 已做 transpose
    #   DataLoader stack 后: (B, N_time, n_components, Nx)
    #   get_conditioning: 返回首帧 IC → (B, n_components, Nx)
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
    # StandardScore: 纯 EDM 参数化 D_θ(x, σ)
    #   - backbone: UNet1D (1D U-Net)
    #   - 前向: D_x = c_skip·x + c_out·F_θ(c_in·x, c_noise(σ))
    #   - in_channels=6:  前 3 通道 = noisy 守恒变量, 后 3 通道 = IC 条件
    #   - out_channels=3: 输出 3 通道 (ρ, ρu, E) 的洁净估计
    #   - 论文对应: 03_method.tex §3.2
    print(
        f"初始化 StandardScore (dim={args.dim}, in_channels={in_channels}, "
        f"out_channels={out_channels}, nu={nu})"
    )
    model = StandardScore(
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # CosineAnnealingLR: 学习率余弦退火 (T_max=epochs, 从 lr → 0 平滑衰减)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # ViscosityMatchedSchedule: σ²(τ) = 2ντ  (论文 §3.1)
    schedule = ViscosityMatchedSchedule(nu=nu, tau_max=tau_max)

    # ========== 3a. EMA 权重初始化 ==========
    # 手动 EMA: 不依赖 ema_pytorch 库, 直接用 register_buffer
    # 逻辑: 首次调用时 deep copy; 后续: θ_ema ← ema_decay·θ_ema + (1-ema_decay)·θ
    ema_enabled = ema_decay > 0
    model._ema_initialized = False

    # ========== 4. Resume 恢复逻辑 ==========
    start_epoch = 0
    log_path = output_dir / f"train_log_{exp_name}_{run_timestamp}.txt"
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint 不存在: {ckpt_path}")
        print(f"[resume] 加载模型权重: {ckpt_path}")
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        # 从文件名提取 epoch 编号: xxx_epN.pt → N
        m = re.search(r'_ep(\d+)', ckpt_path.name)
        start_epoch = int(m.group(1)) if m else 0
        print(f"[resume] 从 epoch {start_epoch} 继续 (优化器/scheduler 从零开始)")
        # 日志以追加模式打开, 不重复写文件头
        log_fp = open(str(log_path), "a", encoding="utf-8")
        log_fp.write(f"# [resume] 从 epoch {start_epoch} 恢复训练 at {run_timestamp}\n")
        log_fp.flush()
    else:
        log_fp = open(str(log_path), "w", encoding="utf-8")
        # 写日志头: 运行元信息
        log_fp.write(f"# E3 Euler StandardScore Training Log\n")
        log_fp.write(f"# Time: {run_timestamp}\n")
        log_fp.write(f"# Config: {config_path}\n")
        log_fp.write(f"# Model: StandardScore dim={args.dim} in={in_channels} out={out_channels}\n")
        log_fp.write(f"# Device: {device}  Batch: {batch_size}  Workers: {num_workers}\n")
        log_fp.write(f"# Params: epochs={epochs} lr={lr} nu={nu} tau_max={tau_max}\n")
        log_fp.write(f"# Loss weights: λ_dsm={lambda_dsm} λ_bv={lambda_bv} λ_time={lambda_time}\n")
        log_fp.write(f"# EMA: decay={ema_decay}  GradClip: {grad_clip}\n")
        log_fp.write(f"# Conditioning: type={cond_type} in_channels={in_channels}\n")
        log_fp.write(f"# Data: {data_path}\n")
        log_fp.write(f"# Output: {output_dir}\n")
        log_fp.write(f"#\n")
        log_fp.write(f"# epoch  loss_total  loss_dsm  loss_bv  loss_time\n")
        log_fp.flush()
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 5. 训练主循环 ==========
    # 物理参数 (与数据生成对齐, 用于 Godunov 时间推进)
    dt_phys = 0.005          # 数据生成时的 dt
    dx_phys = 2.0 / 128      # 数据生成时的 dx (周期域 [-1,1], Nx=128)
    print(f"开始训练 (epoch {start_epoch+1} → {epochs}, device={device})...")
    print(f"  λ_dsm={lambda_dsm}  λ_bv={lambda_bv}  λ_time={lambda_time}")
    print(f"  EMA={ema_enabled}  CosineLR  GradClip={grad_clip}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, total_dsm, total_bv, total_time = 0.0, 0.0, 0.0, 0.0

        for batch in train_loader:
            # batch: (B, N_time, n_components, Nx)
            # 提取 IC 条件: 首帧 3 通道 → (B, 3, Nx)
            cond = train_dataset.get_conditioning(batch)
            if cond is not None:
                cond = cond.to(device)
            # 末帧作为 target 解分布 ρ_T(u): (B, 3, Nx)
            x_target = batch[:, -1, :, :].to(device)

            optimizer.zero_grad()

            # 采样连续时间 σ ~ viscosity-matched schedule
            sigmas = schedule.sample_sigma(x_target.shape[0], device)

            # ---- L_DSM: Denoising Score Matching (论文 §3.2) ----
            # EDM 参数化: E[‖D_θ(x+σε, σ) - x‖²]
            # 实现于 losses.py: get_dsm_loss()
            loss_dsm = get_dsm_loss(model, x_target, sigmas, conditioning=cond)

            # ---- L_BV: Total-Variation 代理损失 (论文 §3.3) ----
            # 约束解空间在 BV(Ω) 内, 保证 Helly 定理下的 L¹ 紧性
            loss_bv = get_bv_loss(model, x_target, sigmas, conditioning=cond)

            # 联合损失: L = λ_dsm·L_DSM + λ_bv·L_BV
            loss = lambda_dsm * loss_dsm + lambda_bv * loss_bv

            # ---- L_time: Godunov 时间一致性 (论文 §3.4) ----
            # 强制去噪输出的一步 Godunov 推进接近真值次帧
            loss_time_val = torch.tensor(0.0, device=device)
            if lambda_time > 0:
                # 次末帧作为 "前一步" 状态 → (B, 3, Nx)
                x_prev = batch[:, -2, :, :].to(device)
                loss_time_val = get_godunov_time_loss(
                    model, x_prev, x_target, sigmas,
                    dt=dt_phys, dx=dx_phys, conditioning=cond,
                )
                loss = loss + lambda_time * loss_time_val

            # 反向传播 + 梯度裁剪 + 参数更新
            loss.backward()
            # W5-NaN 防御 (2026-05-02): grad clipping 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            # ---- EMA 权重更新 (in-batch, 确保每步都更新) ----
            if ema_enabled:
                if not model._ema_initialized:
                    # 首次: deep copy 全部参数到 ema 缓冲
                    for name, param in model.named_parameters():
                        model.register_buffer(
                            f'ema_{name.replace(".", "_")}',
                            param.data.clone()
                        )
                    model._ema_initialized = True
                else:
                    for name, param in model.named_parameters():
                        ema_buf = getattr(model, f'ema_{name.replace(".", "_")}')
                        ema_buf.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

            # 累加各分量损失 (用于 epoch 结束时输出平均日志)
            total_loss += loss.item()
            total_dsm += loss_dsm.item()
            total_bv += loss_bv.item()
            total_time += loss_time_val.item()

        # ---- Epoch 结束: Cosine LR scheduler step ----
        scheduler.step()

        # ---- 日志输出 ----
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_dsm = total_dsm / n_batches
        avg_bv = total_bv / n_batches
        avg_time = total_time / n_batches
        current_lr = scheduler.get_last_lr()[0]

        line = (
            f"Epoch {epoch+1}/{epochs} | L: {avg_loss:.6f} "
            f"DSM: {avg_dsm:.6f}  BV: {avg_bv:.6f}  Time: {avg_time:.6f} "
            f"lr: {current_lr:.2e}"
        )
        print(line)
        log_fp.write(
            f"{epoch+1:4d}  {avg_loss:.8f}  {avg_dsm:.8f}  {avg_bv:.8f}  {avg_time:.8f}\n"
        )
        log_fp.flush()

        # ---- Checkpoint 保存 (每 5 epoch) ----
        if (epoch + 1) % 5 == 0:
            # 常规 checkpoint: 保存当前模型权重
            ckpt_file = (
                output_dir
                / f"entrodiff_{exp_name}_{run_timestamp}_ep{epoch+1}.pt"
            )
            torch.save(model.state_dict(), ckpt_file)
            print(f"  → ckpt: {ckpt_file}")

            # EMA checkpoint: 保存 EMA 权重 (仅当 EMA 已初始化)
            if ema_enabled and model._ema_initialized:
                ema_state = {}
                for name, _param in model.named_parameters():
                    ema_key = f'ema_{name.replace(".", "_")}'
                    ema_state[name] = getattr(model, ema_key).clone()
                ema_ckpt_file = (
                    output_dir
                    / f"entrodiff_{exp_name}_{run_timestamp}_ep{epoch+1}_ema.pt"
                )
                torch.save(ema_state, ema_ckpt_file)
                print(f"  → ema ckpt: {ema_ckpt_file}")

    # ---- 训练结束: 保存最终 checkpoint ----
    final_ckpt = output_dir / f"entrodiff_{exp_name}_{run_timestamp}_final.pt"
    torch.save(model.state_dict(), final_ckpt)
    print(f"最终模型保存: {final_ckpt}")
    if ema_enabled and model._ema_initialized:
        ema_state = {}
        for name, _param in model.named_parameters():
            ema_key = f'ema_{name.replace(".", "_")}'
            ema_state[name] = getattr(model, ema_key).clone()
        final_ema_ckpt = output_dir / f"entrodiff_{exp_name}_{run_timestamp}_final_ema.pt"
        torch.save(ema_state, final_ema_ckpt)
        print(f"最终 EMA 保存: {final_ema_ckpt}")

    log_fp.write(f"# 训练完成. final_epoch={epochs}\n")
    log_fp.close()
    print(f"训练完成. 日志: {log_path}")


if __name__ == "__main__":
    train_euler_standard()
