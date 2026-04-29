# ============================================================================
# EntroDiff MVP 训练主脚本
# 论文对应: 03_method.tex §3.1-§3.3
# 功能: 端到端训练 EntroDiff 的最小可行产品 (Minimum Viable Product)
#       在 1D Inviscid Burgers Equation 上验证 shock 捕捉能力
#       对比基线: 纯 EDM (L_DSM) vs Ours (L_DSM + L_BV + Viscosity-matched
#                  schedule + Godunov guidance)
# ============================================================================
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# 将 PROJECT/black/ 加入 sys.path，确保 src/ 下所有模块可被 import
# 本文件位于 PROJECT/black/scripts/，parent 即 PROJECT/black/
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT  # 跨环境单例配置管理器 (env_config.yaml)
from src.data.burgers_dataset import BurgersDataset   # 1D Burgers 方程数据加载器 (Godunov 真值解)
from src.models.score_param import StandardScore      # 标准 EDM 参数化: D_θ(x, σ) → 预估洁净态
from src.diffusion.schedules import ViscosityMatchedSchedule  # 论文 §3.1: σ²(τ) = 2ντ 的物理匹配调度
from src.diffusion.losses import get_dsm_loss, get_bv_loss     # 论文 §3.2 / §3.3: L_DSM + L_BV 损失函数

import yaml      # 读取实验超参数 YAML 配置
import argparse  # 命令行参数解析 (--config 选择实验配置文件)
from datetime import datetime # 添加时间戳，防止 checkpoint 相互覆盖

def train_mvp():
    """
    EntroDiff MVP 训练主流程。

    三阶段:
      1. 加载 1D Burgers 数据  (Godunov 求解器生成的真值)
      2. 使用 StandardScore (1D U-Net backbone, EDM 参数化) 作为 score 估计器
      3. 联合优化 L = λ_dsm * L_DSM + λ_bv * L_BV
         - L_DSM: Denoising Score Matching, 驱动解分布向物理真值收敛
         - L_BV:  Total-Variation 代理损失, 限制解空间在 BV(Ω) 内,
                 保证 Helly 定理下的 L¹ 紧性 (论文 §3.3)

    对比基线设计:
      Baseline = λ_bv=0, 纯 EDM 调度  → 验证无 BV 约束时 shock 处是否模糊
      Ours     = λ_bv>0 + 粘度匹配调度 → 验证熵约束对 shock 陡度的保持效果
    """
    # ========== 0. 命令行参数: 指定实验 YAML 配置文件路径 ==========
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment/mvp_burgers.yaml",
                        help="实验超参数配置文件 (相对于 PROJECT/black/ 的路径)")
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 恢复训练 (e.g. entrodiff_xxx_ep10.pt)")
    args = parser.parse_args()
    
    # ========== 1. 加载环境与实验配置 ==========

    # 硬件配置: 从 env_config.yaml 读取，设备名含 fallback (CUDA 不可用时自动降级 CPU)
    device = torch.device(env.default_device)
    # 批大小: 从硬件配置读取 max_batch_size，默认 64 (PC 端 RTX 4060 8GB 显存受限)
    batch_size = env._config["hardware"].get("max_batch_size", 64)
    # num_workers: Windows 下默认 0 (避免 multiprocessing 问题)，Linux/Colab 可调
    num_workers = env.num_workers
    # 数据路径: 由 env_manager 将 data_dir 解析为绝对路径，保证跨环境一致性
    # 预期数据文件: burg1d_N5000_Nx128.npy, shape [N_samples, N_time, N_x]
    #              每行是一条完整时空轨迹 (Godunov 求解器生成)
    data_path = env.data_dir / "burgers_1d_N5000_Nx128.npy"
    
    # timestamp 用于本次运行的输出文件后缀
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 实验超参数: 从 YAML 配置文件加载 (training、model、loss 三类参数集中管理)
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)["experiment"]

    # 实验命名: 用于区分不同超参数组合的输出子目录
    exp_name = exp_cfg.get("name", "mvp_run")
    # 输出目录: {env.output_dir}/{exp_name}/ 下存放 checkpoint 和 TensorBoard 日志
    output_dir = env.output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- 训练超参数 (均从 YAML 读取，含默认值兜底) ----
    epochs = exp_cfg.get("epochs", 10)
    lr = float(exp_cfg.get("learning_rate", 2e-4))
    # ν (nu): 物理粘度系数，同时控制扩散调度 (论文 §3.1 方程 3.1: σ²(τ) = 2ντ)
    nu = float(exp_cfg.get("nu", 0.01))
    # τ_max: 最大扩散时间 (τ ∈ [0, τ_max] 均匀采样，对应物理演化时间 T_d)
    tau_max = float(exp_cfg.get("tau_max", 1.0))
    # λ_bv: BV 正则化权重 (论文 §3.3，控制解空间 BV 范数约束的强弱)
    #       Baseline 实验设 λ_bv=0，Ours 设 λ_bv>0
    lambda_bv = float(exp_cfg.get("lambda_bv", 0.1))
    # λ_dsm: DSM 损失的权重 (通常固定为 1.0，λ_bv 相对此为惩罚力度)
    lambda_dsm = float(exp_cfg.get("lambda_dsm", 1.0))

    # 数据文件缺失时提前退出，避免空跑 (提示用户先运行 generate_data.py)
    if not data_path.exists():
        print(f"Data not found at {data_path}. Run generate_data.py first.")
        return

    # ========== 2. 数据加载 ==========
    # train_dataset 加载后自动按 80/10/10 划分 train/val/test
    # 每次 __getitem__ 返回完整时空轨迹 shape [N_time, N_x]
    # 训练循环中取 batch[:, -1, :] 作为 target 解分布 ρ_T(u) (对应终端时间 T)
    print("Loading Dataset...")
    train_dataset = BurgersDataset(data_path, mode='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,      # 随机打乱，防止时间序偏差
                              num_workers=num_workers)

    # ========== 3. 模型与扩散调度初始化 ==========
    print(f"Initializing Model... (nu={nu}, lr={lr}, lambda_bv={lambda_bv})")

    # ---- 训练日志文件 ----
    # 每次运行独立日志, 文件名带时间戳不覆盖
    log_path = output_dir / f"train_log_{exp_name}_{run_timestamp}.txt"

    # StandardScore: EDM 参数化 D_θ(x, σ)
    #   - backbone 为 1D U-Net (src/models/unet_1d.py)
    #   - 前向过程: D_x = c_skip·x + c_out·F_θ(c_in·x, c_noise(σ))
    #   - 含义: 给定带噪输入 x_noisy，直接预测洁净态 x_0 (而非噪声 ε)
    #   - 论文对应: 03_method.tex §3.2 EDM 参数化
    model = StandardScore(in_channels=1).to(device)
    # Adam 优化器: lr 从 YAML 读取，默认 2e-4 (EDM 推荐值)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # ViscosityMatchedSchedule: 物理驱动的扩散时间 → 噪声强度映射
    #   τ ~ U[0, τ_max] → σ = sqrt(2ντ)
    #   论文对应: 03_method.tex §3.1 方程 (3.1)
    #   核心直觉: 扩散过程的噪声方差 σ² 与 Navier-Stokes 的粘性项 2νΔu
    #            共享同一个时间-空间类比: τ ↔ t, σ² ↔ 2ν
    schedule = ViscosityMatchedSchedule(nu=nu, tau_max=tau_max)

    # ---- 恢复训练逻辑 ----
    start_epoch = 0  # 默认从 epoch 0 开始
    if args.resume:
        import re
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        print(f"[resume] 加载模型权重: {ckpt_path}")
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        # 从文件名提取 epoch 编号: xxx_epN.pt → N
        m = re.search(r'_ep(\d+)', ckpt_path.name)
        if m:
            start_epoch = int(m.group(1))
            print(f"[resume] 从 epoch {start_epoch} 继续 (优化器从零开始, momentum 丢失)")
        else:
            print("[resume] 无法识别 epoch 编号, 从 epoch 0 继续")
        # 优化器从零开始 (未保存 optimizer state, 但对继续训练影响可控)
        # 日志以追加模式打开, 不重复写文件头
        log_fp = open(str(log_path), "a", encoding="utf-8")
        log_fp.write(f"# [resume] 从 epoch {start_epoch} 恢复训练 at {run_timestamp}\n")
        log_fp.flush()
    else:
        log_fp = open(str(log_path), "w", encoding="utf-8")
        # 写日志头: 运行元信息
        log_fp.write(f"# EntroDiff MVP Training Log\n")
        log_fp.write(f"# Time: {run_timestamp}\n")
        log_fp.write(f"# Config: {config_path}\n")
        log_fp.write(f"# Device: {device}  Batch: {batch_size}  Workers: {num_workers}\n")
        log_fp.write(f"# Params: epochs={epochs} lr={lr} nu={nu} tau_max={tau_max} lambda_dsm={lambda_dsm} lambda_bv={lambda_bv}\n")
        log_fp.write(f"# Data: {data_path}\n")
        log_fp.write(f"# Output: {output_dir}\n")
        log_fp.write(f"#\n")
        log_fp.write(f"# epoch  loss_total  loss_dsm  loss_bv\n")
        log_fp.flush()

    # ========== 4. 训练主循环 ==========
    print(f"Starting Training on {device} (epoch {start_epoch+1} → {epochs})...")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0   # 本 epoch 总损失累加值
        total_dsm = 0.0    # 本 epoch L_DSM 分量累加 (用于监控日志)
        total_bv = 0.0     # 本 epoch L_BV 分量累加 (用于监控日志)
        
        for batch in train_loader:
            # batch shape: [B, N_time, N_x] → 整条时空轨迹
            # 论文设定: 目标解分布 ρ_T(u) 锁定在终端时间 T = 0.5 sec
            #   取其最后一帧 batch[:, -1, :] 即为 ρ_T 的一个样本点 u_T(x)
            # unsqueeze(1): [B, N_x] → [B, 1, N_x], 增加 channel 维度适配 1D Conv
            x = batch[:, -1, :].unsqueeze(1).to(device)  # Shape [B, 1, Nx]

            optimizer.zero_grad()
            
            # 采样连续扩散时间 τ ~ U[0, τ_max]
            # ViscosityMatchedSchedule.sample_sigma 内部:
            #   1. 从均匀分布采样 τ
            #   2. 通过 σ = sqrt(2ντ) 映射为噪声强度
            # 返回 shape [B] 的 σ 向量，每个样本独立时间步 → 模拟连续时间扩散过程
            sigmas = schedule.sample_sigma(x.shape[0], device)
            
            # ---- L_DSM: Denoising Score Matching (论文 §3.2) ----
            # 数学形式: E_{x, ε, τ} [‖D_θ(x + σε, σ) - x‖₂²]
            # 实现于 src/diffusion/losses.py → get_dsm_loss()
            #   1. 生成高斯噪声 ε ~ N(0, I)
            #   2. 构造带噪样本 x_noisy = x + σ·ε
            #   3. 模型预测洁净态: D_x = model(x_noisy, σ)
            #   4. 计算 MSE: ‖D_x - x‖² → 等价于 score matching (对 EDM 参数化)
            loss_dsm = get_dsm_loss(model, x, sigmas)
            
            # ---- L_BV: Total-Variation 代理损失 (论文 §3.3) ----
            # 动机: 纯 L_DSM 训练出的解在 shock 处可能模糊/震荡
            #       BV 约束强制解的梯度在 L¹ 意义下保持有界
            # 实现: 对预测解 û = D_θ(x_noisy, σ) 计算:
            #       TV(û) = Σ|û_{i+1} - û_i| / N_x (离散全变差的一阶近似)
            # 论文对应: 03_method.tex §3.3 方程 L_BV
            # 注意: MVP 阶段此实为 TV 代理; 未来迭代中将替换为
            #       真正的 Kruzhkov 熵损失 (L_ent, 需对熵-熵通量对积分)
            loss_bv = get_bv_loss(model, x, sigmas)
            
            # 联合损失: L = λ_dsm·L_DSM + λ_bv·L_BV
            # Baseline 对照: λ_bv = 0  → 纯 EDM, 验证无 BV 约束时 shock 模糊
            # Ours (MVP):   λ_bv > 0 → 熵约束生效, 验证 shock 陡度保持
            loss = lambda_dsm * loss_dsm + lambda_bv * loss_bv
            
            # 反向传播 + 参数更新
            loss.backward()
            optimizer.step()
            
            # 累加各分量损失 (用于 epoch 结束时输出平均日志)
            total_loss += loss.item()
            total_dsm += loss_dsm.item()
            total_bv += loss_bv.item()

        # ---- Epoch 结束: 日志输出 ----
        # 分别报告总损失、DSM 分量、BV 分量，便于:
        #   1. 监控训练收敛: total_loss 应单调递减
        #   2. 验证 BV 约束效果: BV 分量应随训练逐渐降低 (解变光滑) 但不过分
        #   3. 调试超参数: 若 BV 分量远大于 DSM, 需调低 λ_bv
        avg_loss = total_loss / len(train_loader)
        avg_dsm = total_dsm / len(train_loader)
        avg_bv = total_bv / len(train_loader)
        # 同时输出到 stdout 和 log 文件
        line = (f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} "
                f"(DSM: {avg_dsm:.6f}, BV/TV: {avg_bv:.6f})")
        print(line)
        # 写入日志文件: epoch loss_total loss_dsm loss_bv (机器可读格式)
        log_fp.write(f"{epoch+1:4d}  {avg_loss:.8f}  {avg_dsm:.8f}  {avg_bv:.8f}\n")
        log_fp.flush()  # 每 epoch 刷新, 实时可读

        # ---- Checkpoint 保存 ----
        # 每 5 个 epoch 保存一次模型权重 (防止训练中断丢失结果)
        # 文件命名: entrodiff_mvp_[timestamp]_ep{epoch}.pt → 便于按 epoch 和实验批次区分
        if (epoch + 1) % 5 == 0:
            ckpt_path = output_dir / f"entrodiff_{exp_name}_{run_timestamp}_ep{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # ---- 训练结束: 关闭日志 ----
    log_fp.write(f"# Training completed.\n")
    log_fp.write(f"# Final epoch: {epochs}  Final ckpt: {output_dir}\n")
    log_fp.close()
    print(f"Training complete. Log saved to {log_path}")
    print(f"Checkpoints saved to {output_dir}")

# ========== 脚本入口 ==========
if __name__ == "__main__":
    train_mvp()
