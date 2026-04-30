# ============================================================================
# EntroDiff E1 Step Ablation — 少步数鲁棒性对比
# 实验目的:
#   少步数对比 — baseline 在少步数下 shock 处发散，
#   BVAwareScore 的建筑先验应保持稳健
#
# 功能:
#   1. 加载指定 checkpoint (StandardScore / BVAwareScore)
#   2. 在 test split 上以不同 Heun 步数 [10, 25, 50, 100] 进行反向采样
#   3. 逐样本计算 W₁ (1-Wasserstein 距离) 和 L¹ 相对误差
#   4. 输出步数 → W₁ → L¹ 的汇总表格 (stdout)
# ============================================================================
import os
import sys
import math
import torch
import numpy as np
import argparse
from pathlib import Path
from scipy.stats import wasserstein_distance  # 1D W₁: 排序后逐元素作差

# 将 PROJECT/black/ 加入 sys.path，确保 src/ 下所有模块可被 import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.data.burgers_dataset import BurgersDataset
from src.models.score_param import StandardScore, BVAwareScore  # 两种模型：纯 EDM vs 建筑先验
from src.diffusion.samplers import entrodiff_heun_sampler
from src.diffusion.schedules import ViscosityMatchedSchedule  # σ²(τ) = 2ντ (论文 §3.1)

# 模型注册表: 通过 --model_type 参数选择
MODEL_REGISTRY = {
    "standard": StandardScore,
    "bvaware":  BVAwareScore,
}

# 固定的 Heun 步数列表 (实验变量)
STEP_COUNTS = [10, 25, 50, 100]

# ============================================================================
# 工具函数
# ============================================================================

def load_model_from_ckpt(ckpt_path, model_type, model_dim, device):
    """
    从 checkpoint 加载模型。

    参数:
        ckpt_path:   .pt 文件路径
        model_type:  "standard" 或 "bvaware"
        model_dim:   BVAwareScore 的 UNet dim (None → 默认 128)
        device:      目标设备
    返回:
        加载了 state_dict 并设为 eval() 的模型
    """
    ModelClass = MODEL_REGISTRY[model_type]
    model_kwargs = {"in_channels": 1}
    if model_type == "bvaware":
        # BVAwareScore 额外参数: dim 控制 UNet 容量, return_denoiser 兼容现有 pipeline
        model_kwargs["dim"] = model_dim or 128
        model_kwargs["return_denoiser"] = True
    model = ModelClass(**model_kwargs).to(device)
    # strict=False: 兼容 checkpoint 中可能存有 optimizer / scheduler 等额外 key
    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict) and any(k.startswith("phi_sm_net") or k.startswith("net.") for k in state):
        # state dict 直接包含模型参数 (train_mvp.py 保存完整 state_dict)
        model.load_state_dict(state, strict=False)
    elif isinstance(state, dict) and "model_state_dict" in state:
        # 备选: checkpoint 以字典方式包装
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def run_ablation():
    """少步数消融实验主流程。"""
    # ========== 0. 命令行参数解析 ==========
    parser = argparse.ArgumentParser(
        description="EntroDiff Step Ablation: 验证少步采样下 BVAwareScore 的建筑先验稳健性"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="模型 checkpoint 路径 (.pt 文件)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=4,
        help="评估的 test 样本数量 (默认 4)"
    )
    parser.add_argument(
        "--model_type", type=str, default="standard",
        choices=["standard", "bvaware"],
        help="模型类型: standard=StandardScore, bvaware=BVAwareScore (默认 standard)"
    )
    parser.add_argument(
        "--model_dim", type=int, default=None,
        help="BVAwareScore 的 UNet dim (None → 自动 128)"
    )
    args = parser.parse_args()

    # ========== 1. 硬件 & 路径初始化 ==========
    device = torch.device(env.default_device)
    nu = 1.0          # 扩散过程单位粘度 (VE-SDE, 非数据 PDE 的 ν_phys=0)
    tau_max = 1.0     # 最大扩散时间 T_d
    sigma_min = 0.002 # 终止噪声水平 (EDM 推荐值)
    sigma_max = math.sqrt(2 * nu * tau_max)  # sqrt(2) ≈ 1.41, 覆盖数据 std≈1.34
    zeta_pde = 0.0    # 关闭 PDE guidance (纯神经生成, 控制变量)

    # ========== 2. 加载 test 数据 ==========
    data_path = env.data_dir / exp_cfg.get("data_file", "burgers_1d_N5000_Nx128.npy")
    if not data_path.exists():
        raise FileNotFoundError(
            f"数据文件不存在: {data_path}\n"
            f"请先运行 scripts/generate_data.py 生成数据"
        )
    print(f"[ablation] 加载 test 数据: {data_path}")
    test_dataset = BurgersDataset(data_path, mode='test')
    n_total = test_dataset.data.shape[0]
    n_samples = min(args.n_samples, n_total)
    print(f"[ablation] test split 共 {n_total} 条, 使用前 {n_samples} 条")

    # 取前 n_samples 条的终端时刻解 u(x, T) 作为 ground truth 分布
    gt_samples = test_dataset.data[:n_samples, -1, :]  # (n_samples, Nx)
    nx_dim = gt_samples.shape[1]

    # ========== 3. 加载模型 ==========
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")
    print(f"[ablation] 加载模型 ({args.model_type}): {ckpt_path}")
    model = load_model_from_ckpt(ckpt_path, args.model_type, args.model_dim, device)
    model_type_label = {"standard": "StandardScore", "bvaware": "BVAwareScore"}[args.model_type]

    # ========== 4. 逐步数评估 ==========
    gen_shape = (n_samples, 1, nx_dim)  # (B, C, Nx)
    records = []  # 存储 (steps, w1_mean, w1_std, l1_mean, l1_std)

    print(f"\n{'=' * 70}")
    print(f"  Step Ablation: {model_type_label}  @ Heun Steps ∈ {STEP_COUNTS}")
    print(f"  ν={nu}, τ_max={tau_max}, ζ_PDE={zeta_pde}")
    print(f"{'=' * 70}")

    for steps in STEP_COUNTS:
        print(f"\n  [Heun steps = {steps:3d}] 采样中...")
        gen = entrodiff_heun_sampler(
            model=model,
            shape=gen_shape,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            tau_max=tau_max,
            nu=nu,
            num_steps=steps,
            device=device,
            zeta_pde=zeta_pde
        ).squeeze().cpu().numpy()  # → (n_samples, Nx)

        w1_vals = []
        l1_vals = []
        for i in range(n_samples):
            gt = gt_samples[i]
            gn = gen[i]
            w1 = wasserstein_distance(gt, gn)
            l1_rel = np.linalg.norm(gn - gt, 1) / (np.linalg.norm(gt, 1) + 1e-10)
            w1_vals.append(w1)
            l1_vals.append(l1_rel)

        w1_mean = np.mean(w1_vals)
        w1_std = np.std(w1_vals)
        l1_mean = np.mean(l1_vals)
        l1_std = np.std(l1_vals)
        records.append((steps, w1_mean, w1_std, l1_mean, l1_std))

        # 逐条打印 (可选调试用)
        for i in range(n_samples):
            print(f"    Sample {i+1}: W₁={w1_vals[i]:.4f}  L¹={l1_vals[i]:.4f}")

    # ========== 5. 汇总表格 ==========
    print(f"\n{'=' * 70}")
    print(f"  汇总: {model_type_label} Step Ablation Results")
    print(f"  {'Steps':>6s}  {'W₁ (mean)':>12s}  {'W₁ (std)':>10s}  {'L¹ (mean)':>12s}  {'L¹ (std)':>10s}")
    print(f"  {'─' * 6}  {'─' * 12}  {'─' * 10}  {'─' * 12}  {'─' * 10}")
    for steps, w1_m, w1_s, l1_m, l1_s in records:
        print(f"  {steps:6d}  {w1_m:12.4f}  {w1_s:10.4f}  {l1_m:12.4f}  {l1_s:10.4f}")
    print(f"{'=' * 70}")

    # 打印趋势提示: 使用人语描述 W₁ 随步数的变化方向
    if len(records) >= 2:
        best_w1 = min(records, key=lambda r: r[1])
        print(f"\n  Step ablation 完成. 最佳 W₁={best_w1[1]:.4f} @ steps={best_w1[0]}")

    print(f"\n[ablation] 评估结束.")


if __name__ == "__main__":
    run_ablation()
