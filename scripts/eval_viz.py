# ============================================================================
# EntroDiff E1 Burgers 评估与可视化脚本
# 论文对应: 05_experiments.tex §E1 · Inviscid Burgers
# 功能:
#   1. 加载 Ours / Baseline checkpoint → 运行 Heun 反向采样生成解
#   2. 与 Godunov 真值 (test split) 对比，计算 W₁ / L¹ / shock-location 三项指标
#   3. 出图: (a) Ours vs GT 对比  (b) Baseline vs Ours 三栏对比
#   4. 输出指标摘要到 stdout，图表保存到 output_dir
# ============================================================================
import os
import sys
import math
import glob as glob_module
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # 无 GUI 后端，避免 DISPLAY 错误
import matplotlib.pyplot as plt
import yaml
import argparse
from pathlib import Path
from scipy.stats import wasserstein_distance  # 1D W₁ 距离 (两数组排序后逐元素作差)

# 将 PROJECT/black/ 加入 sys.path，确保 src/ 下所有模块可被 import
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.data.burgers_dataset import BurgersDataset
from src.models.score_param import StandardScore
from src.diffusion.samplers import entrodiff_heun_sampler
from src.diffusion.schedules import ViscosityMatchedSchedule
from src.diffusion.losses import get_dsm_loss  # 可选: 验证 checkpoint 质量

# ============================================================================
# 工具函数
# ============================================================================

def compute_shock_location(u, x_grid):
    """
    计算 shock 位置: 取 |grad(u)| 最大的 x 坐标
    对于 Burgers 方程，shock 处解的梯度最大 → argmax(|∂u/∂x|)
    
    参数:
        u: 1D numpy 数组, shape (Nx,)
        x_grid: 空间网格点坐标
    返回:
        shock 位置的 x 坐标 (float)
    """
    grad_u = np.abs(np.gradient(u, x_grid))
    return x_grid[np.argmax(grad_u)]

def find_latest_checkpoint(search_dir, pattern_prefix):
    """
    在 search_dir 下搜索 checkpoint .pt 文件，返回最新者。
    
    搜索策略 (按优先级):
      1. 精确前缀匹配: {pattern_prefix}*.pt
      2. 回退: 目录下任意 *.pt 文件 (兼容旧版命名 entrodiff_mvp_ep10.pt)

    返回文件 Path，若无匹配则返回 None
    """
    # 策略1: 前缀匹配
    pattern = str(search_dir / f"{pattern_prefix}*.pt")
    matches = sorted(glob_module.glob(pattern))
    # 策略2: 回退 — 目录下任意 .pt
    if not matches:
        pattern_fallback = str(search_dir / "*.pt")
        matches = sorted(glob_module.glob(pattern_fallback))
    if not matches:
        return None
    # 优先取 ep 编号最大的 (文件名中 *_epN.pt)
    matches_by_ep = sorted(matches, key=lambda p: _extract_epoch(p), reverse=True)
    return Path(matches_by_ep[0])

def _extract_epoch(filepath):
    """从文件名中提取 epoch 编号 (如 'xxx_ep10.pt' → 10)，失败返回 -1"""
    import re
    m = re.search(r'_ep(\d+)', str(filepath))
    return int(m.group(1)) if m else -1

# ============================================================================
# 主评估函数
# ============================================================================

def eval_viz():
    """
    E1 Burgers 评估主流程。

    步骤:
      1. 解析命令行参数 (config / ckpt_ours / ckpt_baseline)
      2. 加载 test split 的 ground truth (前 4 条样本用于可视化)
      3. 加载 Ours 模型 → Heun 采样 → 计算指标 + 出图
      4. (optional) 加载 Baseline 模型 → Heun 采样 → 三栏对比图
    """
    # ========== 0. 命令行参数解析 ==========
    parser = argparse.ArgumentParser(
        description="EntroDiff E1 Burgers 评估与可视化"
    )
    parser.add_argument(
        "--config", type=str, default="configs/experiment/mvp_burgers.yaml",
        help="实验超参数配置文件 (相对于 PROJECT/black/)"
    )
    parser.add_argument(
        "--ckpt_ours", type=str, default=None,
        help="Ours checkpoint 路径 (默认自动搜索 latest *_ep10.pt)"
    )
    parser.add_argument(
        "--ckpt_baseline", type=str, default=None,
        help="Baseline checkpoint 路径 (默认自动搜索 latest)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=4,
        help="可视化用的 test 样本数量 (默认 4)"
    )
    parser.add_argument(
        "--heun_steps", type=int, default=50,
        help="Heun 采样步数 (覆盖 config 中的值)"
    )
    args = parser.parse_args()

    # ========== 1. 加载配置 ==========
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)["experiment"]

    device = torch.device(env.default_device)
    nu = float(exp_cfg.get("nu", 0.01))
    tau_max = float(exp_cfg.get("tau_max", 1.0))
    heun_steps = args.heun_steps if args.heun_steps != 50 else int(exp_cfg.get("heun_steps", 50))
    zeta_pde = float(exp_cfg.get("zeta_pde", 0.0))
    n_samples = args.n_samples

    # ========== 2. 加载 test data (Godunov 真值) ==========
    print("[eval] 加载 test split 数据...")
    data_path = env.data_dir / "burgers_1d_N5000_Nx128.npy"
    if not data_path.exists():
        raise FileNotFoundError(
            f"数据文件不存在: {data_path}\n"
            f"请先运行 python scripts/generate_data.py 生成数据"
        )
    test_dataset = BurgersDataset(data_path, mode='test')

    # 取前 n_samples 条的终端时刻解 (即最后一个时间步) 作为 ground truth ρ_T
    # gt_samples shape: (n_samples, Nx) -- 每行是一条 Godunov 真值解 u(x, T)
    gt_samples = test_dataset.data[:n_samples, -1, :]
    nx_dim = gt_samples.shape[1]
    x_grid = np.linspace(0, 2 * np.pi, nx_dim, endpoint=False)  # 空间坐标 (周期域)

    # ========== 3. 加载 Ours 模型 ==========
    # 自动搜索策略: --ckpt_ours 为空 → glob 匹配 exp_name 前缀的最新 ep10.pt
    exp_name = exp_cfg.get("name", "entrodiff_mvp_run1")
    output_dir = env.output_dir / exp_name
    if args.ckpt_ours is None:
        # 自动搜索: 先按 *_ep10.pt 精确匹配, 否则按 prefix_* 取最新
        ckpt_ours_path = find_latest_checkpoint(output_dir, f"entrodiff_{exp_name}")
        if ckpt_ours_path is None:
            raise FileNotFoundError(
                f"在 {output_dir} 下未找到 checkpoint ({exp_name}前缀)\n"
                f"请先运行 python scripts/train_mvp.py 或通过 --ckpt_ours 指定路径"
            )
    else:
        ckpt_ours_path = Path(args.ckpt_ours)
    print(f"[eval] 加载 Ours 模型: {ckpt_ours_path}")

    model_ours = StandardScore(in_channels=1).to(device)
    model_ours.load_state_dict(torch.load(str(ckpt_ours_path), map_location=device))
    model_ours.eval()

    # ========== 4. Ours Heun 采样 ==========
    print(f"[eval] Ours Heun 采样 ({heun_steps} steps, nu={nu}, zeta_pde={zeta_pde})...")
    gen_shape = (n_samples, 1, nx_dim)
    gen_ours = entrodiff_heun_sampler(
        model=model_ours,
        shape=gen_shape,
        sigma_min=0.002,
        sigma_max=math.sqrt(2 * nu * tau_max),
        tau_max=tau_max,
        nu=nu,
        num_steps=heun_steps,
        device=device,
        zeta_pde=zeta_pde
    ).squeeze().cpu().numpy()  # → (n_samples, Nx)

    # ========== 5. 指标计算 ==========
    # 三个核心指标 (论文 §5 Table 2):
    #   W₁: 1-Wasserstein 距离 (衡量分布差异, 对 shock 位置敏感)
    #   L¹: 相对 L¹ 误差 (逐点绝对值之和 / 真值 L¹ 范数)
    #   Shock-loc: shock 位置误差 (|x_shock_pred - x_shock_gt|)
    w1_list, l1_list, shock_list = [], [], []
    print("\n" + "=" * 60)
    print("  Ours (EntroDiff) vs Ground Truth  指标摘要")
    print("=" * 60)
    for i in range(n_samples):
        w1 = wasserstein_distance(gt_samples[i], gen_ours[i])
        l1_rel = np.linalg.norm(gen_ours[i] - gt_samples[i], 1) / (np.linalg.norm(gt_samples[i], 1) + 1e-10)
        s_loc_gt = compute_shock_location(gt_samples[i], x_grid)
        s_loc_gn = compute_shock_location(gen_ours[i], x_grid)
        shock_err = abs(s_loc_gn - s_loc_gt)
        w1_list.append(w1)
        l1_list.append(l1_rel)
        shock_list.append(shock_err)
        print(f"  Sample {i+1}: W₁={w1:.4f}  L¹_rel={l1_rel:.4f}  Shock-err={shock_err:.4f}")
    print("-" * 60)
    print(f"  平均: W₁={np.mean(w1_list):.4f}  L¹_rel={np.mean(l1_list):.4f}  Shock-err={np.mean(shock_list):.4f}")
    print("=" * 60 + "\n")

    # ========== 6. 出图 1: Ours vs GT 逐样本对比 ==========
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("EntroDiff Generated Shocks vs Ground Truth (Inviscid Burgers)", fontsize=16)
    for i, ax in enumerate(axes.flatten()):
        ax.plot(x_grid, gen_ours[i], 'r-', linewidth=1.5, label="EntroDiff (Ours)")
        ax.plot(x_grid, gt_samples[i], 'k--', linewidth=1.5, alpha=0.6, label="Godunov GT")
        # 标注 shock 位置: 在各自峰值梯度处画竖线
        s_ours = compute_shock_location(gen_ours[i], x_grid)
        s_gt = compute_shock_location(gt_samples[i], x_grid)
        ax.axvline(x=s_ours, color='r', linestyle=':', alpha=0.5)
        ax.axvline(x=s_gt, color='k', linestyle=':', alpha=0.5)
        ax.set_title(f"Sample {i+1}  W₁={w1_list[i]:.2f}  Shock-err={shock_list[i]:.3f}")
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, T)")
    plt.tight_layout()
    out1 = env.output_dir / exp_name / "e1_shock_comparison.png"
    out1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out1), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[eval] 已保存 Ours vs GT 对比图: {out1}")

    # ========== 7. 出图 2: Baseline vs Ours 三栏对比 (若 baseline ckpt 存在) ==========
    # 自动搜索 baseline ckpt
    baseline_dir = env.output_dir / "mvp_baseline"
    if args.ckpt_baseline is None:
        ckpt_baseline_path = find_latest_checkpoint(baseline_dir, "entrodiff_mvp_baseline")
    else:
        ckpt_baseline_path = Path(args.ckpt_baseline)

    if ckpt_baseline_path and ckpt_baseline_path.exists():
        print(f"[eval] 加载 Baseline 模型: {ckpt_baseline_path}")
        model_base = StandardScore(in_channels=1).to(device)
        model_base.load_state_dict(torch.load(str(ckpt_baseline_path), map_location=device))
        model_base.eval()

        # Baseline 采样: zeta_pde=0 (无 Godunov guidance)
        print(f"[eval] Baseline Heun 采样 ({heun_steps} steps, pure EDM)...")
        gen_base = entrodiff_heun_sampler(
            model=model_base,
            shape=gen_shape,
            sigma_min=0.002,
            sigma_max=math.sqrt(2 * nu * tau_max),
            tau_max=tau_max,
            nu=nu,
            num_steps=heun_steps,
            device=device,
            zeta_pde=0.0  # Baseline 不使用 PDE guidance
        ).squeeze().cpu().numpy()

        # 三栏图: GT | EDM Baseline | EntroDiff Ours
        fig2, axes2 = plt.subplots(n_samples, 3, figsize=(15, 3.5 * n_samples))
        fig2.suptitle(
            "Inviscid Burgers: Ground Truth vs EDM Baseline vs EntroDiff (Ours)",
            fontsize=16
        )
        col_titles = ["Ground Truth (Godunov)", "EDM Baseline (pure L_DSM)", "EntroDiff (Ours)"]

        # 处理 n_samples=1 时 axes 不是 2D 的情况
        if n_samples == 1:
            axes2 = np.expand_dims(axes2, axis=0)

        for i in range(n_samples):
            # Column 0: GT
            axes2[i, 0].plot(x_grid, gt_samples[i], 'k-', linewidth=1.5)
            axes2[i, 0].set_title(f"Sample {i+1}: Ground Truth")
            axes2[i, 0].set_xlabel("x"); axes2[i, 0].set_ylabel("u")

            # Column 1: EDM Baseline
            axes2[i, 1].plot(x_grid, gen_base[i], 'b-', linewidth=1.5)
            w1_base = wasserstein_distance(gt_samples[i], gen_base[i])
            l1_base = np.linalg.norm(gen_base[i] - gt_samples[i], 1) / (np.linalg.norm(gt_samples[i], 1) + 1e-10)
            axes2[i, 1].set_title(f"EDM Baseline  W₁={w1_base:.3f}  L¹={l1_base:.3f}")
            axes2[i, 1].set_xlabel("x")

            # Column 2: Ours
            axes2[i, 2].plot(x_grid, gen_ours[i], 'r-', linewidth=1.5)
            axes2[i, 2].set_title(f"EntroDiff (Ours)  W₁={w1_list[i]:.3f}  L¹={l1_list[i]:.3f}")
            axes2[i, 2].set_xlabel("x")

            # 统一 ylim 以便直观对比
            ymin = min(gt_samples[i].min(), gen_base[i].min(), gen_ours[i].min())
            ymax = max(gt_samples[i].max(), gen_base[i].max(), gen_ours[i].max())
            margin = 0.1 * (ymax - ymin)
            for j in range(3):
                axes2[i, j].set_ylim(ymin - margin, ymax + margin)
                axes2[i, j].grid(True, alpha=0.3)

        # 设置列标题
        for j, title in enumerate(col_titles):
            axes2[0, j].set_title(f"{title}\nSample 1", fontsize=10)

        plt.tight_layout()
        out2 = env.output_dir / exp_name / "e1_baseline_vs_ours.png"
        out2.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out2), dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"[eval] 已保存 Baseline vs Ours 对比图: {out2}")

        # 输出对比摘要
        w1_base_list = [wasserstein_distance(gt_samples[i], gen_base[i]) for i in range(n_samples)]
        l1_base_list = [np.linalg.norm(gen_base[i] - gt_samples[i], 1) / (np.linalg.norm(gt_samples[i], 1) + 1e-10) for i in range(n_samples)]
        print("\n" + "=" * 60)
        print("  Baseline vs Ours 对比摘要")
        print("=" * 60)
        print(f"  指标           EDM Baseline         EntroDiff (Ours)")
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  W₁  avg         {np.mean(w1_base_list):.4f}              {np.mean(w1_list):.4f}")
        print(f"  L¹  avg         {np.mean(l1_base_list):.4f}              {np.mean(l1_list):.4f}")
        print("=" * 60 + "\n")

    else:
        print("[eval] 未找到 Baseline checkpoint. 跳過三栏对比图.")
        print(f"       请运行 python scripts/train_baseline.py 生成 baseline ckpt")
        print(f"       或通过 --ckpt_baseline 手动指定路径")

    print("[eval] 全部评估完成.")


if __name__ == "__main__":
    eval_viz()
