# ============================================================================
# EntroDiff E3 Euler Sod 评估与可视化脚本
# 论文对应: 05_experiments.tex §5.4 · Euler Sod 激波管 (三组分守恒律)
# 功能:
#   1. 加载 BVAwareScore / StandardScore checkpoint → Heun 反向采样生成解
#   2. 与 Godunov 真值 (test split) 对比，按通道计算 W₁ / L¹ 相对误差
#   3. 出图: 3 行 (ρ, ρu, E) × 3 列 (GT | Generated | Error) = 3×3 网格
#   4. 输出指标摘要到 stdout，图表保存到 output_dir / e3_eval /
# ============================================================================
import os
import sys
import math
import glob as glob_module
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import argparse
from pathlib import Path
from scipy.stats import wasserstein_distance

# 将 PROJECT/black/ 加入 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.data.euler_dataset import EulerDataset
from src.models.score_param import StandardScore, BVAwareScore
from src.diffusion.samplers import entrodiff_heun_sampler

# ============================================================================
# StandardScore 多通道包装器 (Euler 3 组分适配)
# 原因: StandardScore 原生只输出 1 通道 (hardcoded),
#       而 Euler 系统需要 3 通道同步输出 → 扩展 forward 逻辑
# ============================================================================
class StandardScoreMultiChannel(nn.Module):
    """
    扩展 StandardScore 以支持多通道输出 (out_channels > 1)。
    
    与 StandardScore 的唯一区别:
      - UNet1D 输出 out_channels 通道 (而非硬编码 1)
      - skip connection 作用于前 out_channels 个带噪通道
      - 其余行为 (EDM precond: c_skip/c_out/c_in/c_noise) 完全一致
    """
    def __init__(self, in_channels=6, out_channels=3, sigma_data=0.5):
        super().__init__()
        from src.models.unet_1d import UNet1D
        self.net = UNet1D(in_channels=in_channels, out_channels=out_channels)
        self.sigma_data = sigma_data
        self.out_channels = out_channels

    def forward(self, x, sigma, pde_id=None):
        """
        EDM preconditions 多通道版.
        c_skip * x[:, :out_C] + c_out * F_theta(c_in * x, c_noise)
        
        pde_id: 兼容性参数 (透传给 UNet, 当前不消费)
        """
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2)**0.5
        c_in = 1 / (self.sigma_data**2 + sigma**2)**0.5
        c_noise = sigma.log() / 4.0

        F_x = self.net(c_in[:, None, None] * x, c_noise)
        D_x = c_skip[:, None, None] * x[:, :self.out_channels, :] \
              + c_out[:, None, None] * F_x
        return D_x


# 模型注册表: --model_type 选择
MODEL_REGISTRY = {
    "standard": StandardScoreMultiChannel,
    "bvaware":  BVAwareScore,
}

# Euler 三组分通道名称 (论文 §5.4 符号: ρ, ρu, E)
CHANNEL_NAMES = [
    r"Density $\rho$",
    r"Momentum $\rho u$",
    r"Energy $E$",
]
CHANNEL_LABELS_SHORT = ["rho", "rhou", "E"]


# ============================================================================
# 工具函数
# ============================================================================

def find_latest_checkpoint(search_dir, pattern_prefix):
    """
    在 search_dir 下搜索 checkpoint .pt 文件，返回最新者。
    
    搜索策略:
      1. 精确前缀: {pattern_prefix}*.pt
      2. 回退: 目录下任意 *.pt
    """
    search_dir = Path(search_dir)
    if not search_dir.exists():
        return None
    pattern = str(search_dir / f"{pattern_prefix}*.pt")
    matches = sorted(glob_module.glob(pattern))
    if not matches:
        matches = sorted(glob_module.glob(str(search_dir / "*.pt")))
    if not matches:
        return None
    matches_by_ep = sorted(matches, key=_extract_epoch, reverse=True)
    return Path(matches_by_ep[0])


def _extract_epoch(filepath):
    """从文件名提取 epoch 编号 (e.g. '*_ep200.pt' → 200), 失败返回 -1"""
    import re
    m = re.search(r'_ep(\d+)', str(filepath))
    return int(m.group(1)) if m else -1


# ============================================================================
# 主评估函数
# ============================================================================

def eval_euler():
    """
    E3 Euler Sod 评估主流程.
    
    步骤:
      1. 解析命令行参数
      2. 加载 test split 的 ground truth 与 IC
      3. 加载模型 (StandardScoreMultiChannel 或 BVAwareScore)
      4. Heun 反向采样 (IC-conditioned, 3 通道同步)
      5. 按通道计算 W₁ / L¹ 指标
      6. 出图: 3×3 网格 (ρ/ρu/E × GT/Gen/Error)
      7. 保存到 output_dir / e3_eval / e3_euler_comparison.png
    """
    # ========== 0. 命令行参数 ==========
    parser = argparse.ArgumentParser(
        description="EntroDiff E3 Euler Sod 评估与可视化"
    )
    parser.add_argument(
        "--config", type=str, default="configs/experiment/e3_euler.yaml",
        help="实验超参数配置文件 (相对于 PROJECT/black/)"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="模型 checkpoint 路径 (必填)"
    )
    parser.add_argument(
        "--model_type", type=str, default="bvaware",
        choices=["standard", "bvaware"],
        help="模型类型: standard=StandardScoreMultiChannel, bvaware=BVAwareScore"
    )
    parser.add_argument(
        "--model_dim", type=int, default=128,
        help="UNet 隐藏通道数 (BVAwareScore 用; StandardScore 不受影响)"
    )
    parser.add_argument(
        "--heun_steps", type=int, default=50,
        help="Heun 采样步数 (默认 50)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=4,
        help="可视化用的 test 样本数量 (默认 4)"
    )
    parser.add_argument(
        "--zeta_pde", type=float, default=0.0,
        help="PDE guidance 强度 (默认 0.0)"
    )
    args = parser.parse_args()

    # ========== 1. 加载配置 ==========
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)["experiment"]

    device = torch.device(env.default_device)
    nu = float(exp_cfg.get("nu", 1.0))                          # VE-SDE 粘度
    tau_max = float(exp_cfg.get("tau_max", 1.0))                # 最大扩散时间
    heun_steps = int(exp_cfg.get("heun_steps", args.heun_steps))
    zeta_pde = args.zeta_pde
    n_samples = args.n_samples
    out_channels = int(exp_cfg.get("out_channels", 3))          # Euler 三组分
    in_channels = int(exp_cfg.get("in_channels", 6))            # 3 noisy + 3 IC
    n_components = int(exp_cfg.get("n_components", 3))

    exp_name = exp_cfg.get("name", "e3_euler_run")
    output_dir = env.output_dir / exp_name

    # sigma_max = sqrt(2 * nu * tau_max) — Eq. 3.1 VE-SDE
    sigma_max = math.sqrt(2.0 * nu * tau_max)
    sigma_min = 0.002

    print(f"[eval_euler] config={args.config}  device={device}  nu={nu}  tau_max={tau_max}")
    print(f"[eval_euler] model_type={args.model_type}  in_channels={in_channels}"
          f"  out_channels={out_channels}  heun_steps={heun_steps}")

    # ========== 2. 加载 test data ==========
    data_filename = exp_cfg.get("data_file", "euler_sod_1d_N5000_Nx128.npy")
    data_path = env.data_dir / data_filename
    if not data_path.exists():
        raise FileNotFoundError(
            f"数据文件不存在: {data_path}\n"
            f"请先运行 python scripts/generate_euler_data.py 生成数据"
        )
    print(f"[eval_euler] 加载 test split 数据: {data_path}")

    test_dataset = EulerDataset(
        data_path, mode="test",
        conditioning_type="ic",
        n_components=n_components,
    )

    # 原始数据 shape: (N_test, n_comp, N_time, Nx) = (N_test, 3, N_time, Nx)
    # gt: 末帧  [n, 3, Nx]   — 每个通道的终端守恒变量
    # ic: 初帧  [n, 3, Nx]   — 初始条件 (Sod 激波管间断初值)
    gt_samples = test_dataset.data[:n_samples, :, -1, :].copy()   # (n, 3, Nx)
    ic_samples = test_dataset.data[:n_samples, :, 0, :].copy()    # (n, 3, Nx)
    nx_dim = gt_samples.shape[2]
    x_grid = np.linspace(0.0, 1.0, nx_dim, endpoint=False)        # Sod 管域 [0, 1]

    print(f"[eval_euler] test split: {test_dataset.data.shape[0]} samples"
          f"  Nx={nx_dim}  n_comp={n_components}")

    # ========== 3. 加载模型 ==========
    ModelClass = MODEL_REGISTRY[args.model_type]
    if args.model_type == "bvaware":
        model = ModelClass(
            in_channels=in_channels,
            dim=args.model_dim,
            return_denoiser=True,
            out_channels=out_channels,
        ).to(device)
    else:
        model = ModelClass(
            in_channels=in_channels,
            out_channels=out_channels,
        ).to(device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")
    print(f"[eval_euler] 加载 {args.model_type} 模型: {ckpt_path}")

    state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[eval_euler] 模型参数量: {param_count:,}")

    # ========== 4. Heun 反向采样 ==========
    # shape: (n_samples, 3, Nx) — 三个守恒变量 (ρ, ρu, E) 同步采样
    # 初始噪声: u_tau ~ N(0, sqrt(2*nu*tau_max)) × I_{3×Nx}
    #
    # IC 条件: cond [n, 3, Nx] 在 sampler 内部与带噪 u_tau [n, 3, Nx]
    #          沿 dim=1 拼接为 [n, 6, Nx] → 模型 forward 输入
    print(f"[eval_euler] Heun 采样 ({heun_steps} steps, nu={nu}, zeta_pde={zeta_pde})...")

    gen_shape = (n_samples, out_channels, nx_dim)
    cond_tensor = torch.tensor(ic_samples, dtype=torch.float32, device=device)

    gen_tensor = entrodiff_heun_sampler(
        model=model,
        shape=gen_shape,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        tau_max=tau_max,
        nu=nu,
        num_steps=heun_steps,
        device=device,
        zeta_pde=zeta_pde,
        conditioning=cond_tensor,   # [n, 3, Nx] — 初值条件
    )
    gen_samples = gen_tensor.cpu().numpy()   # → (n, 3, Nx)

    # ========== 5. 按通道计算指标 ==========
    # W₁: 1-Wasserstein 距离 (scipy.stats.wasserstein_distance)
    # L¹: 相对 L¹ 误差  ‖gen - gt‖_1 / ‖gt‖_1
    print("\n" + "=" * 70)
    print(f"  E3 Euler Sod  评估指标摘要  (n={n_samples}, heun_steps={heun_steps})")
    print("=" * 70)

    for c in range(out_channels):
        w1_list, l1_list = [], []
        for i in range(n_samples):
            gt_1d = gt_samples[i, c, :]
            gn_1d = gen_samples[i, c, :]
            w1 = wasserstein_distance(gt_1d, gn_1d)
            l1_rel = np.linalg.norm(gn_1d - gt_1d, 1) \
                     / (np.linalg.norm(gt_1d, 1) + 1e-10)
            w1_list.append(w1)
            l1_list.append(l1_rel)
        print(f"  [{CHANNEL_LABELS_SHORT[c]}] "
              f"W₁ avg={np.mean(w1_list):.4f}  "
              f"L¹ avg={np.mean(l1_list):.4f}  "
              f"W₁ std={np.std(w1_list):.4f}  L¹ std={np.std(l1_list):.4f}")

    # 整体平均
    all_w1, all_l1 = [], []
    for c in range(out_channels):
        for i in range(n_samples):
            all_w1.append(wasserstein_distance(
                gt_samples[i, c, :], gen_samples[i, c, :]
            ))
            all_l1.append(
                np.linalg.norm(gen_samples[i, c, :] - gt_samples[i, c, :], 1)
                / (np.linalg.norm(gt_samples[i, c, :], 1) + 1e-10)
            )
    print("  " + "-" * 50)
    print(f"  [整体] W₁ avg={np.mean(all_w1):.4f}  "
          f"L¹ avg={np.mean(all_l1):.4f}")
    print("=" * 70 + "\n")

    # ========== 6. 出图: 3 行 (通道) × 3 列 (GT | Generated | Error) ==========
    # 网格布局:
    #   Row 0 — Density ρ      : Col 0 GT  |  Col 1 Gen  |  Col 2 Error
    #   Row 1 — Momentum ρu    : Col 0 GT  |  Col 1 Gen  |  Col 2 Error
    #   Row 2 — Energy E       : Col 0 GT  |  Col 1 Gen  |  Col 2 Error
    #
    # 取第 0 个样本绘制 (首样本最有代表性)
    sample_idx = 0  # 可改为循环绘制多样本, 当前为了清晰只画首条

    fig, axes = plt.subplots(out_channels, 3, figsize=(16, 4 * out_channels))
    fig.suptitle(
        f"E3 Euler Sod: Ground Truth vs EntroDiff Generated "
        f"(sample {sample_idx}, heun_steps={heun_steps})",
        fontsize=14, fontweight="bold"
    )

    # 网格最大值 (统一 y 轴尺度以便跨通道对比)
    global_ymin = min(
        gt_samples[sample_idx].min(),
        gen_samples[sample_idx].min()
    )
    global_ymax = max(
        gt_samples[sample_idx].max(),
        gen_samples[sample_idx].max()
    )
    margin = 0.1 * max(abs(global_ymax - global_ymin), 1e-3)

    for c in range(out_channels):
        gt_c = gt_samples[sample_idx, c, :]
        gn_c = gen_samples[sample_idx, c, :]
        err_c = gn_c - gt_c

        # W₁ 与 L¹ for this channel
        w1_c = wasserstein_distance(gt_c, gn_c)
        l1_c = np.linalg.norm(err_c, 1) / (np.linalg.norm(gt_c, 1) + 1e-10)

        # Col 0: Ground Truth
        ax_gt = axes[c, 0]
        ax_gt.plot(x_grid, gt_c, 'k-', linewidth=1.5, label="Godunov GT")
        ax_gt.set_title(f"{CHANNEL_NAMES[c]}  (Ground Truth)", fontsize=11)
        ax_gt.set_xlabel("x")
        ax_gt.set_ylabel(CHANNEL_NAMES[c])
        ax_gt.set_ylim(global_ymin - margin, global_ymax + margin)
        ax_gt.grid(True, alpha=0.3)
        ax_gt.legend(loc='best', fontsize=8)

        # Col 1: Generated
        ax_gen = axes[c, 1]
        ax_gen.plot(x_grid, gn_c, 'r-', linewidth=1.5, label="EntroDiff")
        ax_gen.set_title(
            f"{CHANNEL_NAMES[c]}  (Generated)\nW₁={w1_c:.4f}  L¹={l1_c:.4f}",
            fontsize=11
        )
        ax_gen.set_xlabel("x")
        ax_gen.set_ylim(global_ymin - margin, global_ymax + margin)
        ax_gen.grid(True, alpha=0.3)
        ax_gen.legend(loc='best', fontsize=8)

        # Col 2: Error
        ax_err = axes[c, 2]
        ax_err.plot(x_grid, err_c, 'b-', linewidth=1.0)
        ax_err.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax_err.fill_between(
            x_grid, err_c, 0,
            alpha=0.2, color='b',
            where=(err_c > 0),
            label="gen > GT"
        )
        ax_err.fill_between(
            x_grid, err_c, 0,
            alpha=0.2, color='orange',
            where=(err_c < 0),
            label="gen < GT"
        )
        ax_err.set_title(
            f"{CHANNEL_NAMES[c]}  (Error = Gen - GT)",
            fontsize=11
        )
        ax_err.set_xlabel("x")
        ax_err.set_ylabel("Error")
        ax_err.grid(True, alpha=0.3)
        ax_err.legend(loc='best', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])   # 留空间给 suptitle

    # 保存路径: output_dir / e3_eval / e3_euler_comparison.png
    plot_dir = env.output_dir / exp_name / "e3_eval"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "e3_euler_comparison.png"
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[eval_euler] 已保存 3×3 对比图: {plot_path}")

    # 额外: 逐样本多通道对比图 (n_samples × 3 变量)
    # 此图为补充诊断: 每列一个样品, 叠盖 GT 与 Gen 三条线
    fig2, axes2 = plt.subplots(n_samples, 1, figsize=(14, 3.5 * n_samples))
    if n_samples == 1:
        axes2 = np.array([axes2])   # 统一为 1D 数组便于索引
    fig2.suptitle(
        f"E3 Euler Sod: All Channels Overlay (GT vs EntroDiff, "
        f"heun_steps={heun_steps})",
        fontsize=14, fontweight="bold"
    )
    colors_gt = ['#1f77b4', '#ff7f0e', '#2ca02c']     # 蓝/橙/绿 GT
    colors_gen = ['#d62728', '#9467bd', '#8c564b']     # 红/紫/棕 Gen

    for i in range(n_samples):
        ax = axes2[i]
        for c in range(out_channels):
            ax.plot(x_grid, gt_samples[i, c, :],
                    color=colors_gt[c], linewidth=1.2, linestyle='--',
                    label=f"GT {CHANNEL_LABELS_SHORT[c]}")
            ax.plot(x_grid, gen_samples[i, c, :],
                    color=colors_gen[c], linewidth=1.2,
                    label=f"Gen {CHANNEL_LABELS_SHORT[c]}")
        w1_i = np.mean([
            wasserstein_distance(gt_samples[i, c, :], gen_samples[i, c, :])
            for c in range(out_channels)
        ])
        ax.set_title(f"Sample {i+1}  (avg W₁={w1_i:.4f})", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("Conservative variables")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=7, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path2 = plot_dir / "e3_euler_overlay.png"
    plt.savefig(str(plot_path2), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"[eval_euler] 已保存逐样本叠加图: {plot_path2}")

    print("[eval_euler] 全部评估完成.")


if __name__ == "__main__":
    eval_euler()
