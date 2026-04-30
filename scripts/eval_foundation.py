# ============================================================================
# Foundation Model 跨 PDE 评估脚本 (W5-E)
#
# 工程方案: Docs/black/W5_foundation_engineering_plan.md §2.5
#
# 功能:
#   1. 从 ckpt 加载模型 + 自动重建 (利用 ckpt 内嵌 cfg)
#   2. 对配置中每个 PDE 加载 test split → Heun 采样 → 计算指标
#   3. 输出表格 (rows=PDE, cols=W1/L1/shock_err) + 三栏对比图 (gt / Ours)
#
# 用法:
#   python scripts/eval_foundation.py --ckpt <path/to/foundation_*.pt>
#   python scripts/eval_foundation.py --ckpt <path> --n_samples 8 --num_steps 50
# ============================================================================

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.data.mixed_pde_dataset import MixedPDEDataset
from src.models.foundation_score import FoundationScore
from src.models.score_param import BVAwareScore
from src.diffusion.samplers import entrodiff_heun_sampler


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------

def compute_shock_location(u: np.ndarray, x_grid: np.ndarray) -> float:
    """复用 eval_viz.py 的 shock-loc 度量: argmax(|∇u|)."""
    grad_u = np.abs(np.gradient(u, x_grid))
    return float(x_grid[np.argmax(grad_u)])


def compute_metrics(gt: np.ndarray, gen: np.ndarray, x_grid: np.ndarray) -> dict:
    """
    给定 ground truth 和生成解 (各 (Nx,)), 计算 W1 / L1_rel / shock_err.

    Args:
        gt:     (Nx,) ground truth
        gen:    (Nx,) 生成解
        x_grid: (Nx,) 空间坐标 (用于 shock-loc)

    Returns:
        {'W1': float, 'L1_rel': float, 'shock_err': float}
    """
    # W1: 1D Wasserstein-1, 直接用 scipy
    w1 = float(wasserstein_distance(gt, gen))
    # L1 相对误差: |y - x|_1 / |x|_1
    eps = 1e-8
    l1_rel = float(np.sum(np.abs(gt - gen)) / (np.sum(np.abs(gt)) + eps))
    # Shock 位置误差
    sh_gt = compute_shock_location(gt, x_grid)
    sh_gen = compute_shock_location(gen, x_grid)
    shock_err = float(abs(sh_gt - sh_gen))
    return {"W1": w1, "L1_rel": l1_rel, "shock_err": shock_err}


def build_model_from_ckpt_cfg(cfg: dict, n_pde_types: int, device: torch.device) -> torch.nn.Module:
    """
    根据 ckpt 内嵌的 cfg 重建模型 (与 train_foundation.build_model 同款).

    设计动机: 让 eval 不需要外部 cfg 文件; 只要 ckpt 完整就能 eval.
    """
    model_cfg = cfg["model"]
    dit_kwargs = dict(model_cfg["dit"])
    dit_kwargs["n_pde_types"] = n_pde_types

    model_type = model_cfg["type"]
    in_channels = int(model_cfg.get("in_channels", 2))
    Nx = int(dit_kwargs.get("Nx", 128))

    if model_type == "dit_plain":
        nx_for_dit = dit_kwargs.pop("Nx", Nx)
        if nx_for_dit != Nx:
            raise ValueError(f"dit.Nx={nx_for_dit} != model.Nx={Nx}")
        model = FoundationScore(
            in_channels=in_channels, Nx=Nx, dit_kwargs=dit_kwargs,
        )
    elif model_type == "dit_bvaware":
        model = BVAwareScore(
            in_channels=in_channels, backbone="dit",
            dit_kwargs=dit_kwargs, n_pde_types=n_pde_types,
        )
    else:
        raise ValueError(f"未知 model.type='{model_type}'")
    return model.to(device)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def eval_foundation(args: argparse.Namespace) -> None:
    device = torch.device(env.default_device)

    # ---- 加载 ckpt + 重建模型 ----
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt 不存在: {ckpt_path}")
    print(f"[eval_foundation] 加载 ckpt: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if not isinstance(state, dict) or "config" not in state:
        raise ValueError(
            f"ckpt 缺少内嵌 'config' 字段, 无法自动重建模型. "
            f"请使用 train_foundation.py 产出的 ckpt."
        )
    cfg = state["config"]
    pde_names_trained = state.get("pde_names", [])

    # ---- 数据 (test split) ----
    data_cfg = cfg["data"]
    pdes = data_cfg["pdes"]
    test_dataset = MixedPDEDataset(
        pdes_config=pdes,
        data_dir=env.data_dir,
        mode="test",
        conditioning_type=data_cfg.get("conditioning_type", "ic"),
        mix_strategy="uniform",   # eval 不需要 weighted
    )
    n_pde_types = test_dataset.num_pde_types
    print(f"[eval_foundation] PDE 列表: {test_dataset.pde_names}")

    # ---- 重建模型 + 加载权重 ----
    model = build_model_from_ckpt_cfg(cfg, n_pde_types=n_pde_types, device=device)
    model.load_state_dict(state["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eval_foundation] 模型 type={cfg['model']['type']}  params={n_params:,}")

    # ---- 输出目录 ----
    out_dir = env.output_dir / "foundation_eval" / ckpt_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 采样参数 ----
    nu = float(cfg["experiment"].get("nu", 1.0))
    tau_max = float(cfg["experiment"].get("tau_max", 1.0))
    num_steps = int(args.num_steps)
    n_samples = int(args.n_samples)

    # ---- 按 PDE 评估 ----
    table_rows: list[dict] = []
    for pde_id in range(n_pde_types):
        pde_name = test_dataset.get_pde_name(pde_id)
        flux_type = test_dataset.get_flux_type(pde_id)
        print(f"\n--- 评估 PDE [{pde_id}] {pde_name} (flux={flux_type}) ---")

        # 收集该 PDE 的 test 样本 (前 n_samples 个)
        gt_list, ic_list = [], []
        collected = 0
        for global_idx in range(len(test_dataset)):
            item = test_dataset[global_idx]
            if item["pde_id"] != pde_id:
                continue
            gt_list.append(item["x_target"])    # (1, Nx)
            ic_list.append(item["ic"])          # (1, Nx)
            collected += 1
            if collected >= n_samples:
                break
        if collected == 0:
            print(f"  [warn] PDE {pde_name} test split 为空, 跳过")
            continue

        # Stack 成 batch
        gt_batch = torch.stack(gt_list, dim=0).to(device)        # (n, 1, Nx)
        ic_batch = torch.stack(ic_list, dim=0).to(device)        # (n, 1, Nx)
        Nx = gt_batch.shape[-1]
        x_grid = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
        pde_id_tensor = torch.full((collected,), pde_id, dtype=torch.long, device=device)

        # ---- Heun 采样生成 ----
        gen_batch = entrodiff_heun_sampler(
            model,
            shape=(collected, 1, Nx),
            sigma_min=0.01, sigma_max=10.0,
            tau_max=tau_max, nu=nu,
            num_steps=num_steps,
            device=device,
            zeta_pde=0.0,
            conditioning=ic_batch,
            pde_id=pde_id_tensor,
            flux_type=flux_type,
        )
        # 注意: BVAware 的 sampler 输出 shape 跟 u_tau 走 (B, 1, Nx) ← 这里 OK 因为 cond 仅在外层 cat

        # ---- 逐样本指标 ----
        gen_np = gen_batch.detach().cpu().numpy()        # (n, 1, Nx)
        gt_np = gt_batch.detach().cpu().numpy()           # (n, 1, Nx)
        per_sample_metrics = []
        for i in range(collected):
            m = compute_metrics(gt_np[i, 0], gen_np[i, 0], x_grid)
            per_sample_metrics.append(m)
        # 平均
        avg_metrics = {
            k: float(np.mean([x[k] for x in per_sample_metrics]))
            for k in ["W1", "L1_rel", "shock_err"]
        }
        avg_metrics["pde"] = pde_name
        avg_metrics["n"] = collected
        table_rows.append(avg_metrics)
        print(f"  W1={avg_metrics['W1']:.4f}  L1_rel={avg_metrics['L1_rel']:.4f}  "
              f"shock_err={avg_metrics['shock_err']:.4f}  n={collected}")

        # ---- 出图 (前 4 个样本对比) ----
        n_show = min(4, collected)
        fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 6))
        if n_show == 1:
            axes = axes.reshape(2, 1)   # 保持 2D 索引
        for i in range(n_show):
            axes[0, i].plot(x_grid, gt_np[i, 0], 'k-', label='GT', linewidth=2)
            axes[0, i].plot(x_grid, gen_np[i, 0], 'r--', label='Foundation', linewidth=1.5)
            axes[0, i].set_title(f"{pde_name} #{i}: W1={per_sample_metrics[i]['W1']:.3f}")
            axes[0, i].legend(fontsize=8)
            axes[0, i].grid(alpha=0.3)
            # 误差
            axes[1, i].plot(x_grid, np.abs(gt_np[i, 0] - gen_np[i, 0]), 'b-')
            axes[1, i].set_title("|err|")
            axes[1, i].grid(alpha=0.3)
            axes[1, i].set_xlabel("x")
        fig.suptitle(f"Foundation Model · {pde_name} (model={cfg['model']['type']})")
        fig.tight_layout()
        fig_path = out_dir / f"eval_{pde_name}.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"  [fig] {fig_path}")

    # ---- 输出表格 ----
    print("\n" + "=" * 70)
    print(f"{'PDE':<22}  {'W1':>10}  {'L1_rel':>10}  {'shock_err':>10}  {'n':>5}")
    print("-" * 70)
    table_lines = []
    table_lines.append(f"{'PDE':<22},{'W1':>10},{'L1_rel':>10},{'shock_err':>10},{'n':>5}")
    for row in table_rows:
        line_print = (
            f"{row['pde']:<22}  "
            f"{row['W1']:>10.4f}  {row['L1_rel']:>10.4f}  "
            f"{row['shock_err']:>10.4f}  {row['n']:>5d}"
        )
        line_csv = (
            f"{row['pde']},{row['W1']:.4f},{row['L1_rel']:.4f},"
            f"{row['shock_err']:.4f},{row['n']}"
        )
        print(line_print)
        table_lines.append(line_csv)
    print("=" * 70)
    # 保存 CSV
    table_csv_path = out_dir / "metrics_table.csv"
    with open(table_csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))
    print(f"\n[eval_foundation] 表格 CSV: {table_csv_path}")
    print(f"[eval_foundation] 输出目录: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="train_foundation.py 产出的 ckpt 路径 (.pt)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=8,
        help="每个 PDE 评估的 test 样本数 (默认 8)"
    )
    parser.add_argument(
        "--num_steps", type=int, default=50,
        help="Heun 采样步数 (默认 50, 与 train_bvaware 默认对齐)"
    )
    args = parser.parse_args()
    eval_foundation(args)
