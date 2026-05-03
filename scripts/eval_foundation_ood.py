# ============================================================================
# OOD 评估: 三模型在 OOD 数据上的 W₁/L¹
#
# OOD 集:
#   OOD-1: burgers_ood_kmax10  (k_max=10 vs 训练 k=5)
#   OOD-2: burgers_ood_amp2    (振幅×2 IC)
#   OOD-3: bl_ood_riemann       (BL 训练未见的 Riemann pair)
#
# 关键问题: BVA 是否在 OOD 上比 Plain 退化更慢?
#   若是 → 论文 §5.6 改写为 "BV-aware 不是绝对最优但 OOD 鲁棒"
#   若否 → BV-aware 在 DiT 上对当前 setup 完全不奏效
# ============================================================================

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.stats import wasserstein_distance

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.diffusion.samplers import entrodiff_heun_sampler
from scripts.eval_foundation import build_model_from_ckpt_cfg, compute_metrics


# 默认 ckpt + OOD 数据映射
DEFAULT_CKPTS = {
    "DiT-BVA Big":   "output/experiments/foundation_big/foundation_foundation_big_20260503_024957_ep200.pt",
    "DiT-BVA Small": "output/experiments/foundation_small/foundation_foundation_small_20260503_024957_ep200.pt",
    "DiT-Plain Small": "output/experiments/foundation_small_plain/foundation_foundation_small_plain_20260503_024957_ep200.pt",
}

# OOD 配置: name → (data_file, flux_type, pde_id_to_use)
OOD_DATASETS = [
    {
        "name": "OOD-1 Burgers k=10",
        "data_file": "burgers_ood_kmax10_N200_Nx128.npy",
        "flux_type": "burgers",
        "pde_id": 0,   # 假设 burgers 在 mixed 训练时 pde_id=0
    },
    {
        "name": "OOD-2 Burgers amp×2",
        "data_file": "burgers_ood_amp2_N200_Nx128.npy",
        "flux_type": "burgers",
        "pde_id": 0,
    },
    {
        "name": "OOD-3 BL OOD Riemann",
        "data_file": "bl_ood_riemann_N200_Nx128.npy",
        "flux_type": "buckley_leverett",
        "pde_id": 1,   # bl 在 mixed 训练时 pde_id=1
    },
]

# In-dist baseline (重 eval 一次) 供对比
IN_DIST_DATASETS = [
    {
        "name": "In-dist Burgers",
        "data_file": "burgers_1d_N5000_Nx128.npy",
        "flux_type": "burgers",
        "pde_id": 0,
        "test_split_only": True,   # 用 90/100 的 test split
    },
    {
        "name": "In-dist BL",
        "data_file": "bl_1d_N5000_Nx128.npy",
        "flux_type": "buckley_leverett",
        "pde_id": 1,
        "test_split_only": True,
    },
]


def load_ood_data(data_path: Path, n_samples: int = 16, test_split_only: bool = False) -> dict:
    """加载 OOD .npy, 返回 {ic, x_target}.

    OOD 数据 shape (N, N_time, Nx). 取 first/last frame 作为 ic/target.
    若 test_split_only, 取后 10% (与训练 dataset 切分一致).
    """
    data = np.load(str(data_path))
    if test_split_only:
        n_total = data.shape[0]
        idx_val_end = int(0.9 * n_total)
        data = data[idx_val_end:]
        print(f"  [test split] 取后 10%: {data.shape}")
    # 取前 n_samples
    data = data[:n_samples]
    ic = torch.tensor(data[:, 0, :], dtype=torch.float32).unsqueeze(1)        # (n, 1, Nx)
    x_target = torch.tensor(data[:, -1, :], dtype=torch.float32).unsqueeze(1)  # (n, 1, Nx)
    return {"ic": ic, "x_target": x_target, "n": data.shape[0]}


def evaluate_model_on_ood(
    ckpt_path: Path,
    ood_dataset: dict,
    n_samples: int = 16,
    num_steps: int = 25,
    device: Optional[torch.device] = None,
) -> dict:
    """单 ckpt × 单 OOD 数据集评估."""
    if device is None:
        device = torch.device(env.default_device)

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = state["config"]
    model_type = cfg["model"]["type"]

    # 重建模型 (n_pde_types=2, 与训练一致)
    model = build_model_from_ckpt_cfg(cfg, n_pde_types=2, device=device)
    model.load_state_dict(state["model"])
    model.eval()

    # 加载 OOD 数据
    data_path = env.data_dir / ood_dataset["data_file"]
    if not data_path.exists():
        raise FileNotFoundError(f"OOD 数据不存在: {data_path}")
    test_split = ood_dataset.get("test_split_only", False)
    data = load_ood_data(data_path, n_samples=n_samples, test_split_only=test_split)

    ic_batch = data["ic"].to(device)            # (n, 1, Nx)
    gt_batch = data["x_target"].to(device)
    n = ic_batch.shape[0]
    Nx = ic_batch.shape[-1]
    x_grid = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
    pde_id_tensor = torch.full((n,), ood_dataset["pde_id"], dtype=torch.long, device=device)

    # 采样
    nu = float(cfg["experiment"].get("nu", 1.0))
    tau_max = float(cfg["experiment"].get("tau_max", 1.0))
    gen_batch = entrodiff_heun_sampler(
        model,
        shape=(n, 1, Nx),
        sigma_min=0.01, sigma_max=10.0,
        tau_max=tau_max, nu=nu,
        num_steps=num_steps,
        device=device,
        zeta_pde=0.0,
        conditioning=ic_batch,
        pde_id=pde_id_tensor,
        flux_type=ood_dataset["flux_type"],
    )

    # 指标
    gen_np = gen_batch.detach().cpu().numpy()
    gt_np = gt_batch.detach().cpu().numpy()
    metrics = []
    for i in range(n):
        m = compute_metrics(gt_np[i, 0], gen_np[i, 0], x_grid)
        metrics.append(m)
    avg = {
        "W1": float(np.mean([m["W1"] for m in metrics])),
        "L1_rel": float(np.mean([m["L1_rel"] for m in metrics])),
        "shock_err": float(np.mean([m["shock_err"] for m in metrics])),
        "n_samples": n,
    }
    return avg


def render_table(results: dict, ckpt_names: list[str], dataset_names: list[str]) -> str:
    """生成 markdown 汇总表 + BVA 优势判断."""
    lines = []
    header = "| Setting | " + " | ".join(f"{name} W₁" for name in ckpt_names) + " | BVA 优势? |"
    sep = "|" + "---|" * (len(ckpt_names) + 2)
    lines.append(header)
    lines.append(sep)

    for ds_name in dataset_names:
        # 找各模型在该数据集上的 W1
        row = [ds_name]
        bva_big = bva_small = plain = None
        for ckpt_name in ckpt_names:
            key = (ckpt_name, ds_name)
            v = results.get(key, {}).get("W1", None)
            row.append(f"{v:.4f}" if v is not None else "---")
            if "BVA Big" in ckpt_name:
                bva_big = v
            elif "BVA Small" in ckpt_name:
                bva_small = v
            elif "Plain" in ckpt_name:
                plain = v

        # BVA 优势判断: 任一 BVA 优于 Plain
        if plain is not None and (bva_big is not None or bva_small is not None):
            best_bva = min(filter(None, [bva_big, bva_small]))
            if best_bva < plain:
                advantage = f"✓ ({(plain-best_bva)/plain*100:+.1f}%)"
            else:
                advantage = f"✗ ({(plain-best_bva)/plain*100:+.1f}%)"
        else:
            advantage = "---"
        row.append(advantage)

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Foundation Model OOD 泛化评估")
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--include_in_dist", action="store_true",
                        help="同时跑 in-dist baseline 作为对比")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(env.default_device)
    print(f"[eval_foundation_ood] device={device}  num_steps={args.num_steps}  n_samples={args.n_samples}")

    # 输出目录
    out_dir = Path(args.out_dir) if args.out_dir else env.output_dir / "foundation_ood_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据集列表
    datasets = list(OOD_DATASETS)
    if args.include_in_dist:
        datasets = list(IN_DIST_DATASETS) + datasets

    # 跑所有 ckpt × 所有 数据集
    results = {}
    for ckpt_name, ckpt_rel in DEFAULT_CKPTS.items():
        ckpt_path = PROJECT_ROOT / ckpt_rel
        if not ckpt_path.exists():
            print(f"[skip] {ckpt_name}: ckpt 不存在 {ckpt_path}")
            continue
        print(f"\n=== {ckpt_name} ({ckpt_path.name}) ===")
        for ds in datasets:
            print(f"\n  [{ds['name']}] data={ds['data_file']}")
            try:
                avg = evaluate_model_on_ood(
                    ckpt_path, ds,
                    n_samples=args.n_samples,
                    num_steps=args.num_steps,
                    device=device,
                )
                results[(ckpt_name, ds["name"])] = avg
                print(f"    W₁={avg['W1']:.4f}  L¹={avg['L1_rel']:.4f}  shock_err={avg['shock_err']:.4f}  n={avg['n_samples']}")
            except Exception as e:
                print(f"    [error] {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()

    # 汇总表
    ckpt_names_ordered = list(DEFAULT_CKPTS.keys())
    ds_names_ordered = [d["name"] for d in datasets]
    md_table = render_table(results, ckpt_names_ordered, ds_names_ordered)

    # 输出 markdown
    md_path = out_dir / "summary_ood.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Foundation Model OOD 泛化评估\n\n")
        f.write(f"- num_steps: {args.num_steps}\n")
        f.write(f"- n_samples per dataset: {args.n_samples}\n")
        f.write(f"- Models: {list(DEFAULT_CKPTS.keys())}\n\n")
        f.write("## 主表 (W₁ ↓, BVA 优势 = 任一 BVA 比 Plain 低多少)\n\n")
        f.write(md_table + "\n\n")
        # 详细表 (含 L1 和 shock_err)
        f.write("## 详细 (含 L¹ 和 shock_err)\n\n")
        for ds_name in ds_names_ordered:
            f.write(f"### {ds_name}\n\n")
            f.write("| Model | W₁ | L¹ | shock_err |\n|---|---|---|---|\n")
            for ckpt_name in ckpt_names_ordered:
                avg = results.get((ckpt_name, ds_name))
                if avg:
                    f.write(f"| {ckpt_name} | {avg['W1']:.4f} | {avg['L1_rel']:.4f} | {avg['shock_err']:.4f} |\n")
                else:
                    f.write(f"| {ckpt_name} | --- | --- | --- |\n")
            f.write("\n")

    # JSON 全量
    json_path = out_dir / "summary_ood.json"
    json_results = {f"{ck}|{ds}": v for (ck, ds), v in results.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"num_steps": args.num_steps, "n_samples": args.n_samples},
            "results": json_results,
        }, f, indent=2, ensure_ascii=False)

    # 终端打印
    print("\n" + "=" * 80)
    print("OOD 评估汇总:")
    print("=" * 80)
    print(md_table)
    print("=" * 80)
    print(f"\n[md] {md_path}")
    print(f"[json] {json_path}")


if __name__ == "__main__":
    main()
