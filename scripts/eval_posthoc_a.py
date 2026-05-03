# ============================================================================
# 方案 A 评估: Training-free Post-hoc BV-aware sampler vs Plain
#
# 论文 §5.6 修正叙事验证:
#   用已训 dit_plain ckpt 不变, 采样时加 BV-aware 修正
#   对比:
#     (a) Plain ckpt + 标准 sampler (= 之前 baseline 0.0815)
#     (b) Plain ckpt + post-hoc BV sampler (新方案 A)
#   看 (b) 是否 W₁ < (a)
#
# 在 in-dist + 3 OOD 上对比, 看 OOD-2 amp×2 是否能从 0.2740 拉低
# ============================================================================

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.diffusion.samplers import entrodiff_heun_sampler
from src.diffusion.posthoc_bv_sampler import posthoc_bv_heun_sampler
from scripts.eval_foundation import build_model_from_ckpt_cfg, compute_metrics


# 默认仅评估 dit_plain ckpt (方案 A 的核心: 用已训 plain backbone)
DEFAULT_CKPT = "output/experiments/foundation_small_plain/foundation_foundation_small_plain_20260503_024957_ep200.pt"

# 5 个数据集 (与 eval_foundation_ood 对齐)
DATASETS = [
    {"name": "In-dist Burgers", "data_file": "burgers_1d_N5000_Nx128.npy",
     "flux_type": "burgers", "pde_id": 0, "test_split_only": True},
    {"name": "In-dist BL", "data_file": "bl_1d_N5000_Nx128.npy",
     "flux_type": "buckley_leverett", "pde_id": 1, "test_split_only": True},
    {"name": "OOD-1 Burgers k=10", "data_file": "burgers_ood_kmax10_N200_Nx128.npy",
     "flux_type": "burgers", "pde_id": 0},
    {"name": "OOD-2 Burgers amp×2", "data_file": "burgers_ood_amp2_N200_Nx128.npy",
     "flux_type": "burgers", "pde_id": 0},
    {"name": "OOD-3 BL OOD Riemann", "data_file": "bl_ood_riemann_N200_Nx128.npy",
     "flux_type": "buckley_leverett", "pde_id": 1},
]

# 待对比的 sampler 模式 (默认; 命令行可覆盖 --bv_strengths)
DEFAULT_BV_STRENGTHS = [0.0, 0.5, 1.0, 2.0]


def build_sampler_modes(bv_strengths: list, lambda_mode: str = "exp_decay") -> list:
    """从 bv_strength 列表 + lambda_mode 构造 SAMPLER_MODES."""
    modes = []
    for s in bv_strengths:
        if s == 0.0:
            modes.append({"name": "Plain (no posthoc)", "bv_strength": 0.0, "lambda_mode": lambda_mode})
        else:
            modes.append({"name": f"Plain + posthoc BV (s={s})", "bv_strength": float(s), "lambda_mode": lambda_mode})
    return modes


def load_test_data(data_path: Path, n_samples: int, test_split_only: bool) -> dict:
    data = np.load(str(data_path))
    if test_split_only:
        idx_val_end = int(0.9 * data.shape[0])
        data = data[idx_val_end:]
    data = data[:n_samples]
    ic = torch.tensor(data[:, 0, :], dtype=torch.float32).unsqueeze(1)
    x_target = torch.tensor(data[:, -1, :], dtype=torch.float32).unsqueeze(1)
    return {"ic": ic, "x_target": x_target, "n": data.shape[0]}


def evaluate(ckpt_path, dataset, sampler_mode, n_samples=16, num_steps=25, device=None):
    if device is None:
        device = torch.device(env.default_device)

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = state["config"]
    model = build_model_from_ckpt_cfg(cfg, n_pde_types=2, device=device)
    model.load_state_dict(state["model"])
    model.eval()

    data_path = env.data_dir / dataset["data_file"]
    data = load_test_data(data_path, n_samples=n_samples,
                          test_split_only=dataset.get("test_split_only", False))
    ic_b = data["ic"].to(device)
    gt_b = data["x_target"].to(device)
    n = ic_b.shape[0]
    Nx = ic_b.shape[-1]
    x_grid = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
    pid_t = torch.full((n,), dataset["pde_id"], dtype=torch.long, device=device)

    nu = float(cfg["experiment"].get("nu", 1.0))
    tau_max = float(cfg["experiment"].get("tau_max", 1.0))

    if sampler_mode["bv_strength"] == 0.0:
        # 标准 sampler
        gen = entrodiff_heun_sampler(
            model, shape=(n, 1, Nx),
            sigma_min=0.01, sigma_max=10.0,
            tau_max=tau_max, nu=nu, num_steps=num_steps,
            device=device, zeta_pde=0.0,
            conditioning=ic_b, pde_id=pid_t,
            flux_type=dataset["flux_type"],
        )
    else:
        # Post-hoc BV-aware sampler
        gen = posthoc_bv_heun_sampler(
            model, shape=(n, 1, Nx),
            sigma_min=0.01, sigma_max=10.0,
            tau_max=tau_max, nu=nu, num_steps=num_steps,
            device=device, zeta_pde=0.0,
            conditioning=ic_b, pde_id=pid_t,
            flux_type=dataset["flux_type"],
            bv_strength=sampler_mode["bv_strength"],
            lambda_mode=sampler_mode["lambda_mode"],
        )

    gen_np = gen.detach().cpu().numpy()
    gt_np = gt_b.detach().cpu().numpy()
    metrics = [compute_metrics(gt_np[i, 0], gen_np[i, 0], x_grid) for i in range(n)]
    return {
        "W1": float(np.mean([m["W1"] for m in metrics])),
        "L1_rel": float(np.mean([m["L1_rel"] for m in metrics])),
        "shock_err": float(np.mean([m["shock_err"] for m in metrics])),
        "n_samples": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(PROJECT_ROOT / DEFAULT_CKPT))
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--n_seeds", type=int, default=1, help="多 seed eval")
    parser.add_argument("--bv_strengths", type=float, nargs="+", default=DEFAULT_BV_STRENGTHS,
                        help="bv_strength 列表 (覆盖默认 0/0.5/1.0/2.0)")
    parser.add_argument("--lambda_mode", type=str, default="exp_decay",
                        choices=["exp_decay", "inv_linear", "sigmoid"])
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    SAMPLER_MODES = build_sampler_modes(args.bv_strengths, args.lambda_mode)

    device = torch.device(env.default_device)
    print(f"[posthoc_a] device={device} num_steps={args.num_steps} n_samples={args.n_samples} n_seeds={args.n_seeds}")
    print(f"[posthoc_a] ckpt: {args.ckpt}")
    print(f"[posthoc_a] sampler modes: {[m['name'] for m in SAMPLER_MODES]}")

    out_dir = Path(args.out_dir) if args.out_dir else env.output_dir / "posthoc_a_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sweep: 每个 (ds, sm) 跑 n_seeds 次取均值
    import numpy as np_
    results = {}
    for ds in DATASETS:
        for sm in SAMPLER_MODES:
            print(f"\n  [{ds['name']}]  [{sm['name']}]")
            per_seed = []
            for s_i in range(args.n_seeds):
                seed = 42 + s_i * 1000
                torch.manual_seed(seed)
                np_.random.seed(seed)
                try:
                    r = evaluate(args.ckpt, ds, sm,
                                 n_samples=args.n_samples, num_steps=args.num_steps,
                                 device=device)
                    per_seed.append(r)
                    if args.n_seeds > 1:
                        print(f"    seed={seed:5d}: W₁={r['W1']:.4f}")
                    else:
                        print(f"    W₁={r['W1']:.4f}  L¹={r['L1_rel']:.4f}  shock={r['shock_err']:.4f}")
                except Exception as e:
                    print(f"    [error] {e}")
                    import traceback; traceback.print_exc()
            # 聚合
            if per_seed:
                keys = ["W1", "L1_rel", "shock_err"]
                agg = {k + "_mean": float(np_.mean([r[k] for r in per_seed])) for k in keys}
                agg.update({k + "_std": float(np_.std([r[k] for r in per_seed])) for k in keys})
                agg["n_seeds"] = len(per_seed)
                agg["per_seed"] = per_seed
                results[(ds["name"], sm["name"])] = agg

    # 汇总表
    sm_names = [s["name"] for s in SAMPLER_MODES]
    ds_names = [d["name"] for d in DATASETS]

    md = ["# Post-hoc BV-aware Sampler (方案 A) Eval\n",
          f"- ckpt: foundation_small_plain (DiT-Plain, 7.4M)",
          f"- num_steps: {args.num_steps}, n_samples: {args.n_samples}, n_seeds: {args.n_seeds}",
          f"- bv_strengths: {args.bv_strengths}",
          f"- lambda_mode: {args.lambda_mode}\n",
          "## W₁ ↓ (越低越好)\n"]
    if args.n_seeds > 1:
        md.append("| Setting | " + " | ".join(f"{n} (mean ± std)" for n in sm_names) + " | 最佳救场 |")
    else:
        md.append("| Setting | " + " | ".join(sm_names) + " | 最佳救场 |")
    md.append("|" + "---|" * (len(sm_names) + 2))

    for ds_name in ds_names:
        row = [ds_name]
        baseline = None
        best_posthoc = None
        best_mode = None
        for sm_name in sm_names:
            r = results.get((ds_name, sm_name))
            if r is None:
                row.append("---")
                continue
            v = r["W1_mean"]
            if args.n_seeds > 1:
                cell = f"{v:.4f} ± {r['W1_std']:.4f}"
            else:
                cell = f"{v:.4f}"
            row.append(cell)
            if "Plain (no posthoc)" in sm_name:
                baseline = v
            elif "posthoc" in sm_name.lower():
                if best_posthoc is None or v < best_posthoc:
                    best_posthoc = v
                    best_mode = sm_name

        if baseline and best_posthoc:
            delta = (baseline - best_posthoc) / baseline * 100
            tag = f"{'✓' if delta > 0 else '✗'} ({delta:+.1f}% by {best_mode.replace('Plain + posthoc BV ', '')})"
        else:
            tag = "---"
        row.append(tag)
        md.append("| " + " | ".join(row) + " |")

    md_text = "\n".join(md)
    md_path = out_dir / "summary_posthoc_a.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    json_path = out_dir / "summary_posthoc_a.json"
    # JSON keys 序列化
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args),
                   "results": {f"{k[0]}|{k[1]}": v for k, v in results.items()}},
                  f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(md_text)
    print("=" * 80)
    print(f"\n[md] {md_path}\n[json] {json_path}")


if __name__ == "__main__":
    main()
