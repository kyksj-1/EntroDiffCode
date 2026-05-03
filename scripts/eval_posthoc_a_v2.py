# ============================================================================
# Eval LearnedLambdaSchedule (PostHoc-A v2)
#
# 用法:
#   python scripts/eval_posthoc_a_v2.py --lambda_ckpt <path> [--n_seeds 5]
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
from src.diffusion.posthoc_bv_sampler_v2 import posthoc_bv_heun_sampler_v2
from src.diffusion.samplers import entrodiff_heun_sampler
from src.models.foundation_score import FoundationScore
from src.models.learned_lambda_schedule import LearnedLambdaSchedule
from scripts.eval_foundation import compute_metrics


# 5 个 setting (与 eval_posthoc_a.py 对齐)
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


def load_test_data(data_path: Path, n_samples: int, test_split_only: bool, seed: int = 42) -> dict:
    data = np.load(str(data_path))
    if test_split_only:
        idx_val_end = int(0.9 * data.shape[0])
        data = data[idx_val_end:]
    # seed 用于打乱选择 n_samples 子集 (5-seed eval 时让每 seed 选不同样本)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(data.shape[0])[:n_samples]
    data = data[idx]
    ic = torch.tensor(data[:, 0, :], dtype=torch.float32).unsqueeze(1)
    x_target = torch.tensor(data[:, -1, :], dtype=torch.float32).unsqueeze(1)
    return {"ic": ic, "x_target": x_target, "n": data.shape[0]}


def build_plain_and_lambda(lambda_ckpt: Path, n_pde_types: int, device):
    """加载 LearnedLambda ckpt + plain backbone."""
    state = torch.load(str(lambda_ckpt), map_location=device, weights_only=False)
    plain_path = Path(state["plain_ckpt_path"])
    if not plain_path.exists():
        plain_path = PROJECT_ROOT / "output/experiments/foundation_small_plain" / plain_path.name
    plain_state = torch.load(str(plain_path), map_location=device, weights_only=False)
    plain_cfg = plain_state["config"]
    dit_kwargs = dict(plain_cfg["model"]["dit"])
    dit_kwargs["n_pde_types"] = n_pde_types
    in_channels = int(plain_cfg["model"].get("in_channels", 2))
    Nx = int(dit_kwargs.get("Nx", 128))
    nx_for_dit = dit_kwargs.pop("Nx", Nx)
    plain_backbone = FoundationScore(in_channels=in_channels, Nx=Nx, dit_kwargs=dit_kwargs)
    plain_backbone.load_state_dict(plain_state["model"])
    plain_backbone = plain_backbone.to(device)
    plain_backbone.eval()

    # LearnedLambda
    cfg = state["config"]
    model_cfg = cfg["model"]
    lambda_module = LearnedLambdaSchedule(
        hidden=int(model_cfg.get("hidden", 64)),
        freq_dim=int(model_cfg.get("freq_dim", 64)),
        lam_max=float(model_cfg.get("lam_max", 5.0)),
        zero_init=False,
    )
    lambda_module.load_state_dict(state["lambda_schedule"])
    lambda_module = lambda_module.to(device)
    lambda_module.eval()
    return plain_backbone, lambda_module


def evaluate_one_seed(plain_backbone, lambda_module, dataset, n_samples, num_steps, device, seed):
    """单 seed eval."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_path = env.data_dir / dataset["data_file"]
    data = load_test_data(data_path, n_samples=n_samples,
                          test_split_only=dataset.get("test_split_only", False),
                          seed=seed)
    ic_b = data["ic"].to(device)
    gt_b = data["x_target"].to(device)
    n = ic_b.shape[0]
    Nx = ic_b.shape[-1]
    x_grid = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
    pid_t = torch.full((n,), dataset["pde_id"], dtype=torch.long, device=device)

    gen = posthoc_bv_heun_sampler_v2(
        plain_backbone, shape=(n, 1, Nx),
        sigma_min=0.01, sigma_max=10.0,
        tau_max=1.0, nu=1.0, num_steps=num_steps,
        device=device, zeta_pde=0.0,
        conditioning=ic_b, pde_id=pid_t,
        flux_type=dataset["flux_type"],
        bv_strength=1.0,
        lambda_module=lambda_module,
        enable_grad_for_lambda_module=False,
    )

    gen_np = gen.detach().cpu().numpy()
    gt_np = gt_b.detach().cpu().numpy()
    metrics = [compute_metrics(gt_np[i, 0], gen_np[i, 0], x_grid) for i in range(n)]
    return {
        "W1": float(np.mean([m["W1"] for m in metrics])),
        "L1_rel": float(np.mean([m["L1_rel"] for m in metrics])),
        "shock_err": float(np.mean([m["shock_err"] for m in metrics])),
        "n": n, "seed": seed,
    }


def evaluate_plain_one_seed(plain_backbone, dataset, n_samples, num_steps, device, seed):
    """对照: plain 不加 BV 修正."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    data_path = env.data_dir / dataset["data_file"]
    data = load_test_data(data_path, n_samples=n_samples,
                          test_split_only=dataset.get("test_split_only", False), seed=seed)
    ic_b = data["ic"].to(device)
    gt_b = data["x_target"].to(device)
    n = ic_b.shape[0]
    Nx = ic_b.shape[-1]
    x_grid = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
    pid_t = torch.full((n,), dataset["pde_id"], dtype=torch.long, device=device)

    gen = entrodiff_heun_sampler(
        plain_backbone, shape=(n, 1, Nx),
        sigma_min=0.01, sigma_max=10.0,
        tau_max=1.0, nu=1.0, num_steps=num_steps,
        device=device, zeta_pde=0.0,
        conditioning=ic_b, pde_id=pid_t,
        flux_type=dataset["flux_type"],
    )
    gen_np = gen.detach().cpu().numpy()
    gt_np = gt_b.detach().cpu().numpy()
    metrics = [compute_metrics(gt_np[i, 0], gen_np[i, 0], x_grid) for i in range(n)]
    return {
        "W1": float(np.mean([m["W1"] for m in metrics])),
        "L1_rel": float(np.mean([m["L1_rel"] for m in metrics])),
        "shock_err": float(np.mean([m["shock_err"] for m in metrics])),
        "n": n, "seed": seed,
    }


def aggregate(per_seed: list[dict]) -> dict:
    """5-seed 聚合 mean ± std."""
    keys = ["W1", "L1_rel", "shock_err"]
    out = {}
    for k in keys:
        vals = [r[k] for r in per_seed]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    out["n_seeds"] = len(per_seed)
    out["per_seed"] = per_seed
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_ckpt", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--n_seeds", type=int, default=5, help="多 seed eval 取 mean ± std")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--include_plain_compare", action="store_true",
                        help="同时跑 plain 对照, 输出 t-test")
    args = parser.parse_args()

    device = torch.device(env.default_device)
    print(f"[posthoc_a_v2_eval] device={device} num_steps={args.num_steps} n_samples={args.n_samples} n_seeds={args.n_seeds}")

    out_dir = Path(args.out_dir) if args.out_dir else env.output_dir / "posthoc_a_v2_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    lambda_ckpt = Path(args.lambda_ckpt) if Path(args.lambda_ckpt).is_absolute() else PROJECT_ROOT / args.lambda_ckpt
    plain_backbone, lambda_module = build_plain_and_lambda(lambda_ckpt, n_pde_types=2, device=device)

    results_v2 = {}
    results_plain = {}
    for ds in DATASETS:
        print(f"\n  [{ds['name']}] PostHoc-A v2 ({args.n_seeds} seeds)")
        per_seed_v2 = []
        for s in range(args.n_seeds):
            r = evaluate_one_seed(plain_backbone, lambda_module, ds,
                                  n_samples=args.n_samples, num_steps=args.num_steps,
                                  device=device, seed=42 + s * 1000)
            per_seed_v2.append(r)
            print(f"    seed={42+s*1000:5d}: W₁={r['W1']:.4f} L¹={r['L1_rel']:.4f} shock={r['shock_err']:.4f}")
        results_v2[ds["name"]] = aggregate(per_seed_v2)

        if args.include_plain_compare:
            print(f"  [{ds['name']}] Plain ({args.n_seeds} seeds)")
            per_seed_plain = []
            for s in range(args.n_seeds):
                r = evaluate_plain_one_seed(plain_backbone, ds,
                                            n_samples=args.n_samples, num_steps=args.num_steps,
                                            device=device, seed=42 + s * 1000)
                per_seed_plain.append(r)
                print(f"    seed={42+s*1000:5d}: W₁={r['W1']:.4f}")
            results_plain[ds["name"]] = aggregate(per_seed_plain)

    # markdown 输出
    md = ["# Post-hoc BV-aware A v2 (LearnedLambdaSchedule) Eval\n",
          f"- ckpt: {lambda_ckpt.name}",
          f"- num_steps: {args.num_steps}, n_samples: {args.n_samples}, n_seeds: {args.n_seeds}\n"]

    md.append("## 主表 W₁ ↓ (mean ± std)\n")
    if results_plain:
        md.append("| Setting | Plain W₁ | PostHoc-A v2 W₁ | Δ% | t-test p |")
        md.append("|---|---|---|---|---|")
        for ds in DATASETS:
            n = ds["name"]
            v2 = results_v2[n]
            pl = results_plain.get(n, None)
            if pl is None:
                continue
            # paired t-test
            from scipy.stats import ttest_rel
            v2_arr = np.array([r["W1"] for r in v2["per_seed"]])
            pl_arr = np.array([r["W1"] for r in pl["per_seed"]])
            tstat, pval = ttest_rel(v2_arr, pl_arr)
            delta = (pl["W1_mean"] - v2["W1_mean"]) / pl["W1_mean"] * 100
            tag = " ★" if (delta > 0 and pval < 0.05) else ""
            md.append(
                f"| {n} | {pl['W1_mean']:.4f} ± {pl['W1_std']:.4f} | "
                f"{v2['W1_mean']:.4f} ± {v2['W1_std']:.4f} | "
                f"{delta:+.1f}% | {pval:.4f}{tag} |"
            )
    else:
        md.append("| Setting | PostHoc-A v2 W₁ | L¹ | shock_err |")
        md.append("|---|---|---|---|")
        for ds in DATASETS:
            n = ds["name"]
            r = results_v2[n]
            md.append(f"| {n} | {r['W1_mean']:.4f} ± {r['W1_std']:.4f} | "
                     f"{r['L1_rel_mean']:.4f} ± {r['L1_rel_std']:.4f} | "
                     f"{r['shock_err_mean']:.4f} ± {r['shock_err_std']:.4f} |")

    md_text = "\n".join(md)
    md_path = out_dir / "summary_posthoc_a_v2.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    json_path = out_dir / "summary_posthoc_a_v2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "lambda_ckpt": str(lambda_ckpt),
            "v2_results": results_v2,
            "plain_results": results_plain,
            "args": vars(args),
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(md_text)
    print("=" * 80)
    print(f"[md] {md_path}\n[json] {json_path}")


if __name__ == "__main__":
    main()
