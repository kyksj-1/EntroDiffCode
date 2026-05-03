# ============================================================================
# Revisit 实验主表 eval (2026-05-04 凌晨)
#
# 4 实验 × 5 seed × 3 setting (Burgers in-dist + 2 Burgers OOD):
#   A1: UNet StandardScore (论文 §5.2 baseline)
#   A2: UNet BVAwareScore (论文 §5.2 核心方法)
#   B1: DiT Plain (1000 samples, capacity 控制)
#   B2: DiT BVAware (1000 samples, capacity 控制)
#
# 期望发现:
#   A2 vs A1: BVA 在 UNet 容量瓶颈下胜过 Standard (论文 §5.2 复现)
#   B2 vs B1: BVA 在数据少 (1000 vs 5000) 时胜过 Plain (capacity 控制)
#
# 用法:
#   python scripts/eval_revisit.py --auto --num_steps 25 --n_samples 16
# ============================================================================

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.diffusion.samplers import entrodiff_heun_sampler
from src.models.score_param import StandardScore, BVAwareScore
from src.models.foundation_score import FoundationScore
from scripts.eval_foundation import compute_metrics


# 3 setting (Burgers in-dist + 2 OOD; B 系列 1000 samples 训, A 系列 4000 samples 训)
DATASETS = [
    {"name": "In-dist Burgers", "data_file": "burgers_1d_N5000_Nx128.npy",
     "test_split_only": True, "flux_type": "burgers"},
    {"name": "OOD-1 Burgers k=10", "data_file": "burgers_ood_kmax10_N200_Nx128.npy",
     "test_split_only": False, "flux_type": "burgers"},
    {"name": "OOD-2 Burgers amp×2", "data_file": "burgers_ood_amp2_N200_Nx128.npy",
     "test_split_only": False, "flux_type": "burgers"},
]

SEEDS = [42, 1042, 2042, 3042, 4042]


def load_test_data(data_path: Path, n_samples: int, test_split_only: bool, seed: int) -> dict:
    data = np.load(str(data_path))
    if test_split_only:
        data = data[int(0.9 * data.shape[0]):]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(data.shape[0])[:n_samples]
    data = data[idx]
    ic = torch.tensor(data[:, 0, :], dtype=torch.float32).unsqueeze(1)
    x_target = torch.tensor(data[:, -1, :], dtype=torch.float32).unsqueeze(1)
    return {"ic": ic, "x_target": x_target}


def find_unet_ckpt(exp_name: str, seed: int):
    """A1/A2: train_mvp / train_bvaware 产出 ckpt."""
    suffix = "" if seed == 42 else f"_s{seed}"
    d = env.output_dir / (exp_name + suffix)
    if not d.exists():
        return None
    # train_mvp: entrodiff_{exp_name}_{ts}_ep{N}.pt
    # train_bvaware: entrodiff_{exp_name}_{ts}_ep{N}.pt
    pat = sorted(d.glob("entrodiff_*_ep200.pt"))
    return pat[-1] if pat else None


def find_dit_ckpt(exp_name: str, seed: int):
    """B1/B2: train_foundation 产出 ckpt."""
    suffix = "" if seed == 42 else f"_s{seed}"
    d = env.output_dir / (exp_name + suffix)
    if not d.exists():
        return None
    pat = sorted(d.glob("foundation_*_ep200.pt"))
    return pat[-1] if pat else None


def load_unet_standard(ckpt: Path, in_channels: int, device) -> StandardScore:
    state = torch.load(str(ckpt), map_location=device, weights_only=False)
    m = StandardScore(in_channels=in_channels)
    m.load_state_dict(state)
    return m.to(device).eval()


def load_unet_bvaware(ckpt: Path, in_channels: int, dim: int, device) -> BVAwareScore:
    state = torch.load(str(ckpt), map_location=device, weights_only=False)
    m = BVAwareScore(in_channels=in_channels, dim=dim, return_denoiser=True, backbone="unet")
    m.load_state_dict(state)
    return m.to(device).eval()


def load_dit_ckpt(ckpt: Path, n_pde_types: int, device):
    """加载 train_foundation ckpt (dit_plain 或 dit_bvaware)."""
    from scripts.eval_foundation import build_model_from_ckpt_cfg
    state = torch.load(str(ckpt), map_location=device, weights_only=False)
    cfg = state["config"]
    m = build_model_from_ckpt_cfg(cfg, n_pde_types=n_pde_types, device=device)
    m.load_state_dict(state["model"])
    return m.eval()


def evaluate_method(method_name: str, models_per_seed: list, n_samples: int, num_steps: int,
                    device) -> dict:
    """method × 5 seed × 3 setting eval."""
    out = {}
    for ds in DATASETS:
        per_seed_metrics = []
        for s_i, seed in enumerate(SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)
            data_path = env.data_dir / ds["data_file"]
            if not data_path.exists():
                continue
            data = load_test_data(data_path, n_samples, ds.get("test_split_only", False), seed)
            ic_b = data["ic"].to(device)
            gt_b = data["x_target"].to(device)
            n = ic_b.shape[0]
            Nx = ic_b.shape[-1]
            x_grid = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
            pid_t = torch.zeros(n, dtype=torch.long, device=device)

            mod = models_per_seed[s_i]
            if mod is None:
                continue
            try:
                gen = entrodiff_heun_sampler(
                    mod, shape=(n, 1, Nx),
                    sigma_min=0.01, sigma_max=10.0,
                    tau_max=1.0, nu=1.0, num_steps=num_steps,
                    device=device, zeta_pde=0.0,
                    conditioning=ic_b, pde_id=pid_t,
                    flux_type=ds["flux_type"],
                )
                gen_np = gen.detach().cpu().numpy()
                gt_np = gt_b.detach().cpu().numpy()
                metrics = [compute_metrics(gt_np[i, 0], gen_np[i, 0], x_grid) for i in range(n)]
                per_seed_metrics.append({
                    "seed": seed,
                    "W1": float(np.mean([m["W1"] for m in metrics])),
                    "L1_rel": float(np.mean([m["L1_rel"] for m in metrics])),
                    "shock_err": float(np.mean([m["shock_err"] for m in metrics])),
                })
            except Exception as e:
                print(f"      [error] {method_name} seed={seed} {ds['name']}: {e}")
        if per_seed_metrics:
            keys = ["W1", "L1_rel", "shock_err"]
            agg = {k + "_mean": float(np.mean([r[k] for r in per_seed_metrics])) for k in keys}
            agg.update({k + "_std": float(np.std([r[k] for r in per_seed_metrics])) for k in keys})
            agg["per_seed"] = per_seed_metrics
            agg["n_seeds"] = len(per_seed_metrics)
            out[ds["name"]] = agg
            print(f"    {ds['name']}: W₁={agg['W1_mean']:.4f}±{agg['W1_std']:.4f}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(env.default_device)
    print(f"[eval_revisit] device={device} num_steps={args.num_steps} n_samples={args.n_samples}")

    out_dir = Path(args.out_dir) if args.out_dir else env.output_dir / "revisit_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ---- A1: UNet StandardScore ----
    print("\n=== A1: UNet StandardScore (revisit §5.2 baseline) ===")
    a1_ckpts = [find_unet_ckpt("revisit_unet_standard", s) for s in SEEDS]
    a1_valid = [c for c in a1_ckpts if c is not None]
    if len(a1_valid) == len(SEEDS):
        a1_models = [load_unet_standard(c, in_channels=2, device=device) for c in a1_ckpts]
        all_results["A1: UNet Standard"] = evaluate_method("A1", a1_models, args.n_samples, args.num_steps, device)
        for m in a1_models:
            del m
        torch.cuda.empty_cache() if device.type == "cuda" else None
    else:
        print(f"  [skip] 仅 {len(a1_valid)}/{len(SEEDS)} ckpt 就绪")

    # ---- A2: UNet BVAwareScore ----
    print("\n=== A2: UNet BVAwareScore (revisit §5.2 核心方法) ===")
    a2_ckpts = [find_unet_ckpt("revisit_unet_bvaware", s) for s in SEEDS]
    a2_valid = [c for c in a2_ckpts if c is not None]
    if len(a2_valid) == len(SEEDS):
        a2_models = [load_unet_bvaware(c, in_channels=2, dim=128, device=device) for c in a2_ckpts]
        all_results["A2: UNet BVAware"] = evaluate_method("A2", a2_models, args.n_samples, args.num_steps, device)
        for m in a2_models:
            del m
        torch.cuda.empty_cache() if device.type == "cuda" else None
    else:
        print(f"  [skip] 仅 {len(a2_valid)}/{len(SEEDS)} ckpt 就绪")

    # ---- B1: DiT Plain (1000 samples) ----
    print("\n=== B1: DiT Plain on 1000 samples ===")
    b1_ckpts = [find_dit_ckpt("revisit_dit_plain_small_data", s) for s in SEEDS]
    b1_valid = [c for c in b1_ckpts if c is not None]
    if len(b1_valid) == len(SEEDS):
        b1_models = [load_dit_ckpt(c, n_pde_types=1, device=device) for c in b1_ckpts]
        all_results["B1: DiT Plain (1000)"] = evaluate_method("B1", b1_models, args.n_samples, args.num_steps, device)
        for m in b1_models:
            del m
        torch.cuda.empty_cache() if device.type == "cuda" else None
    else:
        print(f"  [skip] 仅 {len(b1_valid)}/{len(SEEDS)} ckpt 就绪")

    # ---- B2: DiT BVAware (1000 samples) ----
    print("\n=== B2: DiT BVAware on 1000 samples ===")
    b2_ckpts = [find_dit_ckpt("revisit_dit_bvaware_small_data", s) for s in SEEDS]
    b2_valid = [c for c in b2_ckpts if c is not None]
    if len(b2_valid) == len(SEEDS):
        b2_models = [load_dit_ckpt(c, n_pde_types=1, device=device) for c in b2_ckpts]
        all_results["B2: DiT BVAware (1000)"] = evaluate_method("B2", b2_models, args.n_samples, args.num_steps, device)
        for m in b2_models:
            del m
        torch.cuda.empty_cache() if device.type == "cuda" else None
    else:
        print(f"  [skip] 仅 {len(b2_valid)}/{len(SEEDS)} ckpt 就绪")

    # ---- 主表 markdown ----
    md = ["# Revisit 实验主表 (2026-05-04 凌晨, 5-seed mean ± std)\n",
          f"- num_steps: {args.num_steps}, n_samples: {args.n_samples}, n_seeds: {len(SEEDS)}\n",
          "## W₁ ↓ (paired t-test 标★ p<0.05)\n",
          "| Method | " + " | ".join(d["name"] for d in DATASETS) + " |",
          "|" + "---|" * (len(DATASETS) + 1)]

    for method_name, settings in all_results.items():
        row = [method_name]
        for ds in DATASETS:
            r = settings.get(ds["name"])
            if r is None:
                row.append("---")
                continue
            cell = f"{r['W1_mean']:.4f}±{r['W1_std']:.4f}"
            row.append(cell)
        md.append("| " + " | ".join(row) + " |")

    # 关键对比 (A2 vs A1, B2 vs B1)
    md.append("\n## 关键对比 (BVAware vs Plain/Standard, ★ p<0.05)\n")
    pairs = [("A1: UNet Standard", "A2: UNet BVAware", "UNet (4000 samples)"),
             ("B1: DiT Plain (1000)", "B2: DiT BVAware (1000)", "DiT (1000 samples)")]
    md.append("| Pair | " + " | ".join(d["name"] for d in DATASETS) + " |")
    md.append("|" + "---|" * (len(DATASETS) + 1))
    for std_name, bva_name, label in pairs:
        std = all_results.get(std_name, {})
        bva = all_results.get(bva_name, {})
        if not std or not bva:
            continue
        row = [f"{label}: BVA vs Plain/Std"]
        for ds in DATASETS:
            std_r = std.get(ds["name"])
            bva_r = bva.get(ds["name"])
            if std_r is None or bva_r is None:
                row.append("---")
                continue
            arr_std = np.array([s["W1"] for s in std_r["per_seed"]])
            arr_bva = np.array([s["W1"] for s in bva_r["per_seed"]])
            if len(arr_std) > 1 and len(arr_bva) == len(arr_std):
                tstat, pval = ttest_rel(arr_bva, arr_std)
                delta = (std_r["W1_mean"] - bva_r["W1_mean"]) / std_r["W1_mean"] * 100
                tag = " ★" if pval < 0.05 else ""
                cell = f"{delta:+.1f}% (p={pval:.3f}){tag}"
            else:
                cell = "n/a"
            row.append(cell)
        md.append("| " + " | ".join(row) + " |")

    # L1 + shock_err
    for metric, title in [("L1_rel", "L¹"), ("shock_err", "shock_err")]:
        md.append(f"\n## {title} ↓ (mean ± std)\n")
        md.append("| Method | " + " | ".join(d["name"] for d in DATASETS) + " |")
        md.append("|" + "---|" * (len(DATASETS) + 1))
        for method_name, settings in all_results.items():
            row = [method_name]
            for ds in DATASETS:
                r = settings.get(ds["name"])
                row.append(f"{r[metric+'_mean']:.4f}±{r[metric+'_std']:.4f}" if r else "---")
            md.append("| " + " | ".join(row) + " |")

    md_text = "\n".join(md)
    md_path = out_dir / "summary_revisit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    json_path = out_dir / "summary_revisit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(md_text)
    print("=" * 80)
    print(f"\n[md] {md_path}\n[json] {json_path}")


if __name__ == "__main__":
    main()
