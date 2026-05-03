# ============================================================================
# 论文 §5.6 最终主表 Eval — 全方法 × 5 seed × 5 setting
#
# 评估的方法 (按论文叙事顺序):
#   1. Plain (DiT, 无 PostHoc)                    — baseline
#   2. PostHoc-A v1 (training-free, fixed exp_decay)— 方案 A 旧版 (可选)
#   3. PostHoc-A v2 (LearnedLambda)               — 方案 1
#   4. PostHoc-B small (386 trainable, EMA off/on) — 方案 3a
#   5. PostHoc-B large (7K trainable, EMA on)     — 方案 3b
#   6. PostHoc-B aug (386 + IC scale + noise + EMA)— 方案 4
#
# 输出:
#   - 论文级 markdown 主表: row=method (含 Plain), col=5 setting × W₁ mean±std
#   - paired t-test p-value (vs Plain) 标记 ★ (p<0.05)
#   - 详细 JSON (per-seed 数据点)
#
# 关键设计 (TRICK §2.4 + §5.1 反向用):
#   - 5 seed × n_samples=16 共 80 个采样, σ=0.005 量级方差能区分 1.8% 改善
#   - paired t-test 因为同一 IC 下不同模型对比, 适用 paired
#   - 如果想要更严, 改用 mean ± 95% CI (boostrap)
#
# 用法:
#   # 自动找全部 ckpt (从 env.output_dir 下 foundation_posthoc_* 各 5 seed)
#   python scripts/eval_boost_full.py --auto --num_steps 25 --n_samples 16
#
#   # 显式指定 (用于复现):
#   python scripts/eval_boost_full.py --plain_ckpt <path> --seeds 42 1042 2042 3042 4042
# ============================================================================

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.diffusion.samplers import entrodiff_heun_sampler
from src.diffusion.posthoc_bv_sampler import posthoc_bv_heun_sampler
from src.diffusion.posthoc_bv_sampler_v2 import posthoc_bv_heun_sampler_v2
from src.models.foundation_score import FoundationScore
from src.models.posthoc_bv_score import PostHocBVAwareScore
from src.models.learned_lambda_schedule import LearnedLambdaSchedule
from scripts.eval_foundation import build_model_from_ckpt_cfg, compute_metrics


# 5 个 setting (与 eval_posthoc_a 对齐)
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


def load_test_data(data_path: Path, n_samples: int, test_split_only: bool, seed: int) -> dict:
    data = np.load(str(data_path))
    if test_split_only:
        idx_val_end = int(0.9 * data.shape[0])
        data = data[idx_val_end:]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(data.shape[0])[:n_samples]
    data = data[idx]
    ic = torch.tensor(data[:, 0, :], dtype=torch.float32).unsqueeze(1)
    x_target = torch.tensor(data[:, -1, :], dtype=torch.float32).unsqueeze(1)
    return {"ic": ic, "x_target": x_target}


def load_plain_backbone(plain_ckpt: Path, n_pde_types: int, device) -> FoundationScore:
    """加载 dit_plain ckpt → FoundationScore (frozen)."""
    state = torch.load(str(plain_ckpt), map_location=device, weights_only=False)
    cfg = state["config"]
    dit_kwargs = dict(cfg["model"]["dit"])
    dit_kwargs["n_pde_types"] = n_pde_types
    in_channels = int(cfg["model"].get("in_channels", 2))
    Nx = int(dit_kwargs.get("Nx", 128))
    nx_for_dit = dit_kwargs.pop("Nx", Nx)
    backbone = FoundationScore(in_channels=in_channels, Nx=Nx, dit_kwargs=dit_kwargs)
    backbone.load_state_dict(state["model"])
    backbone = backbone.to(device).eval()
    return backbone


def load_lambda_module(lambda_ckpt: Path, device) -> LearnedLambdaSchedule:
    """加载 LearnedLambda ckpt."""
    state = torch.load(str(lambda_ckpt), map_location=device, weights_only=False)
    cfg = state["config"]["model"]
    m = LearnedLambdaSchedule(
        hidden=int(cfg.get("hidden", 64)),
        freq_dim=int(cfg.get("freq_dim", 64)),
        lam_max=float(cfg.get("lam_max", 5.0)),
        zero_init=False,
    )
    m.load_state_dict(state["lambda_schedule"])
    return m.to(device).eval()


def load_posthoc_b(plain_backbone: FoundationScore, b_ckpt: Path, use_ema: bool, device) -> PostHocBVAwareScore:
    """加载 PostHoc-B ckpt + 套到 plain backbone (frozen)."""
    state = torch.load(str(b_ckpt), map_location=device, weights_only=False)
    cfg = state["config"]["model"]
    in_channels = int(cfg.get("in_channels", 2))
    phi_sh_dim = int(cfg.get("phi_sh_dim", 32))
    kappa_dim = int(cfg.get("kappa_dim", 16))
    depth = int(cfg.get("depth", 2))
    m = PostHocBVAwareScore(
        plain_backbone=plain_backbone, in_channels=in_channels,
        phi_sh_dim=phi_sh_dim, kappa_dim=kappa_dim, depth=depth,
    )
    if use_ema and "phi_sh_net_ema" in state:
        m.phi_sh_net.load_state_dict(state["phi_sh_net_ema"])
        m.kappa_net.load_state_dict(state["kappa_net_ema"])
    else:
        m.phi_sh_net.load_state_dict(state["phi_sh_net"])
        m.kappa_net.load_state_dict(state["kappa_net"])
    return m.to(device).eval()


def sample_with_method(method: str, plain_backbone, posthoc_module, dataset_cfg,
                       n: int, Nx: int, ic_b, pid_t, num_steps: int, device,
                       bv_strength: float = 1.0):
    """根据 method 名分派采样器."""
    common = dict(
        shape=(n, 1, Nx), sigma_min=0.01, sigma_max=10.0,
        tau_max=1.0, nu=1.0, num_steps=num_steps,
        device=device, zeta_pde=0.0,
        conditioning=ic_b, pde_id=pid_t,
        flux_type=dataset_cfg["flux_type"],
    )
    if method == "Plain":
        return entrodiff_heun_sampler(plain_backbone, **common)
    elif method == "PostHoc-A v1":
        return posthoc_bv_heun_sampler(plain_backbone, bv_strength=bv_strength,
                                       lambda_mode="exp_decay", **common)
    elif method == "PostHoc-A v2 (LearnedLambda)":
        return posthoc_bv_heun_sampler_v2(plain_backbone, bv_strength=bv_strength,
                                          lambda_mode="exp_decay",
                                          lambda_module=posthoc_module,
                                          enable_grad_for_lambda_module=False,
                                          **common)
    elif method.startswith("PostHoc-B"):
        # B 直接用 PostHocBVAwareScore 当 model
        return entrodiff_heun_sampler(posthoc_module, **common)
    else:
        raise ValueError(f"未知 method: {method}")


def evaluate_method_seeds(method: str,
                          plain_backbone, posthoc_modules_per_seed: list,
                          datasets: list, seeds: list, n_samples: int, num_steps: int,
                          device, bv_strength: float = 1.0) -> dict:
    """
    method × 5 seed × 5 setting 评估.

    posthoc_modules_per_seed: list of len(seeds), 每元素是 PostHocBVAwareScore 或 LearnedLambdaSchedule 或 None (Plain)
        - Plain: posthoc_modules_per_seed = [None] * len(seeds), seed 仅控 noise
        - PostHoc-A v1: 同上 (training-free, no per-seed module)
        - PostHoc-A v2: posthoc_modules_per_seed[i] = LearnedLambda for seed i
        - PostHoc-B: posthoc_modules_per_seed[i] = PostHocBVAwareScore for seed i
    """
    out_per_setting = {}
    for ds in datasets:
        per_seed = []
        for s_i, seed in enumerate(seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            data_path = env.data_dir / ds["data_file"]
            data = load_test_data(data_path, n_samples, ds.get("test_split_only", False), seed)
            ic_b = data["ic"].to(device)
            gt_b = data["x_target"].to(device)
            n = ic_b.shape[0]
            Nx = ic_b.shape[-1]
            x_grid = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
            pid_t = torch.full((n,), ds["pde_id"], dtype=torch.long, device=device)

            # 选 module
            mod = posthoc_modules_per_seed[s_i] if posthoc_modules_per_seed else None

            try:
                gen = sample_with_method(method, plain_backbone, mod, ds, n, Nx,
                                         ic_b, pid_t, num_steps, device, bv_strength=bv_strength)
            except Exception as e:
                print(f"      [error] seed={seed}: {e}")
                continue

            gen_np = gen.detach().cpu().numpy()
            gt_np = gt_b.detach().cpu().numpy()
            metrics = [compute_metrics(gt_np[i, 0], gen_np[i, 0], x_grid) for i in range(n)]
            per_seed.append({
                "seed": seed,
                "W1": float(np.mean([m["W1"] for m in metrics])),
                "L1_rel": float(np.mean([m["L1_rel"] for m in metrics])),
                "shock_err": float(np.mean([m["shock_err"] for m in metrics])),
            })
        if per_seed:
            keys = ["W1", "L1_rel", "shock_err"]
            agg = {k + "_mean": float(np.mean([r[k] for r in per_seed])) for k in keys}
            agg.update({k + "_std": float(np.std([r[k] for r in per_seed])) for k in keys})
            agg["per_seed"] = per_seed
            agg["n_seeds"] = len(per_seed)
            out_per_setting[ds["name"]] = agg
            print(f"    {ds['name']}: W1={agg['W1_mean']:.4f}±{agg['W1_std']:.4f} (n_seed={len(per_seed)})")
    return out_per_setting


def auto_discover(seeds: list[int]) -> dict:
    """自动从 env.output_dir 找全部 ckpt."""
    base = env.output_dir

    def find_ckpt(exp_prefix: str, seed: int, ep: int):
        # seed=42 没 _s42 后缀
        suffix = "" if seed == 42 else f"_s{seed}"
        d = base / (exp_prefix + suffix)
        if not d.exists():
            return None
        # 找 ep<N>.pt
        pat = list(d.glob(f"*_ep{ep}.pt"))
        return pat[0] if pat else None

    found = {
        "plain": base / "foundation_small_plain" /
                 "foundation_foundation_small_plain_20260503_024957_ep200.pt",
        "lambda": [find_ckpt("foundation_posthoc_a_v2", s, 50) for s in seeds],
        "bsmall": [find_ckpt("foundation_posthoc_b", s, 50) for s in seeds],
        "blarge": [find_ckpt("foundation_posthoc_b_large", s, 80) for s in seeds],
        "baug":   [find_ckpt("foundation_posthoc_b_aug", s, 80) for s in seeds],
    }
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true",
                        help="auto 模式从 env.output_dir 找 ckpt")
    parser.add_argument("--plain_ckpt", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 1042, 2042, 3042, 4042])
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--include_v1", action="store_true",
                        help="也评估 PostHoc-A v1 (固定 exp_decay, training-free)")
    parser.add_argument("--bv_strength_v1", type=float, default=1.0,
                        help="PostHoc-A v1 的 bv_strength (默认 1.0)")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(env.default_device)
    seeds = args.seeds
    print(f"[boost_full] device={device} num_steps={args.num_steps} n_samples={args.n_samples}")
    print(f"[boost_full] seeds: {seeds}")

    out_dir = Path(args.out_dir) if args.out_dir else env.output_dir / "boost_full_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 找 ckpt ----
    if args.auto:
        found = auto_discover(seeds)
    else:
        if not args.plain_ckpt:
            parser.error("--plain_ckpt 必须 (除非 --auto)")
        found = {"plain": Path(args.plain_ckpt)}
        # 其他自动找 (TODO)
        raise NotImplementedError("仅支持 --auto 模式; 请加 --auto")

    # 检查
    plain_path = Path(found["plain"])
    if not plain_path.exists():
        raise FileNotFoundError(f"Plain ckpt 不存在: {plain_path}")
    print(f"\n[plain] {plain_path.name}")

    methods_present = ["Plain"]
    if args.include_v1:
        methods_present.append("PostHoc-A v1")
    for key, label in [("lambda", "PostHoc-A v2 (LearnedLambda)"),
                       ("bsmall", "PostHoc-B small"),
                       ("blarge", "PostHoc-B large"),
                       ("baug",   "PostHoc-B aug")]:
        ck_list = found.get(key, [])
        valid = [c for c in ck_list if c and Path(c).exists()]
        if len(valid) == len(seeds):
            methods_present.append(label)
            print(f"[{key}] 全 {len(seeds)} seed ckpt 就绪")
        else:
            print(f"[{key}] 仅 {len(valid)}/{len(seeds)} seed ckpt 存在, 跳过 {label}")
    methods_present.extend(["PostHoc-B small (EMA)", "PostHoc-B large (EMA)", "PostHoc-B aug (EMA)"])
    # EMA 版本由相同 ckpt 加载, 用 use_ema 选择 — 仅当原方法在 methods_present 时才加 EMA 版

    # ---- 加载 plain backbone ----
    plain_backbone = load_plain_backbone(plain_path, n_pde_types=2, device=device)

    # ---- 评估每方法 ----
    all_results = {}
    for method in methods_present:
        print(f"\n=== {method} ===")
        try:
            if method == "Plain":
                modules = [None] * len(seeds)   # Plain 不需 module
                res = evaluate_method_seeds(method, plain_backbone, modules, DATASETS, seeds,
                                             args.n_samples, args.num_steps, device)
            elif method == "PostHoc-A v1":
                modules = [None] * len(seeds)   # v1 training-free
                res = evaluate_method_seeds(method, plain_backbone, modules, DATASETS, seeds,
                                             args.n_samples, args.num_steps, device,
                                             bv_strength=args.bv_strength_v1)
            elif method == "PostHoc-A v2 (LearnedLambda)":
                modules = [load_lambda_module(c, device) for c in found["lambda"]]
                res = evaluate_method_seeds(method, plain_backbone, modules, DATASETS, seeds,
                                             args.n_samples, args.num_steps, device)
            elif method == "PostHoc-B small":
                modules = [load_posthoc_b(plain_backbone, c, use_ema=False, device=device) for c in found["bsmall"]]
                res = evaluate_method_seeds(method, plain_backbone, modules, DATASETS, seeds,
                                             args.n_samples, args.num_steps, device)
            elif method == "PostHoc-B small (EMA)":
                modules = [load_posthoc_b(plain_backbone, c, use_ema=True, device=device) for c in found["bsmall"]]
                res = evaluate_method_seeds(method, plain_backbone, modules, DATASETS, seeds,
                                             args.n_samples, args.num_steps, device)
            elif method == "PostHoc-B large":
                modules = [load_posthoc_b(plain_backbone, c, use_ema=False, device=device) for c in found["blarge"]]
                res = evaluate_method_seeds(method, plain_backbone, modules, DATASETS, seeds,
                                             args.n_samples, args.num_steps, device)
            elif method == "PostHoc-B large (EMA)":
                modules = [load_posthoc_b(plain_backbone, c, use_ema=True, device=device) for c in found["blarge"]]
                res = evaluate_method_seeds(method, plain_backbone, modules, DATASETS, seeds,
                                             args.n_samples, args.num_steps, device)
            elif method == "PostHoc-B aug":
                modules = [load_posthoc_b(plain_backbone, c, use_ema=False, device=device) for c in found["baug"]]
                res = evaluate_method_seeds(method, plain_backbone, modules, DATASETS, seeds,
                                             args.n_samples, args.num_steps, device)
            elif method == "PostHoc-B aug (EMA)":
                modules = [load_posthoc_b(plain_backbone, c, use_ema=True, device=device) for c in found["baug"]]
                res = evaluate_method_seeds(method, plain_backbone, modules, DATASETS, seeds,
                                             args.n_samples, args.num_steps, device)
            else:
                continue
            all_results[method] = res
        except Exception as e:
            print(f"  [error] method {method} 整体失败: {e}")
            import traceback; traceback.print_exc()
            continue

    # ---- 主表 markdown (paired t-test vs Plain) ----
    plain_res = all_results.get("Plain", {})
    md = ["# Foundation Model 全方法 5-seed 主表 (论文 §5.6)\n",
          f"- num_steps: {args.num_steps}, n_samples: {args.n_samples}, n_seeds: {len(seeds)}",
          f"- seeds: {seeds}\n",
          "## W₁ ↓ (mean ± std, paired t-test vs Plain, ★ = p<0.05)\n",
          "| Method | " + " | ".join(d["name"] for d in DATASETS) + " |",
          "|" + "---|" * (len(DATASETS) + 1)]

    for method in all_results:
        row = [method]
        for ds in DATASETS:
            r = all_results[method].get(ds["name"])
            pl_r = plain_res.get(ds["name"]) if method != "Plain" else None
            if r is None:
                row.append("---")
                continue
            cell = f"{r['W1_mean']:.4f}±{r['W1_std']:.4f}"
            if pl_r is not None:
                # paired t-test
                arr_m = np.array([s["W1"] for s in r["per_seed"]])
                arr_p = np.array([s["W1"] for s in pl_r["per_seed"]])
                if len(arr_m) == len(arr_p) and len(arr_m) > 1:
                    tstat, pval = ttest_rel(arr_m, arr_p)
                    sig_tag = "★" if pval < 0.05 else ""
                    delta = (pl_r["W1_mean"] - r["W1_mean"]) / pl_r["W1_mean"] * 100
                    cell += f" ({delta:+.1f}%{sig_tag})"
            row.append(cell)
        md.append("| " + " | ".join(row) + " |")

    md.append("\n## L¹ ↓ (mean ± std)\n")
    md.append("| Method | " + " | ".join(d["name"] for d in DATASETS) + " |")
    md.append("|" + "---|" * (len(DATASETS) + 1))
    for method in all_results:
        row = [method]
        for ds in DATASETS:
            r = all_results[method].get(ds["name"])
            row.append(f"{r['L1_rel_mean']:.4f}±{r['L1_rel_std']:.4f}" if r else "---")
        md.append("| " + " | ".join(row) + " |")

    md.append("\n## shock_err ↓ (mean ± std)\n")
    md.append("| Method | " + " | ".join(d["name"] for d in DATASETS) + " |")
    md.append("|" + "---|" * (len(DATASETS) + 1))
    for method in all_results:
        row = [method]
        for ds in DATASETS:
            r = all_results[method].get(ds["name"])
            row.append(f"{r['shock_err_mean']:.4f}±{r['shock_err_std']:.4f}" if r else "---")
        md.append("| " + " | ".join(row) + " |")

    md_text = "\n".join(md)
    md_path = out_dir / "summary_boost_full.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    json_path = out_dir / "summary_boost_full.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(md_text)
    print("=" * 80)
    print(f"\n[md] {md_path}\n[json] {json_path}")


if __name__ == "__main__":
    main()
