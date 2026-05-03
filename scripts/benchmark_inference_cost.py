# ============================================================================
# Inference Cost Benchmark — 5 方法对比采样时间
#
# 论文 §5.6 关键量化: PostHoc 是 lightweight (vs 重训 BVA 的 2-3x slowdown)
#
# 测的方法:
#   1. Plain DiT (FoundationScore)              — baseline
#   2. PostHoc-A (Plain + posthoc_bv_sampler)   — training-free
#   3. PostHoc-B small (386 trainable)
#   4. PostHoc-B large (7K trainable)
#   5. BVA Small (重训, 含 create_graph=True 二阶导)
#   6. BVA Big (重训 57.9M, 含二阶导)
#
# 测时方法:
#   - torch.cuda.Event(enable_timing=True) 包围采样调用
#   - num_steps=25, batch=16
#   - warm-up 1 次 + 计时 5 次取均值/std
#   - 同一 IC + 同一 noise (seed=42) 保证公平
# ============================================================================

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.diffusion.samplers import entrodiff_heun_sampler
from src.diffusion.posthoc_bv_sampler import posthoc_bv_heun_sampler
from src.models.foundation_score import FoundationScore
from src.models.posthoc_bv_score import PostHocBVAwareScore
from scripts.eval_foundation import build_model_from_ckpt_cfg


# 默认 ckpt 路径
CKPTS = {
    "Plain":             "output/experiments/foundation_small_plain/foundation_foundation_small_plain_20260503_024957_ep200.pt",
    "BVA_Small_retrain": "output/experiments/foundation_small/foundation_foundation_small_20260503_024957_ep200.pt",
    "BVA_Big_retrain":   "output/experiments/foundation_big/foundation_foundation_big_20260503_024957_ep200.pt",
    "PostHoc_B_small":   "output/experiments/foundation_posthoc_b/posthoc_b_foundation_posthoc_b_20260503_193441_ep50.pt",
    "PostHoc_B_large":   "output/experiments/foundation_posthoc_b_large/posthoc_b_foundation_posthoc_b_large_20260503_200031_ep80.pt",
}


def load_plain_or_bva(ckpt_path: Path, n_pde_types: int, device) -> torch.nn.Module:
    """加载 dit_plain / dit_bvaware ckpt → model."""
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = state["config"]
    model = build_model_from_ckpt_cfg(cfg, n_pde_types=n_pde_types, device=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def load_posthoc_b(posthoc_b_ckpt: Path, n_pde_types: int, device) -> PostHocBVAwareScore:
    """加载 PostHoc-B ckpt + plain backbone → PostHocBVAwareScore."""
    state = torch.load(str(posthoc_b_ckpt), map_location=device, weights_only=False)
    plain_path = Path(state["plain_ckpt_path"])
    if not plain_path.exists():
        plain_path = PROJECT_ROOT / CKPTS["Plain"]
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

    posthoc_b_cfg = state["config"]["model"]
    phi_sh_dim = int(posthoc_b_cfg.get("phi_sh_dim", 32))
    kappa_dim = int(posthoc_b_cfg.get("kappa_dim", 16))
    depth = int(posthoc_b_cfg.get("depth", 2))

    model = PostHocBVAwareScore(
        plain_backbone=plain_backbone, in_channels=in_channels,
        phi_sh_dim=phi_sh_dim, kappa_dim=kappa_dim, depth=depth,
    )
    model.phi_sh_net.load_state_dict(state["phi_sh_net"])
    model.kappa_net.load_state_dict(state["kappa_net"])
    model = model.to(device)
    model.eval()
    return model


def time_sampling(sampler_fn, model, sampler_kwargs, n_warmup=1, n_repeat=5,
                  device=None) -> dict:
    """
    测 sampler 调用的 GPU 时间.

    Args:
        sampler_fn:    sampler 函数 (entrodiff_heun_sampler 或 posthoc_bv_heun_sampler)
        model:         传给 sampler 的 model
        sampler_kwargs: 其他参数 (shape, sigma_min, ..., conditioning, pde_id, ...)
        n_warmup:      warm-up 次数 (不计时)
        n_repeat:      正式计时次数

    Returns:
        dict: {mean_ms, std_ms, all_times_ms}
    """
    if device is None:
        device = torch.device(env.default_device)

    if device.type != "cuda":
        # CPU fallback: 用 time.perf_counter
        import time
        all_times = []
        # warm-up
        for _ in range(n_warmup):
            _ = sampler_fn(model, **sampler_kwargs)
        # timed
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            _ = sampler_fn(model, **sampler_kwargs)
            all_times.append((time.perf_counter() - t0) * 1000.0)
        return {
            "mean_ms": float(np.mean(all_times)),
            "std_ms": float(np.std(all_times)),
            "all_times_ms": all_times,
        }

    # CUDA timing
    all_times = []
    # warm-up (不计时, 但要触发 cudnn benchmark / kernel 选择)
    for _ in range(n_warmup):
        torch.manual_seed(42)
        _ = sampler_fn(model, **sampler_kwargs)
        torch.cuda.synchronize()

    # timed
    for trial in range(n_repeat):
        torch.manual_seed(42)   # 同 seed 公平
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        _ = sampler_fn(model, **sampler_kwargs)
        ender.record()
        torch.cuda.synchronize()
        all_times.append(starter.elapsed_time(ender))

    return {
        "mean_ms": float(np.mean(all_times)),
        "std_ms": float(np.std(all_times)),
        "all_times_ms": all_times,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_repeat", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(env.default_device)
    print(f"[benchmark] device={device}  num_steps={args.num_steps}  batch={args.batch_size}  n_repeat={args.n_repeat}")

    # 用 in-dist Burgers 第 1 个样本作 IC
    burgers_path = env.data_dir / "burgers_1d_N5000_Nx128.npy"
    data = np.load(str(burgers_path))[:1]
    ic_single = torch.tensor(data[:, 0, :], dtype=torch.float32).unsqueeze(1).to(device)  # (1, 1, Nx)
    Nx = ic_single.shape[-1]

    # 复制 batch 份
    ic_batch = ic_single.expand(args.batch_size, -1, -1).contiguous()
    pde_id = torch.zeros(args.batch_size, dtype=torch.long, device=device)

    common_kwargs = dict(
        shape=(args.batch_size, 1, Nx),
        sigma_min=0.01, sigma_max=10.0,
        tau_max=1.0, nu=1.0,
        num_steps=args.num_steps,
        device=device, zeta_pde=0.0,
        conditioning=ic_batch, pde_id=pde_id,
        flux_type="burgers",
    )

    posthoc_a_kwargs = dict(common_kwargs)
    posthoc_a_kwargs.update(bv_strength=1.0, lambda_mode="exp_decay")

    # ---- 加载 5 个模型 ----
    out_dir = Path(args.out_dir) if args.out_dir else env.output_dir / "inference_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []   # list of dict: {name, params, time_ms_mean, time_ms_std, sampler_type}

    # 1) Plain
    print("\n[1/6] Plain DiT...")
    plain_path = (PROJECT_ROOT / CKPTS["Plain"]).resolve()
    plain_model = load_plain_or_bva(plain_path, n_pde_types=2, device=device)
    n_params = sum(p.numel() for p in plain_model.parameters())
    timing = time_sampling(entrodiff_heun_sampler, plain_model, common_kwargs,
                           n_warmup=1, n_repeat=args.n_repeat, device=device)
    results.append({"name": "Plain", "params": n_params, "trainable": n_params,
                    "time_ms_mean": timing["mean_ms"], "time_ms_std": timing["std_ms"],
                    "sampler": "entrodiff_heun_sampler"})
    print(f"  params={n_params/1e6:.2f}M  time={timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")

    # 2) PostHoc-A (复用 Plain ckpt + posthoc sampler)
    print("\n[2/6] PostHoc-A (training-free)...")
    timing = time_sampling(posthoc_bv_heun_sampler, plain_model, posthoc_a_kwargs,
                           n_warmup=1, n_repeat=args.n_repeat, device=device)
    results.append({"name": "PostHoc-A (training-free)", "params": n_params, "trainable": 0,
                    "time_ms_mean": timing["mean_ms"], "time_ms_std": timing["std_ms"],
                    "sampler": "posthoc_bv_heun_sampler"})
    print(f"  params={n_params/1e6:.2f}M (no extra trainable)  time={timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")
    del plain_model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # 3) PostHoc-B small
    print("\n[3/6] PostHoc-B small (386 params)...")
    pb_small_path = (PROJECT_ROOT / CKPTS["PostHoc_B_small"]).resolve()
    if not pb_small_path.exists():
        print(f"  [skip] {pb_small_path} not found")
    else:
        pb_small_model = load_posthoc_b(pb_small_path, n_pde_types=2, device=device)
        n_total = sum(p.numel() for p in pb_small_model.parameters())
        n_trainable = sum(p.numel() for p in pb_small_model.trainable_parameters())
        timing = time_sampling(entrodiff_heun_sampler, pb_small_model, common_kwargs,
                               n_warmup=1, n_repeat=args.n_repeat, device=device)
        results.append({"name": "PostHoc-B small", "params": n_total, "trainable": n_trainable,
                        "time_ms_mean": timing["mean_ms"], "time_ms_std": timing["std_ms"],
                        "sampler": "entrodiff_heun_sampler"})
        print(f"  total={n_total/1e6:.2f}M  trainable={n_trainable}  time={timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")
        del pb_small_model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # 4) PostHoc-B large
    print("\n[4/6] PostHoc-B large (~7K params)...")
    pb_large_path = (PROJECT_ROOT / CKPTS["PostHoc_B_large"]).resolve()
    if not pb_large_path.exists():
        print(f"  [skip] {pb_large_path} not found")
    else:
        pb_large_model = load_posthoc_b(pb_large_path, n_pde_types=2, device=device)
        n_total = sum(p.numel() for p in pb_large_model.parameters())
        n_trainable = sum(p.numel() for p in pb_large_model.trainable_parameters())
        timing = time_sampling(entrodiff_heun_sampler, pb_large_model, common_kwargs,
                               n_warmup=1, n_repeat=args.n_repeat, device=device)
        results.append({"name": "PostHoc-B large", "params": n_total, "trainable": n_trainable,
                        "time_ms_mean": timing["mean_ms"], "time_ms_std": timing["std_ms"],
                        "sampler": "entrodiff_heun_sampler"})
        print(f"  total={n_total/1e6:.2f}M  trainable={n_trainable}  time={timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")
        del pb_large_model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # 5) BVA Small (retrained, 含二阶导)
    print("\n[5/6] BVA Small (retrained with 2nd-order autograd)...")
    bva_s_path = (PROJECT_ROOT / CKPTS["BVA_Small_retrain"]).resolve()
    bva_s = load_plain_or_bva(bva_s_path, n_pde_types=2, device=device)
    n_total = sum(p.numel() for p in bva_s.parameters())
    timing = time_sampling(entrodiff_heun_sampler, bva_s, common_kwargs,
                           n_warmup=1, n_repeat=args.n_repeat, device=device)
    results.append({"name": "BVA Small (retrained)", "params": n_total, "trainable": n_total,
                    "time_ms_mean": timing["mean_ms"], "time_ms_std": timing["std_ms"],
                    "sampler": "entrodiff_heun_sampler"})
    print(f"  params={n_total/1e6:.2f}M (含二阶导)  time={timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")
    del bva_s
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # 6) BVA Big
    print("\n[6/6] BVA Big (retrained, 57.9M)...")
    bva_b_path = (PROJECT_ROOT / CKPTS["BVA_Big_retrain"]).resolve()
    bva_b = load_plain_or_bva(bva_b_path, n_pde_types=2, device=device)
    n_total = sum(p.numel() for p in bva_b.parameters())
    timing = time_sampling(entrodiff_heun_sampler, bva_b, common_kwargs,
                           n_warmup=1, n_repeat=args.n_repeat, device=device)
    results.append({"name": "BVA Big (retrained)", "params": n_total, "trainable": n_total,
                    "time_ms_mean": timing["mean_ms"], "time_ms_std": timing["std_ms"],
                    "sampler": "entrodiff_heun_sampler"})
    print(f"  params={n_total/1e6:.2f}M (含二阶导)  time={timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")

    # ---- 计算 overhead vs Plain ----
    plain_time = next(r["time_ms_mean"] for r in results if r["name"] == "Plain")
    for r in results:
        r["overhead_vs_plain"] = (r["time_ms_mean"] - plain_time) / plain_time * 100.0
        r["overhead_factor"] = r["time_ms_mean"] / plain_time
        r["per_step_ms"] = r["time_ms_mean"] / args.num_steps
        r["per_sample_ms"] = r["time_ms_mean"] / args.batch_size

    # ---- 输出 markdown 表 ----
    md = ["# Inference Cost Benchmark — Foundation Model 5 方法对比\n",
          f"- num_steps: {args.num_steps}",
          f"- batch_size: {args.batch_size}",
          f"- n_repeat (取均值): {args.n_repeat}",
          f"- device: {device}",
          f"- timing: torch.cuda.Event\n",
          "## 主表 (W₁ 优势 + Inference cost)\n",
          "| Method | Total params | Trainable | Time per batch (ms) | Per-step (ms) | Per-sample (ms) | Overhead vs Plain |",
          "|---|---|---|---|---|---|---|"]

    for r in results:
        md.append(
            f"| {r['name']} | {r['params']/1e6:.2f}M | {r['trainable']:,} | "
            f"{r['time_ms_mean']:.1f} ± {r['time_ms_std']:.1f} | "
            f"{r['per_step_ms']:.2f} | {r['per_sample_ms']:.2f} | "
            f"{r['overhead_factor']:.2f}x ({r['overhead_vs_plain']:+.1f}%) |"
        )
    md.append("\n## 关键洞察\n")
    pa = next((r for r in results if "PostHoc-A" in r["name"]), None)
    pb_s = next((r for r in results if "PostHoc-B small" in r["name"]), None)
    bva_s = next((r for r in results if "BVA Small (retrained)" in r["name"]), None)
    if pa and bva_s:
        md.append(f"- **PostHoc-A overhead**: {pa['overhead_vs_plain']:+.1f}% vs Plain (training-free, 仅 O(Nx) shock 检测)")
        md.append(f"- **BVA retrained overhead**: {bva_s['overhead_vs_plain']:+.1f}% vs Plain (二阶导 create_graph=True 显著增加 forward 成本)")
        if bva_s["overhead_factor"] / max(pa["overhead_factor"], 1.001) > 1.5:
            ratio = bva_s["time_ms_mean"] / pa["time_ms_mean"]
            md.append(f"- **PostHoc-A 比 BVA 重训快 {ratio:.2f}x** — plug-and-play 直接证据\n")

    md_text = "\n".join(md)
    md_path = out_dir / "summary_inference_cost.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    json_path = out_dir / "summary_inference_cost.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(md_text)
    print("=" * 80)
    print(f"\n[md] {md_path}\n[json] {json_path}")


if __name__ == "__main__":
    main()
