# ============================================================================
# 方案 B 评估: PostHocBVAwareScore (训过的 phi_sh + kappa) vs Plain
#
# 论文 §5.6 终极对比:
#   - Plain (no posthoc):           已训 dit_plain 标准 sampler
#   - PostHoc-A (training-free):    已训 dit_plain + posthoc_bv_sampler
#   - PostHoc-B (trained correction): dit_plain + 训过的 phi_sh/kappa
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
from src.models.foundation_score import FoundationScore
from src.models.posthoc_bv_score import PostHocBVAwareScore
from scripts.eval_foundation import compute_metrics


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


def load_test_data(data_path: Path, n_samples: int, test_split_only: bool) -> dict:
    data = np.load(str(data_path))
    if test_split_only:
        idx_val_end = int(0.9 * data.shape[0])
        data = data[idx_val_end:]
    data = data[:n_samples]
    ic = torch.tensor(data[:, 0, :], dtype=torch.float32).unsqueeze(1)
    x_target = torch.tensor(data[:, -1, :], dtype=torch.float32).unsqueeze(1)
    return {"ic": ic, "x_target": x_target, "n": data.shape[0]}


def build_posthoc_b_model(posthoc_b_ckpt: Path, n_pde_types: int, device) -> PostHocBVAwareScore:
    """加载 posthoc_b ckpt + plain backbone, 重建 PostHocBVAwareScore."""
    state = torch.load(str(posthoc_b_ckpt), map_location=device, weights_only=False)
    plain_path = Path(state["plain_ckpt_path"])
    if not plain_path.exists():
        # 尝试相对当前 PROJECT_ROOT
        plain_path = PROJECT_ROOT / "output/experiments/foundation_small_plain" / plain_path.name
    print(f"[load] plain backbone: {plain_path.name}")

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

    model = PostHocBVAwareScore(plain_backbone=plain_backbone, in_channels=in_channels)
    model.phi_sh_net.load_state_dict(state["phi_sh_net"])
    model.kappa_net.load_state_dict(state["kappa_net"])
    model = model.to(device)
    model.eval()

    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"[load] PostHocBVAwareScore  trainable={n_train}  total={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model


def evaluate(model, dataset, n_samples=16, num_steps=25, device=None):
    if device is None:
        device = torch.device(env.default_device)

    data_path = env.data_dir / dataset["data_file"]
    data = load_test_data(data_path, n_samples=n_samples,
                          test_split_only=dataset.get("test_split_only", False))
    ic_b = data["ic"].to(device)
    gt_b = data["x_target"].to(device)
    n = ic_b.shape[0]
    Nx = ic_b.shape[-1]
    x_grid = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
    pid_t = torch.full((n,), dataset["pde_id"], dtype=torch.long, device=device)

    # 用现有的 entrodiff_heun_sampler — 它接受任何 model 接口为 (x, sigma, pde_id) → D_x
    # PostHocBVAwareScore 的 forward 与之兼容
    gen = entrodiff_heun_sampler(
        model, shape=(n, 1, Nx),
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
        "n_samples": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posthoc_b_ckpt", type=str,
                        default="output/experiments/foundation_posthoc_b/posthoc_b_foundation_posthoc_b_20260503_193441_ep50.pt")
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(env.default_device)
    print(f"[posthoc_b_eval] device={device}")

    out_dir = Path(args.out_dir) if args.out_dir else env.output_dir / "posthoc_b_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载 posthoc-B model
    ckpt_path = (PROJECT_ROOT / args.posthoc_b_ckpt).resolve()
    if not ckpt_path.exists():
        # try absolute
        ckpt_path = Path(args.posthoc_b_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"posthoc_b ckpt 不存在: {ckpt_path}")
    model = build_posthoc_b_model(ckpt_path, n_pde_types=2, device=device)

    # 评估
    results = {}
    for ds in DATASETS:
        print(f"\n  [{ds['name']}]")
        try:
            r = evaluate(model, ds, n_samples=args.n_samples, num_steps=args.num_steps, device=device)
            results[ds["name"]] = r
            print(f"    W₁={r['W1']:.4f}  L¹={r['L1_rel']:.4f}  shock={r['shock_err']:.4f}")
        except Exception as e:
            print(f"    [error] {e}")
            import traceback; traceback.print_exc()

    # markdown 输出
    md = ["# Post-hoc BV-aware (方案 B · trained phi_sh+kappa) Eval\n",
          f"- ckpt: {ckpt_path.name}",
          f"- num_steps: {args.num_steps}, n_samples: {args.n_samples}\n",
          "## W₁ ↓\n"]
    md.append("| Setting | PostHoc-B W₁ | L¹ | shock_err |")
    md.append("|---|---|---|---|")
    for name, r in results.items():
        md.append(f"| {name} | {r['W1']:.4f} | {r['L1_rel']:.4f} | {r['shock_err']:.4f} |")
    md_text = "\n".join(md)

    md_path = out_dir / "summary_posthoc_b.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    json_path = out_dir / "summary_posthoc_b.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"ckpt": str(ckpt_path), "results": results},
                  f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(md_text)
    print("=" * 80)
    print(f"[md] {md_path}\n[json] {json_path}")


if __name__ == "__main__":
    main()
