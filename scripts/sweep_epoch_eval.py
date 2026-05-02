"""
epoch sweet spot sweep (W5-SA4).

给定一个 ckpt 目录 (含 _ep{N}.pt 多个 epoch checkpoint), eval 各 epoch
下的 W₁ 曲线.

用法:
  python scripts/sweep_epoch_eval.py --ckpt_dir output/experiments/bvaware_run \\
                                      --config configs/experiment/bvaware_server.yaml \\
                                      --epochs 50 100 150 200 \\
                                      --num_steps 50 \\
                                      --output sweep_epochs.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import wasserstein_distance

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.data.burgers_dataset import BurgersDataset
from src.models.score_param import BVAwareScore, StandardScore
from src.diffusion.samplers import entrodiff_heun_sampler


def find_ckpt_for_epoch(ckpt_dir: Path, epoch: int) -> Path | None:
    """在 ckpt_dir 下找 _ep{epoch}.pt 的 ckpt."""
    pattern = str(ckpt_dir / f"*_ep{epoch}.pt")
    matches = glob.glob(pattern)
    if not matches:
        return None
    # 多个 timestamp 时取最新
    return Path(sorted(matches)[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output", type=str, default="sweep_epochs.csv")
    parser.add_argument(
        "--model_type", type=str, default="bvaware",
        choices=["standard", "bvaware"],
    )
    parser.add_argument("--in_channels", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(env.default_device)
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["experiment"]

    nu = float(cfg.get("nu", 1.0))
    tau_max = float(cfg.get("tau_max", 1.0))
    data_path = env.data_dir / cfg.get("data_file", "burgers_1d_N5000_Nx128.npy")
    if not data_path.exists():
        sys.exit(f"数据不存在: {data_path}")

    # test split + IC
    test_ds = BurgersDataset(data_path, mode="test", conditioning_type="ic")
    samples = [test_ds[i] for i in range(min(args.n_samples, len(test_ds)))]
    batch = torch.stack(samples)
    gt = batch[:, -1, :].numpy()
    ic = batch[:, 0, :].unsqueeze(1).to(device)
    nx = gt.shape[-1]

    rows = []
    ckpt_dir = Path(args.ckpt_dir)
    for ep in args.epochs:
        ckpt_path = find_ckpt_for_epoch(ckpt_dir, ep)
        if ckpt_path is None:
            print(f"[sweep_epoch] ep={ep}: ckpt 未找到, 跳过")
            rows.append({
                "epoch": ep, "ckpt": "missing",
                "W1_mean": float("nan"), "W1_std": float("nan"),
            })
            continue
        print(f"[sweep_epoch] ep={ep} ← {ckpt_path.name}")
        try:
            if args.model_type == "standard":
                model = StandardScore(in_channels=args.in_channels)
            else:
                model = BVAwareScore(in_channels=args.in_channels, dim=args.dim)
            model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
            model = model.to(device).eval()
            gen = entrodiff_heun_sampler(
                model, shape=(args.n_samples, 1, nx),
                sigma_min=1e-3, sigma_max=math.sqrt(2 * nu * tau_max),
                tau_max=tau_max, nu=nu, num_steps=args.num_steps,
                device=device, zeta_pde=0.0, conditioning=ic,
            ).squeeze(1).detach().cpu().numpy()
            w1_per = [wasserstein_distance(gt[i], gen[i]) for i in range(len(gt))]
            w1_mean = float(np.mean(w1_per))
            w1_std = float(np.std(w1_per))
            print(f"  W1 = {w1_mean:.4f} ± {w1_std:.4f}")
            rows.append({
                "epoch": ep, "ckpt": ckpt_path.name,
                "W1_mean": w1_mean, "W1_std": w1_std,
            })
        except Exception as e:
            print(f"  [error] {e}")
            rows.append({
                "epoch": ep, "ckpt": ckpt_path.name,
                "W1_mean": float("nan"), "W1_std": float("nan"),
            })

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "ckpt", "W1_mean", "W1_std"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[sweep_epoch] csv → {out_path}")


if __name__ == "__main__":
    main()
