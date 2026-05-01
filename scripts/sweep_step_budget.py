"""
fine-grained step budget sweep (W5-SA4).

对一个 ckpt, eval 多个 Heun 步数下的 W₁ 曲线, 输出 csv 用于绘图.

用法:
  python scripts/sweep_step_budget.py --ckpt path/to/ckpt.pt \\
                                       --config configs/experiment/bvaware_server.yaml \\
                                       --steps 5 8 10 12 15 20 25 50 100 \\
                                       --n_samples 50 \\
                                       --output sweep_steps.csv
"""
from __future__ import annotations

import argparse
import csv
import math
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 10, 15, 25, 50, 100])
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output", type=str, default="sweep_steps.csv")
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

    # 数据 + IC
    test_ds = BurgersDataset(data_path, mode="test", conditioning_type="ic")
    samples = []
    for i in range(min(args.n_samples, len(test_ds))):
        samples.append(test_ds[i])
    batch = torch.stack(samples)   # (n, N_time, Nx)
    gt = batch[:, -1, :].numpy()                  # (n, Nx) 末帧
    ic = batch[:, 0, :].unsqueeze(1).to(device)   # (n, 1, Nx)
    nx = gt.shape[-1]
    x_grid = np.linspace(0, 2 * math.pi, nx, endpoint=False)

    # 模型
    if args.model_type == "standard":
        model = StandardScore(in_channels=args.in_channels)
    else:
        model = BVAwareScore(in_channels=args.in_channels, dim=args.dim)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model = model.to(device).eval()

    # sweep
    rows = []
    for n_steps in args.steps:
        print(f"[sweep_steps] n_steps={n_steps}")
        try:
            gen = entrodiff_heun_sampler(
                model,
                shape=(args.n_samples, 1, nx),
                sigma_min=1e-3, sigma_max=math.sqrt(2 * nu * tau_max),
                tau_max=tau_max, nu=nu,
                num_steps=n_steps,
                device=device, zeta_pde=0.0,
                conditioning=ic,
            ).squeeze(1).detach().cpu().numpy()
            w1_per = [wasserstein_distance(gt[i], gen[i]) for i in range(len(gt))]
            w1_mean = float(np.mean(w1_per))
            w1_std = float(np.std(w1_per))
            print(f"  W1 = {w1_mean:.4f} ± {w1_std:.4f}")
            rows.append({"n_steps": n_steps, "W1_mean": w1_mean, "W1_std": w1_std})
        except Exception as e:
            print(f"  [skip] {e}")
            rows.append({"n_steps": n_steps, "W1_mean": float("nan"), "W1_std": float("nan")})

    # 写 csv
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_steps", "W1_mean", "W1_std"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[sweep_steps] csv → {out_path}")


if __name__ == "__main__":
    main()
