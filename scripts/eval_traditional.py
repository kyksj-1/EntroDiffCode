# ============================================================================
# eval_traditional.py — 传统数值方法 baseline 评估
# 对比 EntroDiff vs Lax-Friedrichs / MacCormack / Central-Viscosity
# ============================================================================
import sys, os, math, argparse
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.env_manager import env
from src.data.burgers_1d_solver import burgers_godunov_1d
from src.data.traditional_solvers import (
    burgers_lax_friedrichs_1d,
    burgers_maccormack_1d,
    burgers_central_viscosity_1d,
)

def compute_metrics(u_pred, u_gt, x_grid):
    """计算三项指标: W₁, L¹ 相对误差, shock-location 误差"""
    w1 = wasserstein_distance(u_gt, u_pred)
    l1 = np.linalg.norm(u_pred - u_gt, 1) / (np.linalg.norm(u_gt, 1) + 1e-10)
    grad_gt = np.abs(np.gradient(u_gt, x_grid))
    grad_pr = np.abs(np.gradient(u_pred, x_grid))
    shock_err = abs(x_grid[np.argmax(grad_pr)] - x_grid[np.argmax(grad_gt)])
    return w1, l1, shock_err

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--nt", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.005)
    args = parser.parse_args()

    nx, nt, dt, n_test = args.nx, args.nt, args.dt, args.n_samples
    dx = 2 * math.pi / nx
    x_grid = np.linspace(0, 2*math.pi, nx, endpoint=False)

    # 生成与训练数据同分布的 IC (标准 Fourier)
    np.random.seed(42)
    results = []  # list of (method_name, [w1_list], [l1_list], [shock_list])

    print(f"Evaluating {n_test} test samples on {nx} grid...\n")

    # 定义求解器
    solvers = [
        ("Godunov (GT)", lambda u0: burgers_godunov_1d(u0, nx, nt, dx, dt)),
        ("Lax-Friedrichs", lambda u0: burgers_lax_friedrichs_1d(u0, nx, nt, dx, dt)),
        ("MacCormack",    lambda u0: burgers_maccormack_1d(u0, nx, nt, dx, dt)),
        ("Central+ν=0.01", lambda u0: burgers_central_viscosity_1d(u0, nx, nt, dx, dt, nu_art=0.01)),
        ("Central+ν=0.001",lambda u0: burgers_central_viscosity_1d(u0, nx, nt, dx, dt, nu_art=0.001)),
        ("Central+ν=0",   lambda u0: burgers_central_viscosity_1d(u0, nx, nt, dx, dt, nu_art=0.0)),
    ]

    for name, solver in solvers:
        w1s, l1s, shocks = [], [], []
        for s in range(n_test):
            # 生成随机 IC (同训练数据分布)
            k_max = 5
            A = np.random.randn(k_max) / (np.arange(1, k_max+1)**2)
            B = np.random.randn(k_max) / (np.arange(1, k_max+1)**2)
            u0 = np.zeros(nx)
            for k in range(1, k_max+1):
                u0 += A[k-1] * np.sin(k*x_grid) + B[k-1] * np.cos(k*x_grid)
            # CFL 裁剪
            cfl = np.max(np.abs(u0)) * dt / dx
            if cfl > 0.9:
                u0 *= 0.9 / cfl

            u_hist = solver(u0)
            u_pred = u_hist[-1]  # 终端时刻解

            # 用 Godunov 作为 ground truth
            u_gt = burgers_godunov_1d(u0, nx, nt, dx, dt)[-1]

            w1, l1, se = compute_metrics(u_pred, u_gt, x_grid)
            w1s.append(w1)
            l1s.append(l1)
            shocks.append(se)

            if s % 10 == 0:
                print(f"  {name}: {s+1}/{n_test} W₁={np.mean(w1s):.4f}", flush=True)

        results.append((name, np.mean(w1s), np.std(w1s), np.mean(l1s), np.mean(shocks)))
        print(f"  {name}: W₁={np.mean(w1s):.4f} ± {np.std(w1s):.4f}  L¹={np.mean(l1s):.4f}  Shock={np.mean(shocks):.3f}")

    # 汇总表
    print("\n" + "="*70)
    print("  Traditional Solver Baselines (vs Godunov GT)")
    print("="*70)
    print(f"  {'Method':<20} {'W₁ avg':<12} {'W₁ std':<10} {'L¹ avg':<10} {'Shock err'}")
    print("  " + "-"*65)
    for name, w1, w1s, l1, se in results:
        print(f"  {name:<20} {w1:<12.4f} {w1s:<10.4f} {l1:<10.4f} {se:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()
