# ============================================================================
# 2D 浅水方程 (Shallow-Water) 数据生成 — 圆形溃坝
# 论文对齐: E4 experiment
# 使用 Strang 维度分裂 + HLL Riemann 求解器
# 产生: 溃坝激波的二维传播模式, 输出水深 h 的时空轨迹
# ============================================================================
import sys, numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env
from src.pdes.shallow_water import (
    dam_break_ic,
    solve_shallow_water_2d,
    sw_max_wavespeed,
)
from tqdm import tqdm
from typing import Any


def generate_sw_data(
    n_samples: int = 1000,
    nx: int = 64,
    ny: int = 64,
    nt: int = 200,
    g: float = 9.81,
    cfl: float = 0.5,
) -> np.ndarray:
    """
    生成 2D 浅水方程数据集.

    初始条件: 圆形溃坝, 在不同参数 (半径 / 水深比 / 位置) 下采样.

    参数:
        n_samples: 样本总数
        nx, ny:   空间网格
        nt:       时间步数
        g:        重力加速度
        cfl:      CFL 数 (≤0.6, HLL 稳定性)
    返回:
        data_all: (n_samples, nt+1, nx, ny) float32 数组 — 水深轨迹
    """
    dx = 1.0 / nx
    dy = 1.0 / ny

    # — 参数组合: 溃坝参数范围 —
    param_configs = [
        # (label,         h_in, h_out, r0)
        ("classic",       2.0,  1.0,   0.15),
        ("deep break",    3.0,  0.5,   0.15),
        ("shallow break", 1.5,  1.0,   0.2),
        ("wide dam",      2.0,  1.0,   0.3),
        ("narrow dam",    2.5,  1.0,   0.08),
        ("mild jump",     1.3,  1.0,   0.15),
        ("strong jump",   4.0,  0.5,   0.12),
        ("offset center", 2.0,  1.0,   0.15),
    ]

    n_configs = len(param_configs)
    samples_per_config = n_samples // n_configs
    remainder = n_samples - samples_per_config * n_configs

    print(f"[E4] Generating {n_samples} 2D Shallow-Water trajectories...")
    print(f"  Grid: nx={nx}, ny={ny}, nt={nt}, dx={dx:.4f}, dy={dy:.4f}, cfl={cfl}")
    print(f"  Configs: {n_configs} × ~{samples_per_config}")

    data_all = np.zeros((n_samples, nt + 1, nx, ny), dtype=np.float32)

    sample_idx = 0
    pbar = tqdm(total=n_samples, desc="SW data")

    for cfg_idx, (label, h_in0, h_out0, r00) in enumerate(param_configs):
        n_for_cfg = samples_per_config + (1 if cfg_idx < remainder else 0)

        for _ in range(n_for_cfg):
            # 在基准值 ±15% 内随机扰动
            h_in = h_in0 * (1.0 + np.random.uniform(-0.15, 0.15))
            h_out = h_out0 * (1.0 + np.random.uniform(-0.15, 0.15))
            r0 = r00 * (1.0 + np.random.uniform(-0.15, 0.15))
            # 坝中心随机微移
            cx = 0.5 + np.random.uniform(-0.1, 0.1)
            cy = 0.5 + np.random.uniform(-0.1, 0.1)

            # 确保物理意义
            h_in = max(h_in, h_out + 0.1)  # 坝内水深高于坝外
            h_out = max(h_out, 0.1)
            r0 = np.clip(r0, 0.05, 0.4)

            # 构建 IC
            h0, u0, v0 = dam_break_ic(nx, ny, h_in=h_in, h_out=h_out, r0=r0, cx=cx, cy=cy)

            # CFL 确定 dt
            w_max = sw_max_wavespeed(h0, u0, v0, g)
            dt = cfl * min(dx, dy) / max(w_max, 1e-6)

            # 求解 (只返回 h 轨迹)
            h_hist = solve_shallow_water_2d(h0, u0, v0, nx, ny, nt, dt, dx, dy, g)
            data_all[sample_idx] = h_hist
            sample_idx += 1
            pbar.update(1)

    pbar.close()
    return data_all


if __name__ == "__main__":
    NX = 64
    NY = 64
    NT = 200
    G = 9.81
    CFL = 0.5
    N_SAMPLES = 1000

    output_path = env.data_dir / "sw_dam_N1000_Nx64.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_sw_data(n_samples=N_SAMPLES, nx=NX, ny=NY, nt=NT, g=G, cfl=CFL)
    print(f"Saving SW data to {output_path} (shape: {data.shape})")
    print(f"  Size: {data.nbytes / (1024**2):.1f} MB")
    np.save(output_path, data)
    print("Done.")
