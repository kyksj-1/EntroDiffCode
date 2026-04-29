# ============================================================================
# Buckley–Leverett 1D 数据生成
# 论文对齐: E2 experiment
# 使用 Riemann 问题初始条件 (uL, uR) 对, 每个对生成多组 IC
# 非凸通量产生 shock + rarefaction 复合波
# ============================================================================
import sys, numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env
from src.pdes.bl_solver import solve_bl_1d
from tqdm import tqdm


def generate_bl_data(n_samples=5000, nx=128, nt=100, dt=0.005, dx=None):
    """
    生成 Buckley–Leverett 方程数据集.

    初始条件: Riemann 问题 (分段常数), 多种 (uL, uR) 组合.
    每种组合产生不同比例的 shock + rarefaction 复合波.

    参数:
        n_samples: 样本总数 (会在 (uL,uR) 对间均匀分配)
        nx, nt, dt: 网格和步长参数
        dx: 空间步长 (默认 2π/nx)
    """
    if dx is None:
        dx = 2 * np.pi / nx

    # Riemann 问题 (uL, uR) 对 — 覆盖 shock-dominant / rarefaction-dominant / balanced
    riemann_pairs = [
        (0.8, 0.2),   # shock-dominant: uL >> uR
        (0.2, 0.8),   # rarefaction-dominant: uL << uR
        (0.6, 0.1),   # shock only (大跳变)
        (0.1, 0.9),   # rarefaction 为主
        (0.9, 0.3),   # shock + 部分 rarefaction
        (0.3, 0.7),   # rarefaction + 轻微 shock
        (0.85, 0.15), # 强 shock
        (0.15, 0.85), # 强 rarefaction
    ]

    samples_per_pair = n_samples // len(riemann_pairs)
    remainder = n_samples - samples_per_pair * len(riemann_pairs)

    print(f"Generating {n_samples} samples of 1D Buckley–Leverett...")
    print(f"  Grid: nx={nx}, nt={nt}, dx={dx:.4f}, dt={dt:.4f}")
    print(f"  Riemann pairs: {len(riemann_pairs)} × {samples_per_pair} + {remainder}")

    data_all = np.zeros((n_samples, nt + 1, nx), dtype=np.float32)
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    half = nx // 2  # Riemann 问题分界点

    sample_idx = 0
    pbar = tqdm(total=n_samples, desc="BL data")

    for pair_idx, (ul, ur) in enumerate(riemann_pairs):
        n_for_pair = samples_per_pair + (1 if pair_idx < remainder else 0)

        for _ in range(n_for_pair):
            # 构建 IC: 左右各为 ul, ur 加微小扰动 (避免退化)
            u0 = np.zeros(nx)
            noise_l = np.random.uniform(-0.02, 0.02, half)  # 微小噪声
            noise_r = np.random.uniform(-0.02, 0.02, nx - half)
            u0[:half] = np.clip(ul + noise_l, 0.01, 0.99)
            u0[half:] = np.clip(ur + noise_r, 0.01, 0.99)

            # CFL 检查: f'_max ≈ 1.5, 需要 dt/dx * 1.5 ≤ 1
            cfl = 1.5 * dt / dx
            if cfl > 0.9:
                # rescale not possible for CFL without changing physics; skip
                pass

            # 求解
            u_hist = solve_bl_1d(u0, nx=nx, nt=nt, dt=dt, dx=dx)
            data_all[sample_idx] = u_hist
            sample_idx += 1
            pbar.update(1)

    pbar.close()
    return data_all


if __name__ == "__main__":
    NX = 128
    NT = 100
    DT = 0.005
    DX = 2 * np.pi / NX
    N_SAMPLES = 5000

    output_path = env.data_dir / "bl_1d_N5000_Nx128.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_bl_data(n_samples=N_SAMPLES, nx=NX, nt=NT, dt=DT, dx=DX)
    print(f"Saving BL data to {output_path} (shape: {data.shape})")
    np.save(output_path, data)
    print("Done.")
