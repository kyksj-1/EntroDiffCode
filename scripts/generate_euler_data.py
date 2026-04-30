# ============================================================================
# Euler Sod 激波管 1D 数据生成
# 论文对齐: E3 experiment
# 使用 HLLC+Godunov 求解器, 生成多种 (ρ_L,p_L,ρ_R,p_R) 组合的 Sod 型 IC
# 产出: 激波+接触+稀疏波 三波结构的训练轨迹
# ============================================================================
import sys, numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env
from src.pdes.euler_sod import sod_initial_condition, solve_euler_1d, euler_max_wavespeed
from tqdm import tqdm
from typing import Any


def generate_euler_data(
    n_samples: int = 2000,
    nx: int = 256,
    nt: int = 500,
    gamma: float = 1.4,
    cfl: float = 0.4,
) -> np.ndarray:
    """
    生成 1D Euler Sod 型数据集.

    初始条件: 分段常数 Riemann 问题, 膜位于 x=0.5.
    在经典 Sod 值附近采样, 覆盖不同强度的激波 / 稀疏波组合.

    参数:
        n_samples: 样本总数
        nx:       空间网格点数
        nt:       时间步数
        gamma:    绝热指数
        cfl:      CFL 数 (≤0.5, 保证 HLLC 稳定性)
    返回:
        data_all: (n_samples, 3, nt+1, nx) float32 数组
    """
    dx = 1.0 / nx
    # dx 预先可知, dt 动态确定 (见下文)

    # — 参数组合: 以经典 Sod 为中心在邻域内采样 —
    # 经典 Sod: ρ_L=1, p_L=1, ρ_R=0.125, p_R=0.1, u_L=u_R=0
    # 附加更强的与更弱的激波, 以及非对称压力比
    param_configs = [
        # (label, rho_L, p_L, rho_R, p_R)
        ("classic Sod",   1.0,   1.0,   0.125, 0.1),
        ("weak shock",    1.0,   0.5,   0.3,   0.2),
        ("strong shock",  1.0,   2.0,   0.1,   0.05),
        ("dense left",    2.0,   1.0,   0.1,   0.1),
        ("dense right",   0.5,   1.0,   1.0,   0.5),
        ("hot left",      1.0,   3.0,   0.125, 0.1),
        ("cold right",    1.0,   1.0,   0.5,   0.01),
        ("balanced",      1.0,   1.0,   0.5,   1.0),
        ("mild ratio",    1.0,   1.0,   0.4,   0.4),
        ("extreme P-ratio",1.0,  5.0,   0.125, 0.02),
    ]

    n_configs = len(param_configs)
    samples_per_config = n_samples // n_configs
    remainder = n_samples - samples_per_config * n_configs

    print(f"[E3] Generating {n_samples} Euler Sod trajectories...")
    print(f"  Grid: nx={nx}, nt={nt}, dx={dx:.6f}, cfl={cfl}")
    print(f"  Configs: {n_configs} × ~{samples_per_config}")

    data_all = np.zeros((n_samples, 3, nt + 1, nx), dtype=np.float32)

    sample_idx = 0
    pbar = tqdm(total=n_samples, desc="Euler data")

    for cfg_idx, (label, rho_L0, p_L0, rho_R0, p_R0) in enumerate(param_configs):
        n_for_cfg = samples_per_config + (1 if cfg_idx < remainder else 0)

        for _ in range(n_for_cfg):
            # 在参数基准值 ±20% 范围内随机扰动 (保持物理解)
            rho_L = rho_L0 * (1.0 + np.random.uniform(-0.2, 0.2))
            p_L   = p_L0   * (1.0 + np.random.uniform(-0.2, 0.2))
            rho_R = rho_R0 * (1.0 + np.random.uniform(-0.2, 0.2))
            p_R   = p_R0   * (1.0 + np.random.uniform(-0.2, 0.2))
            # 扰动膜的位置以增加多样性
            x_mid = 0.5 + np.random.uniform(-0.05, 0.05)

            # 确保密度和压力为正
            rho_L = max(rho_L, 0.05)
            p_L = max(p_L, 0.01)
            rho_R = max(rho_R, 0.05)
            p_R = max(p_R, 0.01)

            # 构建 IC
            U0 = sod_initial_condition(
                nx=nx,
                rho_L=rho_L, u_L=0.0, p_L=p_L,
                rho_R=rho_R, u_R=0.0, p_R=p_R,
                x_mid=x_mid, gamma=gamma,
            )

            # CFL 确定 dt (用初始条件估计, 每个样本动态算)
            w_max = euler_max_wavespeed(U0, gamma)
            dt = cfl * dx / max(w_max, 1e-6)

            # 求解
            U_hist = solve_euler_1d(U0, nx=nx, nt=nt, dt=dt, dx=dx, gamma=gamma)
            data_all[sample_idx] = U_hist
            sample_idx += 1
            pbar.update(1)

    pbar.close()
    return data_all


if __name__ == "__main__":
    NX = 256
    NT = 500
    GAMMA = 1.4
    CFL = 0.4
    N_SAMPLES = 2000

    output_path = env.data_dir / "euler_sod_N2000_Nx256.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_euler_data(n_samples=N_SAMPLES, nx=NX, nt=NT, gamma=GAMMA, cfl=CFL)
    print(f"Saving Euler Sod data to {output_path} (shape: {data.shape})")
    print(f"  Size: {data.nbytes / (1024**2):.1f} MB")
    np.save(output_path, data)
    print("Done.")
