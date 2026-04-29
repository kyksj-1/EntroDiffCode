import os
import sys
import numpy as np
from pathlib import Path

# Setup paths to ensure src/ is discoverable
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.env_manager import env
from src.data.burgers_1d_solver import burgers_godunov_1d
from tqdm import tqdm

def generate_burgers_data(n_samples=1000, nx=256, nt=100, dt=0.001, dx=2*np.pi/256):
    """
    生成无粘 Burgers 方程 1D 数据集 (Godunov 有限体积法)。
    初始条件由 Fourier 模态叠加构造，后续时间步产生激波结构。

    Args:
        n_samples: 样本数
        nx: 空间网格点数
        nt: 时间步数
        dt: 时间步长
        dx: 空间步长 (2π / nx)
    Returns:
        data_hist: (n_samples, nt+1, nx) 的 float32 数组
    """
    print(f"Generating {n_samples} samples of 1D Burgers equation...")
    print(f"Grid: nx={nx}, nt={nt}, dx={dx:.4f}, dt={dt:.4f}")

    # 输出数组 (nt+1 因为包含 t=0 时刻)
    data_hist = np.zeros((n_samples, nt + 1, nx), dtype=np.float32)

    x = np.linspace(0, 2*np.pi, nx, endpoint=False)

    for i in tqdm(range(n_samples)):
        # 随机初始条件: Fourier 模态叠加
        k_max = 5
        A = np.random.randn(k_max)
        B = np.random.randn(k_max)

        # 1/k² 衰减: 保证 u(x,0) 幅值约在 [-1.5, 1.5]，避免 CFL 爆炸
        A /= (np.arange(1, k_max+1)**2)
        B /= (np.arange(1, k_max+1)**2)

        u0 = np.zeros(nx)
        for k in range(1, k_max+1):
            u0 += A[k-1] * np.sin(k * x) + B[k-1] * np.cos(k * x)

        u_max = np.max(np.abs(u0))
        cfl = u_max * dt / dx

        if cfl > 0.9:
            # 若 CFL 超标，整体缩放以满足 CFL 条件
            u0 = u0 * (0.9 / cfl)

        # Godunov 求解
        u_hist = burgers_godunov_1d(u0, nx=nx, nt=nt, dx=dx, dt=dt)
        data_hist[i] = u_hist

    return data_hist


if __name__ == "__main__":
    # === 粗网格配置 ===
    # Nx=64 粗网格: 在激波处仅 ~5-8 个网格点
    # 中央差分基线在此分辨率下数值粘性严重不足，激波处会产生大幅伪振荡
    # Godunov 迎风格式天然捕捉激波，不受网格分辨率退化影响 → 预期拉开差距
    NX = 64       # 粗网格: 激波区仅 5-8 个点 (标准 Nx=128 有 ~10-15 个点)
    DT = 0.01     # 时间步长加倍: 保持 T=NT×DT=0.5 (与 Nx=128 配置相同)
    NT = 50       # nt = T / DT = 0.5 / 0.01 = 50
    DX = 2*np.pi / NX
    N_SAMPLES = 5000  # 与原数据集样本数对齐

    output_path = env.data_dir / "burgers_coarse_N5000_Nx64.npy"

    data = generate_burgers_data(n_samples=N_SAMPLES, nx=NX, nt=NT, dt=DT, dx=DX)

    print(f"Saving generated data to {output_path} (shape: {data.shape})")
    np.save(output_path, data)
    print("Done.")
