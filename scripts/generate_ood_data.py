# ============================================================================
# OOD 数据生成: 测 BVA 在分布外是否比 Plain 更稳健
#
# 训练分布:
#   Burgers: k_max=5 Fourier IC, A/B ~ N(0,1)/k², CFL clip 0.9
#   BL:      8 个 Riemann (uL, uR) pair + 噪声 ±0.02
#
# OOD 设计:
#   OOD-1 (burgers_ood_kmax10): k_max=10 (高频 IC, shock 形成更剧烈)
#   OOD-2 (burgers_ood_amp2):   k_max=5 但振幅放大 2x (shock 强度翻倍)
#   OOD-3 (bl_ood_riemann):     新 Riemann pair 集合 (训练不见过)
#
# 各 OOD 数据集 N=200 (用于 eval, 不用训练)
# ============================================================================

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env
from src.data.burgers_1d_solver import burgers_godunov_1d
from src.pdes.bl_solver import solve_bl_1d


# 通用网格参数 (与训练数据一致)
NX = 128
NT = 100
DT = 0.005
DX = 2 * np.pi / NX
N_OOD = 200


def generate_burgers_ood_kmax10(n_samples: int = N_OOD) -> np.ndarray:
    """OOD-1: k_max=10 (训练 k=5)."""
    print(f"[OOD-1] Burgers k_max=10, N={n_samples}, Nx={NX}")
    data = np.zeros((n_samples, NT + 1, NX), dtype=np.float32)
    x = np.linspace(0, 2 * np.pi, NX, endpoint=False)

    rng = np.random.RandomState(seed=20260503)
    k_max = 10
    for i in tqdm(range(n_samples)):
        A = rng.randn(k_max) / (np.arange(1, k_max + 1) ** 2)
        B = rng.randn(k_max) / (np.arange(1, k_max + 1) ** 2)
        u0 = np.zeros(NX)
        for k in range(1, k_max + 1):
            u0 += A[k - 1] * np.sin(k * x) + B[k - 1] * np.cos(k * x)
        # CFL clip
        u_max = np.max(np.abs(u0))
        if u_max * DT / DX > 0.9:
            u0 = u0 * (0.9 / (u_max * DT / DX))
        data[i] = burgers_godunov_1d(u0, nx=NX, nt=NT, dx=DX, dt=DT)
    return data


def generate_burgers_ood_amp2(n_samples: int = N_OOD) -> np.ndarray:
    """OOD-2: k_max=5 但振幅放大 2x (shock 更剧烈)."""
    print(f"[OOD-2] Burgers amp×2, N={n_samples}, Nx={NX}")
    data = np.zeros((n_samples, NT + 1, NX), dtype=np.float32)
    x = np.linspace(0, 2 * np.pi, NX, endpoint=False)

    rng = np.random.RandomState(seed=20260503)
    k_max = 5
    for i in tqdm(range(n_samples)):
        # 训练: A,B ~ N(0,1)/k². OOD: ×2 振幅 (但 CFL clip 后实际效果取决于 u_max)
        A = 2.0 * rng.randn(k_max) / (np.arange(1, k_max + 1) ** 2)
        B = 2.0 * rng.randn(k_max) / (np.arange(1, k_max + 1) ** 2)
        u0 = np.zeros(NX)
        for k in range(1, k_max + 1):
            u0 += A[k - 1] * np.sin(k * x) + B[k - 1] * np.cos(k * x)
        # CFL clip
        u_max = np.max(np.abs(u0))
        if u_max * DT / DX > 0.9:
            u0 = u0 * (0.9 / (u_max * DT / DX))
        data[i] = burgers_godunov_1d(u0, nx=NX, nt=NT, dx=DX, dt=DT)
    return data


def generate_bl_ood_riemann(n_samples: int = N_OOD) -> np.ndarray:
    """OOD-3: BL 用训练未见的 Riemann (uL, uR) 对."""
    print(f"[OOD-3] BL OOD Riemann, N={n_samples}, Nx={NX}")
    # 训练用的 8 对见 generate_bl_data.py
    # OOD 用极端 / 中间值组合
    ood_riemann = [
        (0.5, 0.5),    # 退化: 无跳变 (训练未见)
        (0.99, 0.01),  # 极端 shock
        (0.01, 0.99),  # 极端 rarefaction
        (0.7, 0.5),    # 中等 shock (训练见 0.6/0.1, 0.85/0.15)
        (0.4, 0.6),    # 中等 rarefaction (训练见 0.2/0.8, 0.15/0.85)
        (0.5, 0.2),    # 训练未见的中间组合
        (0.5, 0.8),    # 同上
        (0.95, 0.5),   # 高强度 shock
    ]
    data = np.zeros((n_samples, NT + 1, NX), dtype=np.float32)
    half = NX // 2

    rng = np.random.RandomState(seed=20260503)
    samples_per_pair = n_samples // len(ood_riemann)
    sample_idx = 0
    for pair_idx, (ul, ur) in enumerate(ood_riemann):
        n_for_pair = samples_per_pair + (1 if pair_idx < n_samples - samples_per_pair * len(ood_riemann) else 0)
        for _ in range(n_for_pair):
            u0 = np.zeros(NX)
            noise_l = rng.uniform(-0.02, 0.02, half)
            noise_r = rng.uniform(-0.02, 0.02, NX - half)
            u0[:half] = np.clip(ul + noise_l, 0.01, 0.99)
            u0[half:] = np.clip(ur + noise_r, 0.01, 0.99)
            data[sample_idx] = solve_bl_1d(u0, nx=NX, nt=NT, dt=DT, dx=DX)
            sample_idx += 1
            if sample_idx >= n_samples:
                break
        if sample_idx >= n_samples:
            break
    return data


def main():
    out_dir = env.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # OOD-1
    p1 = out_dir / f"burgers_ood_kmax10_N{N_OOD}_Nx{NX}.npy"
    if p1.exists():
        print(f"[skip] {p1} 已存在")
    else:
        data1 = generate_burgers_ood_kmax10(N_OOD)
        np.save(p1, data1)
        print(f"[save] {p1} shape={data1.shape}")

    # OOD-2
    p2 = out_dir / f"burgers_ood_amp2_N{N_OOD}_Nx{NX}.npy"
    if p2.exists():
        print(f"[skip] {p2} 已存在")
    else:
        data2 = generate_burgers_ood_amp2(N_OOD)
        np.save(p2, data2)
        print(f"[save] {p2} shape={data2.shape}")

    # OOD-3
    p3 = out_dir / f"bl_ood_riemann_N{N_OOD}_Nx{NX}.npy"
    if p3.exists():
        print(f"[skip] {p3} 已存在")
    else:
        data3 = generate_bl_ood_riemann(N_OOD)
        np.save(p3, data3)
        print(f"[save] {p3} shape={data3.shape}")

    print(f"\n[ood] 全部完成. 数据在 {out_dir}/")


if __name__ == "__main__":
    main()
