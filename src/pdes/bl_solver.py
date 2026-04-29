# ============================================================================
# Buckley–Leverett 1D Godunov 求解器
# 论文对齐: E2 Buckley–Leverett experiment
# PDE: ∂_t u + ∂_x f(u) = 0, f(u) = u²/(u²+(1-u)²), u ∈ [0,1]
#
# 数值方法: 一阶 Godunov 有限体积法 (迎风格式)
#   因为 f'(u) ≥ 0 (monotonically increasing), 波始终从左向右传播,
#   Godunov flux F_{i+1/2} = f(u_i), 退化为简单迎风格式:
#     u_i^{n+1} = u_i^n - (dt/dx) * (f(u_i^n) - f(u_{i-1}^n))
#
# CFL 条件: max|f'(u)| * dt/dx ≤ 1, 其中 f'_max ≈ 1.5 at u≈0.2
# ============================================================================
import numpy as np
from src.pdes.bl_flux import bl_flux


def bl_godunov_step(u, dt, dx):
    """
    一阶 Godunov 时间推进 (迎风格式).

    参数:
        u:   当前解向量 [Nx], 每元素 ∈ [0,1]
        dt:  时间步长
        dx:  空间步长
    返回:
        下一时间步解向量 [Nx]
    """
    nx = len(u)
    u_next = np.zeros_like(u)

    # 计算所有网格点的通量 f(u_i)
    f_all = bl_flux(u)  # [Nx]

    # 周期边界: i=0 时左通量为 f(u_{Nx-1})
    f_left = np.roll(f_all, 1)  # f(u_{i-1}), 周期位移

    # Godunov 推进: u_i^{n+1} = u_i^n - (dt/dx) * (f(u_i) - f(u_{i-1}))
    u_next = u - (dt / dx) * (f_all - f_left)

    # 裁剪到 [0, 1] (保持物理解范围, 防止数值溢出)
    u_next = np.clip(u_next, 0.0, 1.0)

    return u_next


def solve_bl_1d(u0, nx, nt, dt, dx):
    """
    完整时间积分: 从初始条件 u0 推进 nt 步.

    参数:
        u0: 初始条件 [Nx]
        nx: 网格点数
        nt: 时间步数
        dt: 时间步长
        dx: 空间步长
    返回:
        u_hist: 时空轨迹 [nt+1, Nx], u_hist[0] = u0
    """
    u_hist = np.zeros((nt + 1, nx), dtype=np.float32)
    u_hist[0] = u0
    u = u0.copy()

    for n in range(nt):
        u = bl_godunov_step(u, dt, dx)
        u_hist[n + 1] = u

    return u_hist
