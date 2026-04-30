# ============================================================================
# 2D 浅水方程 (Shallow-Water Equations): Godunov + 维度分裂 求解器
# 论文对齐: E4 实验 — 圆形溃坝 (circular dam break)
#
# 控制方程 (守恒型, 平底无摩擦):
#   ∂_t h   + ∂_x (hu)  + ∂_y (hv)   = 0                (质量守恒)
#   ∂_t(hu) + ∂_x(hu²+½gh²) + ∂_y(huv) = 0              (x-动量)
#   ∂_t(hv) + ∂_x(huv) + ∂_y(hv²+½gh²) = 0              (y-动量)
#
#   其中 h = 水深, u = x-流速, v = y-流速, g = 重力加速度
#
# 数值方法:
#   - Strang 维度分裂: U^{n+1} = S_y(dt/2) ∘ S_x(dt) ∘ S_y(dt/2) U^n
#   - 每个 1D 扫描用 HLL Riemann 求解器 (Harten-Lax-van Leer)
#   - HLL 在浅水方程中与 Euler (γ=2) 同构, 捕捉溃坝激波
#   - 周期 BC: 各方向单独处理; 分裂误差 O(dt²)
#
# HLL 通量 (1D):
#   F_HLL = (S_R·F_L - S_L·F_R + S_L·S_R·(U_R - U_L)) / (S_R - S_L)
#   波速: S_L = min(u_L - √(gh_L), u_R - √(gh_R))
#         S_R = max(u_L + √(gh_L), u_R + √(gh_R))
# ============================================================================

import numpy as np
from typing import Tuple


# ---- 浅水基本函数 ----

def sw_cons_to_prim(U: np.ndarray, g: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    守恒变量 U=(h, hu, hv) → 原始变量 (h, u, v).
    
    返回: h, u, v  — 各为 (Nx, Ny) 或 (Nx,) 或 (Ny,)
    """
    eps = 1e-12
    h = np.maximum(U[0], eps)     # 水深 > 0
    hu = U[1]
    hv = U[2]
    u = hu / h
    v = hv / h
    return h, u, v


def sw_flux_x(U: np.ndarray, g: float) -> np.ndarray:
    """
    x-方向浅水物理通量 F_x(U), 形状 (3, ...).
    
    F_x[0] = hu
    F_x[1] = hu²/h + ½ g h²
    F_x[2] = huv/h
    """
    h, u, v = sw_cons_to_prim(U, g)
    F = np.zeros_like(U)
    F[0] = h * u                               # hu
    F[1] = h * u * u + 0.5 * g * h * h         # hu² + ½gh²
    F[2] = h * u * v                            # huv
    return F


def sw_flux_y(U: np.ndarray, g: float) -> np.ndarray:
    """
    y-方向浅水物理通量 F_y(U), 形状 (3, ...).
    
    F_y[0] = hv
    F_y[1] = huv = hv · u  (huv/h = hv·u in terms of cons vars)
    F_y[2] = hv²/h + ½ g h²
    """
    h, u, v = sw_cons_to_prim(U, g)
    F = np.zeros_like(U)
    F[0] = h * v                                # hv
    F[1] = h * u * v                             # huv
    F[2] = h * v * v + 0.5 * g * h * h          # hv² + ½gh²
    return F


# ---- HLL Riemann 求解器 (用于维度分裂中的 1D 扫描) ----

def sw_hll_flux_1d(UL: np.ndarray, UR: np.ndarray, g: float, vel_idx: int) -> np.ndarray:
    """
    1D 浅水 HLL 数值通量, 向量化到最后一个轴.
    
    适用场景: 维度分裂中的 x-扫描 (vel_idx=1) 或 y-扫描 (vel_idx=2).
    
    波速:
      c = √(gh)
      vel_idx=1 (x-扫描): S = u ± √(gh)
      vel_idx=2 (y-扫描): S = v ± √(gh)
    
    F_HLL = (S_R·F_L - S_L·F_R + S_L·S_R·(U_R - U_L)) / (S_R - S_L)
    
    参数:
        UL, UR:  (3, N) 左/右状态
        g:       重力加速度
        vel_idx: 1 表示 x-扫描 (用 u 算波速), 2 表示 y-扫描 (用 v 算波速)
    返回:
        F: (3, N) HLL 数值通量
    """
    eps = 1e-12
    # 原始变量
    h_L = np.maximum(UL[0], eps)
    h_R = np.maximum(UR[0], eps)
    vel_L = np.maximum(UL[vel_idx], -1e10) / h_L  # 流速
    vel_R = np.maximum(UR[vel_idx], -1e10) / h_R

    # 声速 (浅水重力波速)
    c_L = np.sqrt(g * h_L)
    c_R = np.sqrt(g * h_R)

    # 波速估计
    S_L = np.minimum(vel_L - c_L, vel_R - c_R)
    S_R = np.maximum(vel_L + c_L, vel_R + c_R)
    # 拒绝退化的零宽度波系
    S_L = np.minimum(S_L, -eps)
    S_R = np.maximum(S_R, eps)

    # 物理通量 — 根据扫描方向选择
    if vel_idx == 1:
        F_L = sw_flux_x(UL, g)
        F_R = sw_flux_x(UR, g)
    else:
        F_L = sw_flux_y(UL, g)
        F_R = sw_flux_y(UR, g)

    # HLL 通量
    denom = np.maximum(S_R - S_L, eps)
    F = (S_R * F_L - S_L * F_R + S_L * S_R * (UR - UL)) / denom

    # NaN guard: 异常列回退为算术平均通量
    nan_cols = ~np.isfinite(F).all(axis=0)
    if nan_cols.any():
        F[:, nan_cols] = 0.5 * (F_L[:, nan_cols] + F_R[:, nan_cols])

    return F


# ---- 维度分裂扫描 ----

def _sw_1d_sweep(U: np.ndarray, dt: float, ds: float, g: float,
                 vel_idx: int, sweep_axis: int) -> np.ndarray:
    """
    通用 1D 扫描: 沿 sweep_axis 方向推进 dt.
    
    参数:
        U:          (3, Nx, Ny) — 守恒变量
        dt:         扫描时间步长
        ds:         该方向的空间步长
        g:          重力加速度
        vel_idx:    1 (x-流速) 或 2 (y-流速) — 决定波速用哪个分量
        sweep_axis: 1 (沿 x) 或 2 (沿 y) — numpy axis
    返回:
        U_next: (3, Nx, Ny) 扫描后状态
    """
    nx, ny = U.shape[1], U.shape[2]
    eps = 1e-12

    # 周期边界构造左右 interface 状态
    # UL[:, i, :] = U[:, i-1, :],  ghost left = U[:, -1, :]
    # UR[:, i, :] = U[:, i, :],    ghost right = U[:, 0, :]
    if sweep_axis == 1:  # x-扫描
        N = nx  # 网格数 = nx, 界面数 = nx+1
        ghost_left = U[:, -1:, :]    # (3, 1, ny)
        ghost_right = U[:, :1, :]    # (3, 1, ny)
        # UL = [ghost_left, U0, U1, ..., U_{nx-1}]  共 nx+1 列
        UL = np.concatenate([ghost_left, U], axis=1)             # (3, nx+1, ny)
        # UR = [U0, U1, ..., U_{nx-1}, ghost_right]
        UR = np.concatenate([U, ghost_right], axis=1)            # (3, nx+1, ny)
    else:  # y-扫描 (sweep_axis == 2)
        N = ny  # 网格数 = ny, 界面数 = ny+1
        ghost_left = U[:, :, -1:]    # (3, nx, 1)
        ghost_right = U[:, :, :1]    # (3, nx, 1)
        # UL = [ghost_left, U0, U1, ..., U_{ny-1}]  共 ny+1 列
        UL = np.concatenate([ghost_left, U], axis=2)             # (3, nx, ny+1)
        # UR = [U0, U1, ..., U_{ny-1}, ghost_right]
        UR = np.concatenate([U, ghost_right], axis=2)            # (3, nx, ny+1)

    # 沿扫描轴逐条计算 HLL 通量
    # 重组为 (3, N, M) → 对每条做 HLL
    if sweep_axis == 1:
        # 重组: (3, N, ny) → ny 条独立的 1D 问题
        F_interfaces = np.zeros_like(UL)  # (3, N, ny)
        for j in range(ny):
            F_interfaces[:, :, j] = sw_hll_flux_1d(UL[:, :, j], UR[:, :, j], g, vel_idx)
        # 更新
        U_next = U - (dt / ds) * (F_interfaces[:, 1:, :] - F_interfaces[:, :-1, :])
    else:
        # 重组: (3, nx, N) → nx 条独立的 1D 问题
        F_interfaces = np.zeros_like(UL)  # (3, nx, N)
        for i in range(nx):
            F_interfaces[:, i, :] = sw_hll_flux_1d(UL[:, i, :], UR[:, i, :], g, vel_idx)
        U_next = U - (dt / ds) * (F_interfaces[:, :, 1:] - F_interfaces[:, :, :-1])

    # 正值性修复: 水深 h < 0 时回退
    bad = U_next[0] <= 0.0
    if bad.any():
        U_next[0, bad] = U[0, bad]
        U_next[1, bad] = U[1, bad]
        U_next[2, bad] = U[2, bad]

    return U_next


def sw_x_sweep(U: np.ndarray, dt: float, dx: float, g: float) -> np.ndarray:
    """x-方向 1D 扫描: ∂_t U + ∂_x F_x(U) = 0."""
    return _sw_1d_sweep(U, dt, dx, g, vel_idx=1, sweep_axis=1)


def sw_y_sweep(U: np.ndarray, dt: float, dy: float, g: float) -> np.ndarray:
    """y-方向 1D 扫描: ∂_t U + ∂_y F_y(U) = 0."""
    return _sw_1d_sweep(U, dt, dy, g, vel_idx=2, sweep_axis=2)


# ---- Strang 分裂时间积分 ----

def solve_shallow_water_2d(
    h0: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    nx: int,
    ny: int,
    nt: int,
    dt: float,
    dx: float,
    dy: float,
    g: float = 9.81,
) -> np.ndarray:
    """
    二维浅水方程完整时间积分 (Strang 维度分裂).
    
    S(dt) = S_y(dt/2) ∘ S_x(dt) ∘ S_y(dt/2)
    
    参数:
        h0, u0, v0: 初始水深及流速场, 形状 (nx, ny)
        nx, ny:     网格点数
        nt:         时间步数 (每个完整步含三次 1D 扫描)
        dt:         完整步时间步长
        dx, dy:     空间步长
        g:          重力加速度
    返回:
        h_hist: (nt+1, nx, ny) 水深时间轨迹
    """
    # 初始化守恒变量
    U = np.zeros((3, nx, ny), dtype=np.float64)
    U[0] = h0
    U[1] = h0 * u0
    U[2] = h0 * v0

    h_hist = np.zeros((nt + 1, nx, ny), dtype=np.float32)
    h_hist[0] = h0.astype(np.float32)

    dt_half = 0.5 * dt

    for n in range(nt):
        # Strang splitting: Y(dt/2) → X(dt) → Y(dt/2)
        U = sw_y_sweep(U, dt_half, dy, g)       # 半步 y-扫描
        U = sw_x_sweep(U, dt, dx, g)             # 完整 x-扫描
        U = sw_y_sweep(U, dt_half, dy, g)        # 半步 y-扫描
        h_hist[n + 1] = U[0].astype(np.float32)

    return h_hist


# ---- 圆形溃坝 IC ----

def dam_break_ic(
    nx: int,
    ny: int,
    h_in: float = 2.0,
    h_out: float = 1.0,
    r0: float = 0.15,
    cx: float = 0.5,
    cy: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    圆形溃坝初始条件: 半径 r0 内 h=h_in, 外 h=h_out, 初速为零.

    参数:
        nx, ny: 网格
        h_in:   坝内水深
        h_out:  坝外水深
        r0:     坝半径
        cx, cy: 坝中心坐标
    返回:
        h0, u0, v0 — (nx, ny) 初始场
    """
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')  # (nx, ny)
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    h0 = np.where(R <= r0, h_in, h_out)
    u0 = np.zeros((nx, ny))
    v0 = np.zeros((nx, ny))
    return h0, u0, v0


def sw_max_wavespeed(h: np.ndarray, u: np.ndarray, v: np.ndarray, g: float) -> float:
    """二维浅水最大波速 max(|u|+√(gh), |v|+√(gh)), 用于 CFL."""
    c = np.sqrt(g * np.maximum(h, 1e-12))
    return float(max(np.max(np.abs(u) + c), np.max(np.abs(v) + c)))
