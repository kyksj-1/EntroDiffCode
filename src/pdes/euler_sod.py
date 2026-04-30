# ============================================================================
# 1D Euler 方程组: Godunov + HLLC Riemann 求解器
# 论文对齐: E3 实验 — Sod 激波管问题
#
# 控制方程: 守恒型 1D Euler 方程
#   U_t + F(U)_x = 0
#   其中 U = (ρ, ρu, E)^T (密度、动量、总能)
#        F = (ρu, ρu²+p, u(E+p))^T
#   状态方程: p = (γ-1)[E - (ρu)²/(2ρ)],  γ = 1.4 (理想气体)
#   声速: c = sqrt(γ p / ρ)
#
# 数值方法: 一阶 Godunov 有限体积法
#   - HLLC Riemann 求解器 (Harten-Lax-van Leer + Contact)
#   - 三波结构: 左行波(S_L), 接触间断(S_*), 右行波(S_R)
#   - 精确捕捉接触间断, 减小中间波的数值耗散
#   - 透射 (零梯度) 边界条件: 波不碰壁时自洽
#
# CFL 条件: max(|u|+c) * dt/dx ≤ 0.5
# ============================================================================

import numpy as np
from typing import Tuple


# ---- 欧拉基本函数 ----

def euler_prim_to_cons(rho: np.ndarray, u: np.ndarray, p: np.ndarray, gamma: float) -> np.ndarray:
    """
    原始变量 (ρ, u, p) → 守恒变量 (ρ, ρu, E).
    
    返回: U (3, ...)  — 密度, 动量, 总能
    """
    rho = np.asarray(rho)
    u = np.asarray(u)
    p = np.asarray(p)
    mom = rho * u
    E = p / (gamma - 1.0) + 0.5 * mom * u
    return np.stack([rho, mom, E], axis=0)


def euler_cons_to_prim(U: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    守恒变量 U=(ρ, ρu, E) → 原始变量 (ρ, u, p).
    
    返回: ρ, u, p  — 各分量与 U 具有相同的 trailing 形状
    """
    eps = 1e-12
    rho = np.maximum(U[0], eps)           # 密度 > 0 (防除零/真空)
    mom = U[1]
    E = U[2]
    u = mom / rho
    # p = (γ-1)[E - (ρu)²/(2ρ)], 保证 ≥ ε
    p = (gamma - 1.0) * (E - 0.5 * mom * u)
    p = np.maximum(p, eps)
    return rho, u, p


def euler_flux(U: np.ndarray, gamma: float) -> np.ndarray:
    """
    欧拉方程物理通量 F(U), 形状与 U 一致.
    
    F[0] = ρu (质量通量)
    F[1] = ρu² + p (动量通量)
    F[2] = u(E + p) (能量通量)
    """
    rho, u, p = euler_cons_to_prim(U, gamma)
    F = np.zeros_like(U)
    F[0] = rho * u                      # ρu
    F[1] = rho * u * u + p              # ρu² + p
    F[2] = u * (U[2] + p)              # u(E + p)
    return F


def euler_max_wavespeed(U: np.ndarray, gamma: float) -> float:
    """网格中最大波速 max(|u|+c), 用于 CFL 条件."""
    rho, u, p = euler_cons_to_prim(U, gamma)
    c = np.sqrt(gamma * p / rho)
    return float(np.max(np.abs(u) + c))


# ---- HLLC Riemann 求解器 ----

def euler_hllc_flux(UL: np.ndarray, UR: np.ndarray, gamma: float) -> np.ndarray:
    """
    HLLC 数值通量, 向量化到最后一个轴.
    
    三波模型:
      左波 S_L  ─→ 星区 L (U*_L) ─→ 接触波 S_* ─→ 星区 R (U*_R) ─→ 右波 S_R
    
    波速估计 (Einfeldt 型, Batten et al. 1997):
      S_L = min(u_L - c_L,  u_R - c_R)
      S_R = max(u_L + c_L,  u_R + c_R)
      S_* = [p_R-p_L + ρ_L·u_L·(S_L-u_L) - ρ_R·u_R·(S_R-u_R)] /
            [ρ_L·(S_L-u_L) - ρ_R·(S_R-u_R)]
    
    中间态 U_*K (K=L,R):
      U_*K = ρ_K · (S_K - u_K)/(S_K - S_*) · [1, S_*, E_K/ρ_K + (S_*-u_K)(S_* + p_K/(ρ_K(S_K-u_K)))]^T
    
    参数:
        UL, UR: (3, N) 左/右状态 — 3 个守恒分量, N 个界面
        gamma:   绝热指数
    返回:
        F: (3, N) HLLC 数值通量
    """
    eps = 1e-12
    # — 原始变量 —
    rho_L, u_L, p_L = euler_cons_to_prim(UL, gamma)
    rho_R, u_R, p_R = euler_cons_to_prim(UR, gamma)
    E_L, E_R = UL[2], UR[2]

    # 声速
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)

    # — 物理通量 —
    F_L = euler_flux(UL, gamma)
    F_R = euler_flux(UR, gamma)

    # — 波速估计 (Einfeldt 型) —
    S_L = np.minimum(u_L - c_L, u_R - c_R)   # 左波: 取两状态较小者
    S_R = np.maximum(u_L + c_L, u_R + c_R)   # 右波: 取两状态较大者
    # 确保 S_L < 0 < S_R (拒绝退化的全同号波系)
    S_L = np.minimum(S_L, -eps)
    S_R = np.maximum(S_R, eps)

    # — 接触波速度 S_* (Batten et al. 1997, Eq. 31) —
    # du_K = S_K - u_K   (相对波速, 可为负! 左波通常为负)
    du_L = S_L - u_L     # 左波相对流速 (< 0 正常!)
    du_R = S_R - u_R     # 右波相对流速 (> 0 正常!)
    denom_s = rho_L * du_L - rho_R * du_R
    # 退化防护: denom_s ≈ 0 → S_star 无定义 → 回退 HLL
    degenerate = np.abs(denom_s) < eps
    denom_s = np.where(degenerate, eps, denom_s)  # 临时防除零
    S_star = (p_R - p_L + rho_L * u_L * du_L - rho_R * u_R * du_R) / denom_s
    S_star = np.clip(S_star, S_L + eps, S_R - eps)  # 保证 S_L < S_* < S_R

    # — 中间态 U_*K 的比值因子 —
    # ratio_K = (S_K - u_K) / (S_K - S_*)  (分子/分母可同号, 比值通常正)
    denom_L = S_L - S_star  # 左差: 通常 < 0
    denom_R = S_R - S_star  # 右差: 通常 > 0
    # 对退化界面用 HLL 回退: 任一 |denom_K| < eps → 标记
    bad_L = np.abs(du_L) < eps
    bad_R = np.abs(du_R) < eps

    ratio_L = du_L / np.where(np.abs(denom_L) < eps, eps, denom_L)
    ratio_R = du_R / np.where(np.abs(denom_R) < eps, eps, denom_R)

    # — 构造 U_*K —
    # 能量分量: E_*K = ρ_K·(S_K-u_K)/(S_K-S_*) · [E_K/ρ_K + (S_*-u_K)·(S_* + p_K/(ρ_K·(S_K-u_K)))]
    # 其中 p_K/(ρ_K·du_K) 在 du_K→0 时发散 → 退化标记
    safe_du_L = np.where(np.abs(du_L) < eps, eps, du_L)
    safe_du_R = np.where(np.abs(du_R) < eps, eps, du_R)

    U_star_L = np.zeros_like(UL)
    U_star_L[0] = rho_L * ratio_L
    U_star_L[1] = rho_L * ratio_L * S_star
    U_star_L[2] = rho_L * ratio_L * (
        E_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * safe_du_L))
    )

    U_star_R = np.zeros_like(UR)
    U_star_R[0] = rho_R * ratio_R
    U_star_R[1] = rho_R * ratio_R * S_star
    U_star_R[2] = rho_R * ratio_R * (
        E_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * safe_du_R))
    )

    # — HLL 回退通量 —
    F_hll = (S_R * F_L - S_L * F_R + S_L * S_R * (UR - UL)) / np.maximum(S_R - S_L, eps)

    # — 通量选取 (四路选择) —
    F = np.zeros_like(UL)

    # Case 1: S_L ≥ 0 → 全右行, 通量 = F_L
    m1 = S_L >= 0.0
    _assign_cols(F, F_L, m1)

    # Case 2: S_L < 0 ≤ S_* → 左星区, F = F_L + S_L·(U_*L - U_L)
    m2 = (S_L < 0.0) & (S_star >= 0.0)
    _assign_cols(F, F_L + S_L * (U_star_L - UL), m2)

    # Case 3: S_* < 0 ≤ S_R → 右星区, F = F_R + S_R·(U_*R - U_R)
    m3 = (S_star < 0.0) & (S_R >= 0.0)
    _assign_cols(F, F_R + S_R * (U_star_R - UR), m3)

    # Case 4: S_R < 0 → 全左行, 通量 = F_R
    m4 = S_R < 0.0
    _assign_cols(F, F_R, m4)

    # 回退: degenerate / NaN / inf → HLL
    fallback = degenerate | bad_L | bad_R | (~np.isfinite(F).all(axis=0))
    if fallback.any():
        _assign_cols(F, F_hll, fallback)

    return F


def _assign_cols(dst: np.ndarray, src: np.ndarray, mask: np.ndarray) -> None:
    """原地赋值: dst[:, mask] = src[:, mask], 用于布尔掩码跨列选取."""
    if mask.any():
        dst[:, mask] = src[:, mask]


# ---- Godunov 时间推进 ----

def euler_godunov_step(U: np.ndarray, dt: float, dx: float, gamma: float) -> np.ndarray:
    """
    单步 Godunov 有限体积推进, 透射 BC.

    参数:
        U: (3, Nx) 当前守恒变量
        dt: 时间步长
        dx: 空间步长
        gamma: 绝热指数
    返回:
        U_next: (3, Nx) 下一时刻守恒变量
    """
    Nx = U.shape[1]
    # 透射 BC: ghost cell 拷贝边界值 (∂/∂x = 0 at boundary)
    ghost_left = U[:, 0:1]    # (3, 1)
    ghost_right = U[:, -1:]   # (3, 1)

    # 构建 Nx+1 个界面左右状态
    # 界面 i 的左状态 = cell i-1 (或 ghost), 右状态 = cell i (或 ghost)
    # UL = [ghost_left, U0, U1, ..., U_{Nx-2}, U_{Nx-1}]  共 Nx+1 列
    # UR = [U0, U1, ..., U_{Nx-1}, ghost_right]           共 Nx+1 列
    UL_interfaces = np.concatenate([ghost_left, U], axis=1)          # (3, Nx+1)
    UR_interfaces = np.concatenate([U, ghost_right], axis=1)         # (3, Nx+1)

    # 各界面 HLLC 通量
    F_interfaces = euler_hllc_flux(UL_interfaces, UR_interfaces, gamma)  # (3, Nx+1)

    # 守恒更新: U^{n+1}_i = U^n_i - (dt/dx) * (F_{i+1/2} - F_{i-1/2})
    # F_interfaces[:, i] = 界面 i 的通量
    # cell i 用 F_interfaces[:, i+1] - F_interfaces[:, i]
    U_next = U - (dt / dx) * (F_interfaces[:, 1:] - F_interfaces[:, :-1])

    # 正值性修复: 密度 < 0 或 压力 < 0 时裁剪 (极少发生但防崩溃)
    rho_next, _, p_next = euler_cons_to_prim(U_next, gamma)
    fix_mask = (rho_next <= 0.0) | (p_next <= 0.0)
    if fix_mask.any():
        # 将异常 cell 回退为上一时间步的值
        U_next[:, fix_mask] = U[:, fix_mask]

    return U_next


def solve_euler_1d(
    U0: np.ndarray,
    nx: int,
    nt: int,
    dt: float,
    dx: float,
    gamma: float = 1.4,
) -> np.ndarray:
    """
    完整一维 Euler 方程时间积分.

    参数:
        U0:    初始条件 (3, nx) — 守恒变量 (ρ, ρu, E)
        nx:    空间网格点数
        nt:    时间步数
        dt:    时间步长
        dx:    空间步长
        gamma: 绝热指数 (默认 1.4)
    返回:
        U_hist: (3, nt+1, nx) 时间-空间轨迹
    """
    U_hist = np.zeros((3, nt + 1, nx), dtype=np.float32)
    U_hist[:, 0, :] = U0
    U = U0.copy()

    for n in range(nt):
        U = euler_godunov_step(U, dt, dx, gamma)
        U_hist[:, n + 1, :] = U

    return U_hist


# ---- Sod 激波管 IC ----

def sod_initial_condition(
    nx: int,
    rho_L: float = 1.0,
    u_L: float = 0.0,
    p_L: float = 1.0,
    rho_R: float = 0.125,
    u_R: float = 0.0,
    p_R: float = 0.1,
    x_mid: float = 0.5,
    gamma: float = 1.4,
) -> np.ndarray:
    """
    Sod 激波管初始条件: 左右各均匀态, x=x_mid 处有膜.

    经典 Sod 参数:
      (ρ_L, u_L, p_L) = (1.0, 0.0, 1.0)
      (ρ_R, u_R, p_R) = (0.125, 0.0, 0.1)

    返回: U (3, nx) 守恒变量
    """
    x = np.linspace(0.0, 1.0, nx)
    rho = np.where(x < x_mid, rho_L, rho_R)
    u = np.where(x < x_mid, u_L, u_R)
    p = np.where(x < x_mid, p_L, p_R)
    return euler_prim_to_cons(rho, u, p, gamma)
