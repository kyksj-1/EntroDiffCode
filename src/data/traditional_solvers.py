# ============================================================================
# traditional_solvers.py — 传统数值方法求解器 (非神经网络 baseline)
# 提供 Lax-Friedrichs / MacCormack / WENO5 等经典格式, 用于与 EntroDiff 对比
# ============================================================================
import numpy as np

# ============================================================================
# 1. Lax-Friedrichs 格式 (一阶, 极度耗散, shock 处模糊)
#    论文用途: 展示"传统低阶格式在 shock 处也难以准确"
# ============================================================================
def burgers_lax_friedrichs_1d(u0, nx, nt, dx, dt):
    """
    Lax-Friedrichs 一阶迎风格式求解 1D inviscid Burgers.
    通量: f(u) = 0.5*u^2
    Lax-Friedrichs: u_i^{n+1} = 0.5*(u_{i+1}^n + u_{i-1}^n) - dt/(2*dx)*(f_{i+1}^n - f_{i-1}^n)

    特点: 无条件稳定但数值粘性极大, shock 处严重抹平.
    返回: [nt+1, nx] 轨迹数组
    """
    u_hist = np.zeros((nt + 1, nx), dtype=np.float32)
    u_hist[0] = u0.copy()
    u = u0.copy()

    for n in range(nt):
        u_new = np.zeros_like(u)
        # 通量 f = 0.5*u^2
        f = 0.5 * u**2
        # 周期边界索引
        for i in range(nx):
            ip1 = (i + 1) % nx
            im1 = (i - 1) % nx
            u_new[i] = 0.5 * (u[ip1] + u[im1]) - dt/(2*dx) * (f[ip1] - f[im1])
        u = u_new
        u_hist[n+1] = u
    return u_hist


# ============================================================================
# 2. MacCormack 格式 (二阶预测-校正, shock 处震荡)
#    论文用途: 展示"高阶格式在 shock 处产生 Gibbs 震荡"
# ============================================================================
def burgers_maccormack_1d(u0, nx, nt, dx, dt):
    """
    MacCormack 预测-校正格式求解 1D inviscid Burgers.
    预测步: u*_i = u_i^n - dt/dx * (f_{i+1}^n - f_i^n)
    校正步: u^{n+1}_i = 0.5*(u_i^n + u*_i - dt/dx*(f*_i - f*_{i-1}))

    特点: 二阶精度但 shock 处产生非物理震荡 (Gibbs 现象).
    返回: [nt+1, nx] 轨迹数组
    """
    u_hist = np.zeros((nt + 1, nx), dtype=np.float32)
    u_hist[0] = u0.copy()
    u = u0.copy()

    for n in range(nt):
        # ---- 预测步: 前向差分 ----
        f = 0.5 * u**2
        u_star = np.zeros_like(u)
        for i in range(nx):
            ip1 = (i + 1) % nx
            u_star[i] = u[i] - dt/dx * (f[ip1] - f[i])

        # ---- 校正步: 后向差分 ----
        f_star = 0.5 * u_star**2
        u_new = np.zeros_like(u)
        for i in range(nx):
            im1 = (i - 1) % nx
            u_new[i] = 0.5 * (u[i] + u_star[i] - dt/dx * (f_star[i] - f_star[im1]))

        u = u_new
        u_hist[n+1] = u
    return u_hist


# ============================================================================
# 3. 中心差分 + 人工粘性 (最朴素的格式, 论文中作为"最差 baseline")
# ============================================================================
def burgers_central_viscosity_1d(u0, nx, nt, dx, dt, nu_art=0.01):
    """
    中心差分 + 显式人工粘性求解 Burgers 方程.
    格式: u^{n+1} = u^n - dt/dx * 0.5*u^2_x (中心差分) + nu_art*dt/dx^2 * u_xx

    特点: 不引入人工粘性 (nu_art=0) 时在 shock 处发散;
          引入后 shock 被抹平为宽过渡层, 与 Lax-Friedrichs 类似但更差.
    返回: [nt+1, nx] 轨迹数组
    """
    u_hist = np.zeros((nt + 1, nx), dtype=np.float32)
    u_hist[0] = u0.copy()
    u = u0.copy()
    coeff = dt / dx
    visc = nu_art * dt / (dx**2)

    for n in range(nt):
        f = 0.5 * u**2
        u_new = np.zeros_like(u)
        for i in range(nx):
            ip1 = (i + 1) % nx
            im1 = (i - 1) % nx
            # 中心差分通量
            conv = -coeff * (f[ip1] - f[im1]) / 2.0
            # 人工粘性
            diff = visc * (u[ip1] - 2*u[i] + u[im1])
            u_new[i] = u[i] + conv + diff
        u = u_new
        u_hist[n+1] = u
    return u_hist
