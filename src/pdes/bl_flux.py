# ============================================================================
# Buckley–Leverett 通量函数 + Godunov 数值通量
# 论文对齐: E2 实验, flux f(u) = u²/(u²+(1-u)²)
#
# 通量性质:
#   - f'(u) = 2u(1-u)/(u²+(1-u)²)² ≥ 0 in [0,1] → 单调增
#   - f''(u) 在 u≈0.5 变号 → 非凸 (S 形通量)
#   - Godunov flux: 对单调通量退化为简单迎风 F = f(uL)
#     (因为 min_{[uL,uR]} f = f(uL) when uL ≤ uR, max = f(uL) when uL > uR)
# ============================================================================
import numpy as np
import torch


def bl_flux(u):
    """
    计算 Buckley–Leverett 通量 f(u) = u²/(u²+(1-u)²).

    参数:
        u: 饱和度值 (标量或 numpy/torch 数组), 应在 [0,1] 内
    返回:
        f(u) 通量值
    """
    if isinstance(u, torch.Tensor):
        u2 = u ** 2
        return u2 / (u2 + (1 - u) ** 2 + 1e-12)  # +ε 防除零
    u2 = np.asarray(u) ** 2
    return u2 / (u2 + (1 - np.asarray(u)) ** 2 + 1e-12)


def bl_godunov_flux(ul, ur):
    """
    Buckley–Leverett 的 Godunov 数值通量.

    对单调通量 f(u), Godunov flux 退化为迎风格式:
      F(ul, ur) = f(ul)  (通量始终从左侧流入, 因为 f'≥0 波速>0)

    参数:
        ul: 左侧状态 (或 tensor)
        ur: 右侧状态 (或 tensor)
    返回:
        Godunov 数值通量 F(ul, ur)
    """
    # 对单调增通量, Godunov flux = f(ul) (迎风格式)
    # 无论 ul>ur (shock) 还是 ul<ur (rarefaction), 信息从左侧来
    return bl_flux(ul)
