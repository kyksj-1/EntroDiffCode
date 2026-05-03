# ============================================================================
# Post-hoc BV-aware Sampler (方案 A · Training-free)
#
# 论文对应: §5.6 修正叙事 — "BV-aware as plug-and-play modulation"
#
# 核心思想:
#   已训 EDM Standard backbone (e.g., dit_plain / FoundationScore) 不变.
#   采样时在反向 ODE 内动态加入 tanh shock 修正:
#     s_total = s_std + λ(σ) · (κ_local / 2) · tanh(d_shock / (2 σ²)) · sign(d_shock)
#   其中:
#     s_std       = (D_θ(u, σ) - u) / σ²    [由 plain backbone 给出]
#     d_shock(x)  = x - x_shock              [x 到检测到的 shock 位置的 signed distance]
#     x_shock     = argmax_x |∂_x D_x|        [从 denoised D_x 检测的 shock 位置]
#     κ_local     = |D_x(x_shock⁺) - D_x(x_shock⁻)|   [shock 跳跃强度]
#     λ(σ)        = clip( 1 / (1 + 5σ), 0, 1)        [小 σ 强, 大 σ 弱; 避免压扁窗口]
#
# 与现有 sampler (entrodiff_heun_sampler) 的差异:
#   - 在每个 Heun 半步后, 给 score 加 BV-aware 修正项再积分
#   - 完全 inference-time, 不修改 backbone weights
#   - 周期边界处理: shock 位置 + signed distance 在 [0, 2π] 周期意义下
#
# 兼容性:
#   - 接受任意 EDM-style score model (StandardScore / FoundationScore / BVAwareScore 都行)
#   - 接受 IC-conditioning (cond=None 时退化为 unconditional)
#   - 接受 mixed-PDE (pde_id 透传)
# ============================================================================

import math
from typing import Optional

import torch
import torch.nn.functional as F


def detect_shock_loc_and_jump(D_x: torch.Tensor, dx: float = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 denoised prediction D_x 检测 shock 位置 + 跳跃强度.

    Args:
        D_x:  (B, 1, Nx) — denoised solution
        dx:   空间步长 (默认 2π/Nx)

    Returns:
        shock_idx:   (B,) long — argmax(|∂_x D_x|) 的 grid index
        kappa_local: (B,) float — |D_x[+1] - D_x[-1]| at shock_idx
    """
    B, _, Nx = D_x.shape
    if dx is None:
        dx = 2.0 * math.pi / Nx

    # 中心差分梯度 (周期边界)
    D_left = torch.roll(D_x, shifts=1, dims=2)         # (B, 1, Nx)
    D_right = torch.roll(D_x, shifts=-1, dims=2)
    grad_D = (D_right - D_left) / (2.0 * dx)            # (B, 1, Nx)

    # 每 batch 找梯度绝对值最大的位置
    grad_abs = grad_D.abs().squeeze(1)                  # (B, Nx)
    shock_idx = grad_abs.argmax(dim=1)                  # (B,)

    # 跳跃强度: |D_x[i+1] - D_x[i-1]|
    kappa_local = torch.zeros(B, device=D_x.device, dtype=D_x.dtype)
    for b in range(B):
        i = shock_idx[b].item()
        i_plus = (i + 1) % Nx
        i_minus = (i - 1) % Nx
        kappa_local[b] = (D_x[b, 0, i_plus] - D_x[b, 0, i_minus]).abs()
    return shock_idx, kappa_local


def signed_distance_to_shock(shock_idx: torch.Tensor, Nx: int, device, dtype=torch.float32) -> torch.Tensor:
    """
    周期边界下的 signed distance to shock.

    Args:
        shock_idx: (B,) long — shock 的 grid index
        Nx:        网格数

    Returns:
        d_shock: (B, 1, Nx) float — 每位置到 shock 的 signed distance (周期意义),
                 范围 [-π, π]
    """
    B = shock_idx.shape[0]
    # x 网格 [0, 2π) 但对 shock 来说计算的是 grid 间距相对位置
    grid_idx = torch.arange(Nx, device=device, dtype=torch.float32).unsqueeze(0).expand(B, Nx)  # (B, Nx)
    sh = shock_idx.float().unsqueeze(1)                                                          # (B, 1)
    d_idx = grid_idx - sh                                                                        # (B, Nx)
    # 周期最短距离: 把 d_idx 映射到 [-Nx/2, Nx/2]
    d_idx = (d_idx + Nx // 2) % Nx - Nx // 2
    # 转换为物理距离: dx · idx_diff
    dx = 2.0 * math.pi / Nx
    d_shock = (d_idx * dx).unsqueeze(1).to(dtype=dtype)   # (B, 1, Nx)
    return d_shock


def lambda_schedule(sigma: float, mode: str = "exp_decay") -> float:
    """
    BV-aware 修正强度的 σ-依赖调制.

    设计目标:
      - 小 σ: λ ≈ 1 (BV-aware 强发挥, shock 已成形)
      - 大 σ: λ ≈ 0 (避免在大噪声端添加 spurious modulation)

    模式:
      'exp_decay': λ(σ) = exp(-α σ),   α=2.0
      'inv_linear': λ(σ) = 1/(1 + 5σ)
      'sigmoid':   λ(σ) = sigmoid(-(σ-1)·3)
    """
    if mode == "exp_decay":
        return math.exp(-2.0 * sigma)
    elif mode == "inv_linear":
        return max(0.0, min(1.0, 1.0 / (1.0 + 5.0 * sigma)))
    elif mode == "sigmoid":
        return float(torch.sigmoid(torch.tensor(-(sigma - 1.0) * 3.0)).item())
    else:
        raise ValueError(f"Unknown lambda mode: {mode}")


def posthoc_bv_score_correction(
    D_x: torch.Tensor,
    sigma: float,
    bv_strength: float = 1.0,
    lambda_mode: str = "exp_decay",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    给定 plain backbone 的 D_x, 计算 BV-aware 修正项 (要加到 score 上).

    Args:
        D_x:         (B, 1, Nx)
        sigma:       float, 当前噪声水平
        bv_strength: float, 全局 BV 修正强度系数 (论文超参, 默认 1.0)
        lambda_mode: λ(σ) 调制模式

    Returns:
        s_correction: (B, 1, Nx) float — 加到 s_std 上的修正
    """
    B, _, Nx = D_x.shape
    device = D_x.device
    dtype = D_x.dtype

    # 1. 从 D_x 检测 shock 位置 + 跳跃强度
    shock_idx, kappa_local = detect_shock_loc_and_jump(D_x)         # (B,), (B,)

    # 2. signed distance to shock (周期)
    d_shock = signed_distance_to_shock(shock_idx, Nx, device, dtype)  # (B, 1, Nx)

    # 3. tanh interfacial profile (论文 Eq. 3.2)
    #    tanh(d / (2σ²))
    sigma_sq = max(sigma ** 2, eps)
    tanh_factor = torch.tanh(d_shock / (2.0 * sigma_sq))               # (B, 1, Nx)

    # 4. ∇d_shock = sign(d_shock) (signed distance 的梯度方向就是 sign)
    sign_d = torch.sign(d_shock)                                       # (B, 1, Nx)

    # 5. BV term: (κ_local / 2) · tanh · sign(d)
    #    论文 Eq. 3.2: s_θ = ∇φ_sm + (κ/2)·tanh(φ_sh/2σ²)·∇φ_sh
    #    这里 ∇φ_sm 由 backbone 给, BV term 即 (κ_local/2)·tanh·sign
    kappa_b = kappa_local.view(B, 1, 1)                                # (B, 1, 1)
    bv_term = (kappa_b / 2.0) * tanh_factor * sign_d                   # (B, 1, Nx)

    # 6. λ(σ) 调制
    lam = lambda_schedule(sigma, mode=lambda_mode)
    s_correction = bv_strength * lam * bv_term                          # (B, 1, Nx)

    return s_correction


def posthoc_bv_heun_sampler(
    model,
    shape,
    sigma_min,
    sigma_max,
    tau_max,
    nu,
    num_steps=50,
    device="cpu",
    zeta_obs=0.0,
    zeta_pde=0.0,
    conditioning=None,
    ic=None,
    pde_id: Optional[torch.Tensor] = None,
    flux_type: str = "burgers",
    bv_strength: float = 1.0,
    lambda_mode: str = "exp_decay",
):
    """
    Heun reverse sampler with post-hoc BV-aware score modulation.

    与 entrodiff_heun_sampler 的差异:
      在每个 Heun 半步前, 计算 D_x → 检测 shock → 加 BV-aware 修正给 score.

    Args:
        bv_strength:  BV 修正全局强度 (0 → 退化为 standard sampler)
        lambda_mode:  λ(σ) 调制模式 ('exp_decay' / 'inv_linear' / 'sigmoid')
        其他参数: 与 entrodiff_heun_sampler 一致

    Returns:
        u_final: (B, 1, Nx) 生成的解
    """
    cond = conditioning if conditioning is not None else ic
    has_cond = cond is not None

    with torch.no_grad():
        tau_steps = torch.linspace(tau_max, 0, num_steps + 1, device=device)
        u_tau = torch.randn(shape, device=device) * math.sqrt(2.0 * nu * tau_max)

        for i in range(num_steps):
            tau_t = tau_steps[i]
            tau_next = tau_steps[i + 1]
            if tau_t == 0:
                break

            sigma_t = math.sqrt(2.0 * nu * tau_t.item())
            sigma_t_tensor = torch.tensor([sigma_t], device=device).expand(shape[0])
            sigma_next = math.sqrt(2.0 * nu * max(float(tau_next), 1e-6))
            sigma_dot_t = nu / sigma_t

            # ---- Plain backbone forward ----
            u_input = torch.cat([u_tau, cond], dim=1) if has_cond else u_tau
            D_u = model(u_input, sigma_t_tensor, pde_id=pde_id)
            score_std = (D_u - u_tau) / (sigma_t ** 2)

            # ---- Post-hoc BV-aware 修正 ----
            if bv_strength > 0:
                s_correction = posthoc_bv_score_correction(
                    D_u, sigma_t, bv_strength=bv_strength, lambda_mode=lambda_mode
                )
                score_t = score_std + s_correction
            else:
                score_t = score_std

            # ---- ODE drift ----
            d_t = -sigma_t * sigma_dot_t * score_t

            u_next = u_tau + d_t * (tau_next - tau_t)

            # ---- Heun corrector ----
            if tau_next > 0:
                sigma_dot_next = nu / sigma_next
                u_n_input = torch.cat([u_next, cond], dim=1) if has_cond else u_next
                D_u_next = model(u_n_input,
                                 torch.tensor([sigma_next], device=device).expand(shape[0]),
                                 pde_id=pde_id)
                score_std_next = (D_u_next - u_next) / (sigma_next ** 2)

                if bv_strength > 0:
                    s_corr_next = posthoc_bv_score_correction(
                        D_u_next, sigma_next, bv_strength=bv_strength, lambda_mode=lambda_mode
                    )
                    score_next = score_std_next + s_corr_next
                else:
                    score_next = score_std_next

                d_next = -sigma_next * sigma_dot_next * score_next
                u_next = u_tau + 0.5 * (d_t + d_next) * (tau_next - tau_t)

            u_tau = u_next

        return u_tau
