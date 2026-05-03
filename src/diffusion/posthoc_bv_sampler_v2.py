# ============================================================================
# Post-hoc BV-aware Sampler v2 — 兼容 LearnedLambdaSchedule
#
# 与 v1 (posthoc_bv_sampler.py) 的差异:
#   v1: λ(σ) = exp_decay/inv_linear/sigmoid 三种固定 schedule, lambda_mode 选
#   v2: 多接受一个 lambda_module: LearnedLambdaSchedule (可选)
#       lambda_module=None → 退回 v1 行为
#       lambda_module=instance → 用 lambda_module(σ, IC) 替代 fixed schedule
#
# 论文对应: §5.6 plug-and-play 升级 — 自适应 modulation
# ============================================================================

import math
from typing import Optional

import torch
import torch.nn.functional as F

# 复用 v1 的 shock 检测函数
from src.diffusion.posthoc_bv_sampler import (
    detect_shock_loc_and_jump,
    signed_distance_to_shock,
    lambda_schedule as fixed_lambda_schedule,
)


def posthoc_bv_score_correction_v2(
    D_x: torch.Tensor,
    sigma: float,
    ic: Optional[torch.Tensor] = None,
    bv_strength: float = 1.0,
    lambda_mode: str = "exp_decay",
    lambda_module=None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    v2 版 BV-aware 修正.

    Args:
        D_x:           (B, 1, Nx)
        sigma:         float, 当前噪声水平
        ic:            (B, 1, Nx) IC; lambda_module is not None 时必传
        bv_strength:   全局 BV 修正强度 (与 lambda_module 是 multiplicative; 通常 lambda_module 已含尺度)
        lambda_mode:   v1 的 schedule 模式 (lambda_module is None 时使用)
        lambda_module: LearnedLambdaSchedule 实例; None 退回 v1
    Returns:
        (B, 1, Nx)
    """
    B, _, Nx = D_x.shape
    device = D_x.device
    dtype = D_x.dtype

    # 1. 检测 shock + 跳跃强度 (v1 复用)
    shock_idx, kappa_local = detect_shock_loc_and_jump(D_x)
    # 2. signed distance (v1 复用)
    d_shock = signed_distance_to_shock(shock_idx, Nx, device, dtype)  # (B, 1, Nx)

    # 3. tanh 剖面
    sigma_sq = max(sigma ** 2, eps)
    tanh_factor = torch.tanh(d_shock / (2.0 * sigma_sq))  # (B, 1, Nx)
    sign_d = torch.sign(d_shock)                          # (B, 1, Nx)

    kappa_b = kappa_local.view(B, 1, 1)
    bv_term = (kappa_b / 2.0) * tanh_factor * sign_d  # (B, 1, Nx)

    # 4. λ(σ) 调制 — v2 关键差异
    if lambda_module is not None:
        # 学得的 λ(σ, IC), 必须传 ic
        if ic is None:
            raise ValueError("lambda_module is not None 时必须传 ic 参数")
        sigma_t = torch.tensor([sigma] * B, device=device)  # (B,)
        with torch.no_grad():
            # eval 模式下不需要梯度; 训练时由 train_learned_lambda 自己开 enable_grad
            lam_per_sample = lambda_module(sigma_t, ic)  # (B,)
        # broadcast (B,) → (B, 1, 1)
        lam = lam_per_sample.view(B, 1, 1)
    else:
        # 退回 v1: 标量 λ
        lam_scalar = fixed_lambda_schedule(sigma, mode=lambda_mode)
        lam = torch.tensor(lam_scalar, device=device, dtype=dtype).view(1, 1, 1)

    s_correction = bv_strength * lam * bv_term  # (B, 1, Nx)
    return s_correction


def posthoc_bv_heun_sampler_v2(
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
    lambda_module=None,                       # ← v2 新增
    enable_grad_for_lambda_module: bool = False,
):
    """
    v2 Heun reverse sampler with 可选 LearnedLambdaSchedule.

    Args:
        lambda_module: LearnedLambdaSchedule 或 None
        enable_grad_for_lambda_module: 训练 lambda_module 时设 True (让计算图保留),
                                        eval 时设 False
        其他: 同 v1 sampler

    与 v1 的关键区别:
        - 调 posthoc_bv_score_correction_v2 替代 v1 版
        - 训练 lambda_module 时 sampler 内的 score 计算需要 gradient (但 backbone forward 不要)
    """
    cond = conditioning if conditioning is not None else ic
    has_cond = cond is not None
    # ic for lambda_module: 优先用 cond (IC-conditioning 时 cond 就是 IC); fallback 用 ic 参数
    ic_for_lambda = cond if cond is not None else ic

    grad_context = torch.enable_grad if enable_grad_for_lambda_module else torch.no_grad

    with grad_context():
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

            # Plain backbone forward (无梯度, 即使训 lambda_module 也是)
            u_input = torch.cat([u_tau, cond], dim=1) if has_cond else u_tau
            with torch.no_grad():
                D_u = model(u_input, sigma_t_tensor, pde_id=pde_id)
            score_std = (D_u - u_tau) / (sigma_t ** 2)

            # v2 修正
            if bv_strength > 0:
                s_correction = posthoc_bv_score_correction_v2(
                    D_u, sigma_t, ic=ic_for_lambda,
                    bv_strength=bv_strength, lambda_mode=lambda_mode,
                    lambda_module=lambda_module,
                )
                score_t = score_std + s_correction
            else:
                score_t = score_std

            d_t = -sigma_t * sigma_dot_t * score_t
            u_next = u_tau + d_t * (tau_next - tau_t)

            # Heun corrector
            if tau_next > 0:
                sigma_dot_next = nu / sigma_next
                u_n_input = torch.cat([u_next, cond], dim=1) if has_cond else u_next
                with torch.no_grad():
                    D_u_next = model(u_n_input,
                                     torch.tensor([sigma_next], device=device).expand(shape[0]),
                                     pde_id=pde_id)
                score_std_next = (D_u_next - u_next) / (sigma_next ** 2)

                if bv_strength > 0:
                    s_corr_next = posthoc_bv_score_correction_v2(
                        D_u_next, sigma_next, ic=ic_for_lambda,
                        bv_strength=bv_strength, lambda_mode=lambda_mode,
                        lambda_module=lambda_module,
                    )
                    score_next = score_std_next + s_corr_next
                else:
                    score_next = score_std_next

                d_next = -sigma_next * sigma_dot_next * score_next
                u_next = u_tau + 0.5 * (d_t + d_next) * (tau_next - tau_t)

            u_tau = u_next

        return u_tau
