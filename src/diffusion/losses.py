import torch
import torch.nn.functional as F

def get_dsm_loss(model, x, sigma: torch.Tensor):
    """
    Denoising Score Matching loss (Standard EDM formulation).
    min E_{epsilon, x} || D_theta(x + sigma*epsilon, sigma) - x ||_2^2
    """
    noise = torch.randn_like(x)
    sigma = sigma.view(-1, 1, 1).to(x.device)
    x_noisy = x + sigma * noise

    # Denoised prediction
    D_x = model(x_noisy, sigma.squeeze())

    # L2 score matching loss
    loss = (D_x - x) ** 2
    
    # Optional weighting could be added here; EDM default weight is 1.0 (or something complex)
    # We take simple MSE for MVP.
    return loss.mean()

def get_bv_loss(model, x, sigma: torch.Tensor):
    r"""
    Total-Variation (TV) Penalty (Section 3.3 \mathcal{L}_{BV}).
    Limits the search space to BV(Omega) to guarantee L1 compactness under Helly's theorem.
    NOTE: MVP stage implements this BV proxy explicitly. True Kruzhkov entropy loss (\mathcal{L}_{ent})
    requires integration over test constants and time derivatives, to be added in future iterations.
    """
    noise = torch.randn_like(x)
    sigma = sigma.view(-1, 1, 1).to(x.device)
    x_noisy = x + sigma * noise
    u_hat = model(x_noisy, sigma.squeeze())  # predicted state

    u_diff = u_hat[:, :, 1:] - u_hat[:, :, :-1]
    tv_loss = torch.abs(u_diff).mean()

    return tv_loss

def get_godunov_time_loss(model, x_prev, x_target, sigma, dt, dx):
    """
    时间一致性损失 (轻量版): 强制去噪输出的一步 Godunov 推进接近真值.
    
    论文对应: Theorem 1 的时间连续性 + Claim 2 的物理一致性
    数学: L_time = E[||Godunov_step(D_θ(x_prev+σε, σ)) - x_target||²]
    
    参数:
        x_prev:  干净的前一帧 [B, 1, Nx] (t = t_{n-1})
        x_target: 干净的目标帧 [B, 1, Nx] (t = t_n)
        sigma:  噪声水平 [B]
        dt, dx: 时间步长和空间步长
    
    返回: MSE(time_prediction, target)
    """
    # 1. 加噪声: x_prev_noisy = x_prev + σ*ε
    noise = torch.randn_like(x_prev)
    sigma = sigma.view(-1, 1, 1).to(x_prev.device)
    x_prev_noisy = x_prev + sigma * noise

    # 2. 去噪: û_prev = D_θ(x_prev_noisy, σ)
    D_prev = model(x_prev_noisy, sigma.squeeze())  # [B, 1, Nx]

    # 3. Godunov 一步推进: û_next = û_prev - dt/dx * (F_{i+1/2} - F_{i-1/2})
    #    对 û_prev 的每个 batch 逐单元计算 Godunov flux
    ul = D_prev[:, :, :-1]        # 左状态 u_L
    ur = D_prev[:, :, 1:]          # 右状态 u_R
    flux_interior = godunov_flux(ul, ur)  # 内点通量 [B, 1, Nx-1]

    # 周期边界: u_{N-1} → u_0 的通量
    flux_boundary = godunov_flux(D_prev[:, :, -1:], D_prev[:, :, :1])  # [B, 1, 1]

    # 拼接通量: [F_{N-1→0}, F_{0→1}, ..., F_{Nx-2→Nx-1}]
    flux_full = torch.cat([flux_boundary, flux_interior], dim=2)  # [B, 1, Nx]

    # 散度: -dt/dx * (F_i - F_{i-1}) (周期平移)
    flux_shifted = torch.roll(flux_full, shifts=1, dims=2)  # F_{i-1}
    D_next = D_prev - (dt / dx) * (flux_full - flux_shifted)

    # 4. MSE: ||û_next - x_target||²
    loss = (D_next - x_target) ** 2
    return loss.mean()


def godunov_flux(ul, ur):
    """
    Inviscid Godunov flux for f(u) = 0.5*u^2
    Exact formula for convex flux without sonic point approximations.
    """
    f_shock = torch.max(0.5 * ul**2, 0.5 * ur**2)
    f_rarefaction = torch.where((ul <= 0.0) & (ur >= 0.0), torch.zeros_like(ul), torch.min(0.5 * ul**2, 0.5 * ur**2))
    
    f = torch.where(ul >= ur, f_shock, f_rarefaction)
    return f

def pde_residual(u, dx: float):
    """
    Godunov spatial residual: - (f_{i+1/2} - f_{i-1/2}) / dx
    """
    ul = u[:, :, :-1]
    ur = u[:, :, 1:]
    fluxes = godunov_flux(ul, ur)
    # Add periodic boundaries
    flux_0 = godunov_flux(u[:, :, -1:], u[:, :, :1])
    
    flux_full = torch.cat([flux_0, fluxes, flux_0], dim=2)
    div_f = (flux_full[:, :, 1:] - flux_full[:, :, :-1]) / dx
    return div_f
