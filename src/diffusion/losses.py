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
