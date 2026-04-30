import torch
import math
from src.diffusion.losses import pde_residual

def entrodiff_heun_sampler(model, shape, sigma_min, sigma_max, tau_max, nu,
                           num_steps=50, device="cpu", zeta_obs=0.0, zeta_pde=0.0, ic=None):
    """
    Reverse-time Heun sampler with Godunov-form PDE guidance (Algorithm 1).
    若 ic 不为 None: 拼接作为条件通道 (IC-conditioned inference).
    """
    with torch.no_grad():
        tau_steps = torch.linspace(tau_max, 0, num_steps + 1, device=device)
        u_tau = torch.randn(shape, device=device) * math.sqrt(2.0 * nu * tau_max)
        has_ic = ic is not None
        dx = 2.0 * math.pi / shape[-1]

        for i in range(num_steps):
            tau_t = tau_steps[i]
            tau_next = tau_steps[i + 1]
            if tau_t == 0:
                break

            sigma_t = math.sqrt(2.0 * nu * tau_t.item())
            sigma_t_tensor = torch.tensor([sigma_t], device=device).expand(shape[0])
            sigma_next = math.sqrt(2.0 * nu * max(float(tau_next), 1e-6))
            sigma_dot_t = nu / sigma_t

            # Denoised prediction
            u_input = torch.cat([u_tau, ic], dim=1) if has_ic else u_tau
            D_u = model(u_input, sigma_t_tensor)
            score_t = (D_u - u_tau) / (sigma_t ** 2)

            # PDE guidance (optional, zeta_pde=0 时跳过)
            if zeta_pde > 0:
                with torch.enable_grad():
                    u_grad = u_tau.clone().detach().requires_grad_(True)
                    res = pde_residual(u_grad, dx)
                    loss_pde = res.pow(2).mean()
                    l_pde_t = torch.autograd.grad(loss_pde, u_grad)[0]
                    norm = l_pde_t.norm(p=2, dim=(1, 2), keepdim=True) + 1e-6
                    l_pde_t = l_pde_t * torch.clamp(1.0 / norm, max=1.0)
            else:
                l_pde_t = torch.zeros_like(u_tau)

            # ODE drift
            d_t = -sigma_t * sigma_dot_t * score_t \
                  - zeta_obs * torch.zeros_like(u_tau) \
                  - zeta_pde * l_pde_t

            # Euler predictor
            u_next = u_tau + d_t * (tau_next - tau_t)

            # Heun corrector
            if tau_next > 0:
                sigma_dot_next = nu / sigma_next
                u_n_input = torch.cat([u_next, ic], dim=1) if has_ic else u_next
                D_u_next = model(u_n_input, torch.tensor([sigma_next], device=device).expand(shape[0]))
                score_next = (D_u_next - u_next) / (sigma_next ** 2)

                if zeta_pde > 0:
                    with torch.enable_grad():
                        ug = u_next.clone().detach().requires_grad_(True)
                        rn = pde_residual(ug, dx)
                        lpn = torch.autograd.grad(rn.pow(2).mean(), ug)[0]
                        nm = lpn.norm(p=2, dim=(1, 2), keepdim=True) + 1e-6
                        l_pde_next = lpn * torch.clamp(1.0 / nm, max=1.0)
                else:
                    l_pde_next = torch.zeros_like(u_next)

                d_next = -sigma_next * sigma_dot_next * score_next \
                         - zeta_pde * l_pde_next
                u_next = u_tau + 0.5 * (d_t + d_next) * (tau_next - tau_t)

            u_tau = u_next

        return u_tau
