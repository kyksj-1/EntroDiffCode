import torch
import math
from tqdm import tqdm
from src.diffusion.losses import pde_residual

def entrodiff_heun_sampler(model, shape, sigma_min, sigma_max, tau_max, nu, num_steps=50, device="cpu", zeta_obs=0.0, zeta_pde=0.0, target_T=0.5):
    """
    Reverse-time sampler with Godunov-form guidance. (Algorithm 1 / Eq. 3.8)
    Implements Heun second-order integration with physical time scaling.
    """
    with torch.no_grad():
        # tau goes from target diffusion time tau_max down to 0
        tau_steps = torch.linspace(tau_max, 0, num_steps + 1, device=device)
        
        # Init sample from noise prior: u_T_d ~ N(0, sigma^2(T_d)*I) where sigma(tau) = sqrt(2*nu*tau)
        u_tau = torch.randn(shape, device=device) * math.sqrt(2.0 * nu * tau_max)
        
        dx = 2.0 * math.pi / shape[-1]

        for i in tqdm(range(num_steps), desc="Heun Sampling"):
            tau_t = tau_steps[i]
            tau_next = tau_steps[i+1]
            
            if tau_t == 0:
                break
                
            sigma_t = math.sqrt(2.0 * nu * tau_t)
            sigma_next = math.sqrt(2.0 * nu * float(tau_next)) if tau_next > 0 else float(sigma_min)
            
            # Evaluates \dot{\sigma}(\tau) = \nu / \sqrt{2\nu\tau}
            sigma_dot_t = nu / sigma_t
            
            # 1. Denoised state and score prediction at t
            D_u = model(u_tau, torch.tensor(sigma_t).expand(shape[0]).to(device))
            score_t = (D_u - u_tau) / (sigma_t ** 2)

            # Evaluate guidance residuals
            # NOTE (Algorithm 1 Deviation / MVP): 
            # The paper (Alg 1) dictates \nabla_u L_PDE^Godunov(u) as the descendent direction. 
            # Here we use the physical residual itself `pde_residual(u)` as a guidance directional proxy, 
            # which intuitively pushes `u` towards a PDE-satisfying state along the vector field directly. 
            # A rigorous implementation strictly following Eq. 3.8 requires computing the topological gradient:
            #   loss_pde = pde_residual(u_tau).norm()
            #   grad_u = torch.autograd.grad(outputs=loss_pde, inputs=u_tau)[0]
            l_pde_t = pde_residual(u_tau, dx) # Godunov PDE guidance proxy
            l_obs_t = 0.0 # Optional observation guidance

            # ODE drift: Eq. 3.8 / Alg 1
            d_t = -sigma_t * sigma_dot_t * score_t - zeta_obs * l_obs_t - zeta_pde * l_pde_t

            # Euler Step
            u_next = u_tau + d_t * (tau_next - tau_t)
            
            # 2. Heun Correction Step (if tau_next > 0)
            if tau_next > 0:
                sigma_dot_next = nu / sigma_next
                D_u_next = model(u_next, torch.tensor(sigma_next).expand(shape[0]).to(device))
                score_next = (D_u_next - u_next) / (sigma_next ** 2)
                
                l_pde_next = pde_residual(u_next, dx)
                d_next = -sigma_next * sigma_dot_next * score_next - zeta_obs * l_obs_t - zeta_pde * l_pde_next
                
                # Correcting Euler with Trapz rule
                u_next = u_tau + 0.5 * (d_t + d_next) * (tau_next - tau_t)

            u_tau = u_next

        return u_tau
