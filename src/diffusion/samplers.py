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

            # BVAwareScore 内部 enable_grad 会修改 u_tau 的 requires_grad 状态,
            # 导致后续 PDE guidance 的 autograd 冲突. detach 确保干净计算图.
            u_tau = u_tau.detach()

            # ===== Godunov PDE Guidance (Algorithm 1 / Eq. 3.8) =====
            # 论文: l_pde = ∇_u ‖pde_residual(u)‖² 作为 guidance direction
            if zeta_pde > 0:
                with torch.enable_grad():
                    u_tau_grad = u_tau.clone().requires_grad_(True)
                    res = pde_residual(u_tau_grad, dx)
                    loss_pde = res.pow(2).mean()
                    l_pde_t = torch.autograd.grad(loss_pde, u_tau_grad)[0]
                    # 梯度裁剪: Godunov 残差对随机噪声可达 ~200,
                    # 梯度可轻易超过 500, 需要归一化到合理范围
                    grad_norm = l_pde_t.norm(p=2, dim=(1,2), keepdim=True) + 1e-6
                    max_norm = 1.0  # 限制每步 guidance 最大方向幅度
                    scale = torch.clamp(max_norm / grad_norm, max=1.0)
                    l_pde_t = l_pde_t * scale
            else:
                # zeta_pde = 0 → 关闭 PDE guidance，跳过 costly autograd
                l_pde_t = torch.zeros_like(u_tau)
            # zeta_obs 当前实验设为 0 (无观测引导)
            l_obs_t = torch.zeros_like(u_tau)

            # ODE drift: Eq. 3.8 / Alg 1
            # dτ 方向上 u 的漂移速度 = -σ·σ̇·s_θ - ζ_obs·l_obs - ζ_pde·∇_u L_PDE
            d_t = -sigma_t * sigma_dot_t * score_t - zeta_obs * l_obs_t - zeta_pde * l_pde_t

            # Euler Step
            u_next = u_tau + d_t * (tau_next - tau_t)
            
            # 2. Heun Correction Step (if tau_next > 0)
            if tau_next > 0:
                sigma_dot_next = nu / sigma_next
                D_u_next = model(u_next, torch.tensor(sigma_next).expand(shape[0]).to(device))
                score_next = (D_u_next - u_next) / (sigma_next ** 2)
                
                # 真梯度 PDE guidance for Heun correction (同 Euler 步骤逻辑)
                if zeta_pde > 0:
                    with torch.enable_grad():
                        u_next_grad = u_next.clone().requires_grad_(True)
                        res_next = pde_residual(u_next_grad, dx)
                        loss_pde_next = res_next.pow(2).mean()
                        l_pde_next = torch.autograd.grad(loss_pde_next, u_next_grad)[0]
                        # 同 Euler 步: 梯度裁剪防止 NaN
                        grad_norm_n = l_pde_next.norm(p=2, dim=(1,2), keepdim=True) + 1e-6
                        scale_n = torch.clamp(1.0 / grad_norm_n, max=1.0)
                        l_pde_next = l_pde_next * scale_n
                else:
                    l_pde_next = torch.zeros_like(u_next)
                d_next = -sigma_next * sigma_dot_next * score_next - zeta_obs * l_obs_t - zeta_pde * l_pde_next
                
                # Correcting Euler with Trapz rule
                u_next = u_tau + 0.5 * (d_t + d_next) * (tau_next - tau_t)

            u_tau = u_next

        return u_tau
