# ============================================================================
# E2 Buckley-Leverett 独立评估脚本 (自包含, 不依赖 Foundation 代码)
# 部署方式: scp 到服务器, CUDA_VISIBLE_DEVICES=X python 直接跑
# ============================================================================
import os, sys, math, glob
import numpy as np
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 轻量级 UNet1D (不需要 import 服务器代码)
# ============================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, time):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=time.device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class Block1D(nn.Module):
    def __init__(self, ic, oc, ted):
        super().__init__()
        self.tmlp = nn.Linear(ted, oc)
        self.c1 = nn.Conv1d(ic, oc, 3, padding=1)
        self.c2 = nn.Conv1d(oc, oc, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, oc)
        self.gn2 = nn.GroupNorm(8, oc)
        self.act = nn.SiLU()
    def forward(self, x, t):
        h = self.act(self.gn1(self.c1(x)))
        h = h + self.tmlp(t).unsqueeze(-1)
        return self.act(self.gn2(self.c2(h)))

class UNet1D(nn.Module):
    def __init__(self, ic=1, oc=1, dim=64):
        super().__init__()
        td = dim * 4
        self.tmlp = nn.Sequential(SinusoidalPositionEmbeddings(dim),
                                   nn.Linear(dim, td), nn.SiLU(), nn.Linear(td, td))
        self.icv = nn.Conv1d(ic, dim, 3, padding=1)
        self.d1 = Block1D(dim, dim, td); self.d2 = Block1D(dim, dim*2, td); self.d3 = Block1D(dim*2, dim*4, td)
        self.m1 = Block1D(dim*4, dim*4, td); self.m2 = Block1D(dim*4, dim*4, td)
        self.u1 = Block1D(dim*4+dim*4, dim*2, td); self.u2 = Block1D(dim*2+dim*2, dim, td); self.u3 = Block1D(dim+dim, dim, td)
        self.fc = nn.Conv1d(dim, oc, 1)
    def forward(self, x, t):
        te = self.tmlp(t)
        x0 = self.icv(x)
        d1 = self.d1(x0, te); d1p = F.avg_pool1d(d1, 2)
        d2 = self.d2(d1p, te); d2p = F.avg_pool1d(d2, 2)
        d3 = self.d3(d2p, te); d3p = F.avg_pool1d(d3, 2)
        m = self.m1(d3p, te); m = self.m2(m, te)
        u = F.interpolate(m, scale_factor=2, mode='linear', align_corners=False)
        u = self.u1(torch.cat([u, d3], 1), te)
        u = F.interpolate(u, scale_factor=2, mode='linear', align_corners=False)
        u = self.u2(torch.cat([u, d2], 1), te)
        u = F.interpolate(u, scale_factor=2, mode='linear', align_corners=False)
        u = self.u3(torch.cat([u, d1], 1), te)
        return self.fc(u)

# ============================================================
# StandardScore (EDM preconditioning)
# ============================================================
class StandardScore(nn.Module):
    def __init__(self, ic=1, dim=64):
        super().__init__()
        self.net = UNet1D(ic, ic, dim)
        self.sd = 0.5
    def forward(self, x, sigma):
        s2 = sigma**2 + self.sd**2
        c_skip = self.sd**2 / s2
        c_out = sigma * self.sd / s2**0.5
        c_in = 1 / s2**0.5
        c_noise = sigma.log() / 4.0
        Fx = self.net(c_in[:, None, None] * x, c_noise)
        return c_skip[:, None, None] * x + c_out[:, None, None] * Fx

# ============================================================
# BVAwareScore (简化版, Eq. 3.2: s = ∇φ_sm + (κ/2)·tanh(φ_sh/2σ²)·∇φ_sh)
# ============================================================
class BVAwareScore(nn.Module):
    def __init__(self, ic=1, dim=64):
        super().__init__()
        self.phi_sm_net = UNet1D(ic, ic, dim)
        self.phi_sh_net = nn.Sequential(nn.Conv1d(ic, 32, 3, padding=1), nn.SiLU(), nn.Conv1d(32, ic, 3, padding=1))
        self.kappa_net = nn.Sequential(nn.Conv1d(ic, 16, 1), nn.SiLU(), nn.Conv1d(16, ic, 1), nn.Softplus())
    def forward(self, x, sigma):
        x_in = x
        with torch.enable_grad():
            x = x.requires_grad_(True)
            phi_sm = self.phi_sm_net(x, sigma.log() / 4.0)
            phi_sh = self.phi_sh_net(x)
            kappa = self.kappa_net(x) + 1e-4
            tanh_f = torch.tanh(phi_sh / (2 * sigma.pow(2).view(-1, 1, 1) + 1e-6))
            g_sm = torch.autograd.grad(phi_sm.sum(), x, create_graph=True)[0]
            g_sh = torch.autograd.grad(phi_sh.sum(), x, create_graph=True)[0]
            s_theta = g_sm + (kappa / 2.0) * tanh_f * g_sh
        # Tweedie: D_x = x + σ²·s_theta
        return x_in + sigma.pow(2).view(-1, 1, 1) * s_theta

# ============================================================
# Godunov Flux (Burgers: f(u) = u²/2, BL: f(u) = u²/(u²+(1-u)²))
# ============================================================
def godunov_flux_burgers(ul, ur):
    fs = torch.max(0.5*ul**2, 0.5*ur**2)
    fr = torch.where((ul <= 0) & (ur >= 0), torch.zeros_like(ul), torch.min(0.5*ul**2, 0.5*ur**2))
    return torch.where(ul >= ur, fs, fr)

def godunov_flux_bl(ul, ur):
    # Buckley-Leverett: f(u) = u²/(u²+(1-u)²) for u in [0,1]
    def f(u): return u**2 / (u**2 + (1-u)**2 + 1e-10)
    # Approximate Godunov: take max/min of f for shock/rarefaction
    fl, fr = f(ul), f(ur)
    f_shock = torch.max(fl, fr)
    f_rare = torch.where(ul <= ur, torch.min(fl, fr), torch.zeros_like(ul))
    return torch.where(ul >= ur, f_shock, f_rare)

def pde_residual(u, dx, flux_fn=godunov_flux_burgers):
    ul, ur = u[:, :, :-1], u[:, :, 1:]
    fluxes = flux_fn(ul, ur)
    fb = flux_fn(u[:, :, -1:], u[:, :, :1])
    flux_full = torch.cat([fb, fluxes, fb], dim=2)
    return (flux_full[:, :, 1:] - flux_full[:, :, :-1]) / dx

# ============================================================
# Heun Sampler (简化版, 不依赖外部代码)
# ============================================================
def heun_sample(model, shape, sig_max, sig_min, tau_max, nu, n_steps, device, zeta_pde=0, dx=None, flux_fn=None):
    tau_steps = torch.linspace(tau_max, 0, n_steps + 1, device=device)
    u = torch.randn(shape, device=device) * sig_max
    if dx is None: dx = 2 * math.pi / shape[-1]
    
    for i in range(n_steps):
        tau_t, tau_n = tau_steps[i], tau_steps[i+1]
        if tau_t == 0: break
        sig_t = math.sqrt(2*nu*tau_t)
        sig_n = math.sqrt(2*nu*float(tau_n)) if tau_n > 0 else sig_min
        sd_t = nu / sig_t
        
        D = model(u, torch.tensor(sig_t, device=device).expand(shape[0]))
        score = (D - u) / sig_t**2
        
        # PDE guidance (optional)
        l_pde = torch.zeros_like(u)
        if zeta_pde > 0 and flux_fn:
            with torch.enable_grad():
                ug = u.clone().requires_grad_(True)
                r = pde_residual(ug, dx, flux_fn)
                loss = r.pow(2).mean()
                grad = torch.autograd.grad(loss, ug)[0]
                gn = grad.norm(p=2, dim=(1,2), keepdim=True) + 1e-6
                l_pde = grad * torch.clamp(1.0/gn, max=1.0)
        
        d = -sig_t * sd_t * score - zeta_pde * l_pde
        u_n = u + d * (tau_n - tau_t)
        
        if tau_n > 0:
            sd_n = nu / sig_n
            Dn = model(u_n, torch.tensor(sig_n, device=device).expand(shape[0]))
            sn = (Dn - u_n) / sig_n**2
            l_pn = torch.zeros_like(u_n)
            if zeta_pde > 0 and flux_fn:
                with torch.enable_grad():
                    ug2 = u_n.clone().requires_grad_(True)
                    r2 = pde_residual(ug2, dx, flux_fn)
                    l2 = r2.pow(2).mean()
                    g2 = torch.autograd.grad(l2, ug2)[0]
                    gn2 = g2.norm(p=2, dim=(1,2), keepdim=True) + 1e-6
                    l_pn = g2 * torch.clamp(1.0/gn2, max=1.0)
            dn = -sig_n * sd_n * sn - zeta_pde * l_pn
            u_n = u + 0.5 * (d + dn) * (tau_n - tau_t)
        u = u_n.detach()
    return u

# ============================================================
# 主评估
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='BL data .npy path')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint path')
    parser.add_argument('--name', type=str, default='E2_MODEL')
    parser.add_argument('--ic', type=int, default=1, help='input channels (1=无IC, 2=含IC)')
    parser.add_argument('--dim', type=int, default=64, help='UNet dim')
    parser.add_argument('--steps', type=str, default='10,25,50,100')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--nu', type=float, default=1.0)
    parser.add_argument('--zeta', type=float, default=0.0)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    step_list = [int(s) for s in args.steps.split(',')]

    # Load BL data
    data = np.load(args.data)
    n_total = data.shape[0]
    test_start = int(0.9 * n_total)
    gt = data[test_start:test_start + args.n_samples, -1, :]  # last time step
    nx = gt.shape[1]
    dx = 1.0 / nx  # BL domain is [0,1]
    print(f'Data: {args.data}, test={gt.shape}, nx={nx}')

    # Load model
    sd = torch.load(args.ckpt, map_location=device)
    # Auto-detect model type from state_dict keys
    model = StandardScore(ic=args.ic, dim=args.dim).to(device)
    
    # Check if BVAwareScore (has phi_sm_net keys) or StandardScore (has net keys)
    is_bv = any(k.startswith('phi_sm_net.') for k in sd.keys())
    if is_bv:
        model = BVAwareScore(ic=args.ic, dim=args.dim).to(device)
        print(f'Detected: BVAwareScore')
    else:
        print(f'Detected: StandardScore')
    
    # Strip wrapper prefixes
    stripped = {}
    for k, v in sd.items():
        for prefix in ['base.', 'module.']:
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        stripped[k] = v
    model.load_state_dict(stripped)
    model.eval()
    print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')

    sigma_max = math.sqrt(2 * args.nu * 1.0)
    flux_fn = godunov_flux_bl if 'bl' in args.data.lower() else godunov_flux_burgers

    print(f'\n=== {args.name} ===')
    for ns in step_list:
        gen = heun_sample(model, (args.n_samples, args.ic, nx),
                         sigma_max, 0.002, 1.0, args.nu, ns, device,
                         zeta_pde=args.zeta, dx=dx, flux_fn=flux_fn)
        gen = gen.squeeze().cpu().numpy()
        if gen.ndim == 3: gen = gen[:, 0, :]  # take first channel if multi-channel
        w1s = [wasserstein_distance(gt[i], gen[i]) for i in range(args.n_samples)]
        l1s = [np.linalg.norm(gen[i] - gt[i], 1) / (np.linalg.norm(gt[i], 1) + 1e-10) for i in range(args.n_samples)]
        print(f'{ns}steps: W1={np.mean(w1s):.4f} L1={np.mean(l1s):.4f}')
    print('DONE')

if __name__ == '__main__':
    main()
