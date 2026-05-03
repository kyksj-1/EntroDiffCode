import torch
import torch.nn.functional as F

def get_dsm_loss(model, x, sigma: torch.Tensor, conditioning=None, ic=None, pde_id=None):
    """
    Denoising Score Matching loss (Standard EDM formulation).
    min E_{epsilon, x} || D_theta(x + sigma*epsilon, sigma) - x ||_2^2

    conditioning: 条件张量 [B, C_cond, Nx] — 拼接到带噪输入 (推荐新接口)
    ic: 向后兼容旧参数名 (同 conditioning)
    pde_id: (B,) long 或 None (W5-C 新增, mixed-PDE 训练时透传给 model;
            backbone='unet' 的 model 会忽略此参数)
    """
    # 向后兼容: ic= 参数优先使用新名 conditioning=
    cond = conditioning if conditioning is not None else ic
    noise = torch.randn_like(x)
    sigma = sigma.view(-1, 1, 1).to(x.device)
    x_noisy = x + sigma * noise

    # 条件通道: 将 conditioning 拼接到带噪输入的第 1 通道轴 (dim=1)
    x_input = torch.cat([x_noisy, cond], dim=1) if cond is not None else x_noisy

    # Denoised prediction (W5-C: 透传 pde_id; 单 PDE 训练时为 None)
    D_x = model(x_input, sigma.squeeze(), pde_id=pde_id)

    loss = (D_x - x) ** 2
    return loss.mean()

def get_bv_loss(model, x, sigma: torch.Tensor, conditioning=None, ic=None, pde_id=None):
    r"""
    Total-Variation (TV) Penalty (Section 3.3 \mathcal{L}_{BV}).
    conditioning: 条件张量 — 拼接到带噪输入 (推荐新接口)
    ic: 向后兼容旧参数名
    pde_id: W5-C 新增, mixed-PDE 训练时透传给 model
    """
    cond = conditioning if conditioning is not None else ic
    noise = torch.randn_like(x)
    sigma = sigma.view(-1, 1, 1).to(x.device)
    x_noisy = x + sigma * noise
    x_input = torch.cat([x_noisy, cond], dim=1) if cond is not None else x_noisy
    u_hat = model(x_input, sigma.squeeze(), pde_id=pde_id)

    u_diff = u_hat[:, :, 1:] - u_hat[:, :, :-1]
    tv_loss = torch.abs(u_diff).mean()
    return tv_loss

def get_godunov_time_loss(model, x_prev, x_target, sigma, dt, dx, conditioning=None, ic=None,
                          pde_id=None, flux_type: str = "burgers"):
    """
    时间一致性损失 (轻量版): 强制去噪输出的一步 Godunov 推进接近真值.
    conditioning: 条件张量 — 拼接到带噪输入
    ic: 向后兼容旧参数名
    pde_id: (B,) long 或 None (W5-C 新增) — 当前实现未按 pde_id 分桶 flux,
            而是用 flux_type 字符串选定 (整 batch 同一种 PDE 训练时使用).
            未来 W5-D Phase 2 实现 batch 内异质 flux 分桶时再扩展.
    flux_type: 'burgers' | 'buckley_leverett' | ... — 决定调用哪个 godunov_flux.
    """
    cond = conditioning if conditioning is not None else ic
    noise = torch.randn_like(x_prev)
    sigma = sigma.view(-1, 1, 1).to(x_prev.device)
    x_prev_noisy = x_prev + sigma * noise
    x_input = torch.cat([x_prev_noisy, cond], dim=1) if cond is not None else x_prev_noisy
    D_prev = model(x_input, sigma.squeeze(), pde_id=pde_id)

    # W5-C: 通过 flux_type 派遣 (默认 'burgers' 维持现有行为)
    flux_fn = _resolve_flux_fn(flux_type)

    # 3. Godunov 一步推进: û_next = û_prev - dt/dx * (F_{i+1/2} - F_{i-1/2})
    #    对 û_prev 的每个 batch 逐单元计算 Godunov flux
    ul = D_prev[:, :, :-1]        # 左状态 u_L
    ur = D_prev[:, :, 1:]          # 右状态 u_R
    flux_interior = flux_fn(ul, ur)  # 内点通量 [B, 1, Nx-1]

    # 周期边界: u_{N-1} → u_0 的通量
    flux_boundary = flux_fn(D_prev[:, :, -1:], D_prev[:, :, :1])  # [B, 1, 1]

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
    Inviscid Godunov flux for f(u) = 0.5*u^2  (Burgers).
    Exact formula for convex flux without sonic point approximations.
    """
    f_shock = torch.max(0.5 * ul**2, 0.5 * ur**2)
    f_rarefaction = torch.where((ul <= 0.0) & (ur >= 0.0), torch.zeros_like(ul), torch.min(0.5 * ul**2, 0.5 * ur**2))

    f = torch.where(ul >= ur, f_shock, f_rarefaction)
    return f


def godunov_flux_bl(ul, ur):
    """
    Buckley-Leverett 非凸通量的 Godunov flux (W5-C 新增).
    f(u) = u^2 / (u^2 + (1-u)^2),  u ∈ [0, 1].

    复用 src/pdes/bl_flux.py 已有实现, 在此包一层让接口与 godunov_flux 对齐 (ul, ur) → f.
    """
    # 延迟导入避免循环依赖 (src.pdes 加载时不需要 src.diffusion)
    from src.pdes.bl_flux import bl_godunov_flux
    return bl_godunov_flux(ul, ur)


def godunov_flux_euler_density(ul, ur):
    """
    Step 5: Euler 密度通道 (ρ) 的 Godunov flux (代理).

    严格 Euler 密度方程: ρ_t + (ρu)_x = 0
    单独看 ρ 通道时缺少 u, 无法严格守恒律. 用 Burgers Godunov 作代理:
    ρ 当作"速度"处理. 这仅作为 time loss 的物理一致性正则项, 不影响主 W₁.

    论文 §A2 注明 "approximate Godunov flux for Euler density channel".
    """
    return godunov_flux(ul, ur)


# ---- W5-C: flux 派遣表 (从 flux_type 字符串路由到对应函数) ----
# 设计: 字典查表; 新增 PDE flux 仅在此表追加项, 不需改 get_godunov_time_loss
# Step 5 (2026-05-04): 加 'euler_density' 支持 Euler ρ 通道 1D 混训
_FLUX_REGISTRY = {
    "burgers": godunov_flux,
    "buckley_leverett": godunov_flux_bl,
    "euler_density": godunov_flux_euler_density,
}


def _resolve_flux_fn(flux_type: str):
    """根据 flux_type 字符串返回对应 godunov flux 函数."""
    if flux_type not in _FLUX_REGISTRY:
        available = list(_FLUX_REGISTRY.keys())
        raise KeyError(f"未知 flux_type='{flux_type}', 可用: {available}")
    return _FLUX_REGISTRY[flux_type]


def pde_residual(u, dx: float, flux_type: str = "burgers"):
    """
    Godunov spatial residual: - (f_{i+1/2} - f_{i-1/2}) / dx

    flux_type: W5-C 新增, 默认 'burgers' 维持现有行为.
    """
    flux_fn = _resolve_flux_fn(flux_type)
    ul = u[:, :, :-1]
    ur = u[:, :, 1:]
    fluxes = flux_fn(ul, ur)
    # Add periodic boundaries
    flux_0 = flux_fn(u[:, :, -1:], u[:, :, :1])

    flux_full = torch.cat([flux_0, fluxes, flux_0], dim=2)
    div_f = (flux_full[:, :, 1:] - flux_full[:, :, :-1]) / dx
    return div_f
