import torch
import torch.nn as nn
from typing import Optional
from src.models.unet_1d import UNet1D
from src.models.dit_1d import DiT1D   # W5-C: DiT 作为 phi_sm 子网的可选 backbone

class StandardScore(nn.Module):
    """
    Standard EDM parameterization: D_theta(x, sigma)
    predicts the denoised x_0, from which score is derived.

    Args:
        in_channels:  输入通道数 (noisy + conditioning)
        out_channels: 输出通道数, 即 noisy 分量数 (默认 1, 标量 PDE;
                      系统级 PDE 如 Euler 设 3)
        sigma_data:   EDM 数据标准差 (默认 0.5)
    """
    def __init__(self, in_channels=1, sigma_data=0.5, out_channels: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.net = UNet1D(in_channels=in_channels, out_channels=out_channels)
        self.sigma_data = sigma_data

    def forward(self, x, sigma, pde_id: Optional[torch.Tensor] = None):
        """
        EDM preconditions.
        c_skip * x + c_out * F_theta(c_in * x, c_noise(sigma))
        x 含 in_channels 通道 (noisy_u + IC), 仅前 out_channels 通道参与 skip connection

        pde_id: 兼容性参数 (W5-C 新增, 默认 None) — UNet 不消费, 仅供 mixed-PDE 训练时透传
        """
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2)**0.5
        c_in = 1 / (self.sigma_data**2 + sigma**2)**0.5
        c_noise = sigma.log() / 4.0

        F_x = self.net(c_in[:, None, None] * x, c_noise)

        # c_skip * x_0: 仅对 noisy 通道做 skip (IC 通道不参与)
        # 切前 out_channels 通道 (对应 noisy 部分, 非 conditioning)
        D_x = c_skip[:, None, None] * x[:, :self.out_channels, :] + c_out[:, None, None] * F_x
        return D_x

class BVAwareScore(nn.Module):
    """
    BV-aware Score Parameterization for EntroDiff Theory (§3.2 Method).
    严格遵循 Eq. 3.2:
    S_theta = grad(phi_sm) + (kappa/2) * tanh(phi_sh / (2*sigma^2)) * grad(phi_sh)

    参数:
        in_channels:  输入通道数 (默认 1; Burgers IC-cond=2; Euler IC-cond=6=3 noisy+3 IC)
        dim: UNet 基础通道数 (PC=64, 服务器=128~256)
        return_denoiser: True → 输出 D_x = x + σ²·s_θ (兼容现有 loss/sampler)
                        False → 输出裸 s_θ
        backbone:    'unet' (默认, 现有行为) | 'dit' (W5-C 新增, foundation model)
                     当 backbone='dit' 时, phi_sm_net 替换为 DiT1D, 其他 (phi_sh / kappa)
                     保持小 conv stack 不变 (它们是局部预测, 不需 attention 全局感受野).
        dit_kwargs:  当 backbone='dit' 时必传, 含 Nx, dim, n_layers, n_heads, patch_size,
                     n_pde_types (默认 1), dropout (默认 0.0).
        n_pde_types: ≥1; 与 dit_kwargs.n_pde_types 互斥 (优先以 dit_kwargs 为准).
        out_channels: 输出通道数 (W5-SA2 新增). 默认 1 (标量守恒律, e.g. Burgers/BL).
                     系统级 PDE (Euler=3 components) 可设 out_channels=3.
                     注意: D_x 切前 out_channels 通道做 Tweedie 反演, 输入前
                     out_channels 通道是 noisy 部分, 其余为 conditioning (e.g. IC).
    """
    def __init__(self, in_channels=1, dim=64, return_denoiser=True,
                 backbone: str = "unet",
                 dit_kwargs: Optional[dict] = None,
                 n_pde_types: int = 1,
                 out_channels: int = 1):
        super().__init__()
        self.return_denoiser = return_denoiser
        self.backbone = backbone   # 用于 forward 中 phi_sm 输入 sigma 编码方式分支
        self.n_pde_types = n_pde_types
        self.out_channels = out_channels   # W5-SA2: 系统级 PDE 支持

        # 1. Smooth background potential phi_sm
        #    out_channels: 与 noisy 通道数一致 (标量=1, Euler=3)
        if backbone == "unet":
            # 现有行为 (默认): UNet1D 输出 1 通道 (单 PDE) 或 out_channels 通道 (系统)
            self.phi_sm_net = UNet1D(
                in_channels=in_channels, out_channels=out_channels, dim=dim,
            )
        elif backbone == "dit":
            # W5-C 新增: DiT-1D 升级 phi_sm 子网
            if dit_kwargs is None:
                raise ValueError("backbone='dit' 时 dit_kwargs 必填 (含 Nx/dim/n_layers/n_heads/patch_size)")
            dit_kw = dict(dit_kwargs)   # 浅拷贝, 不污染调用方
            for required_key in ("Nx", "dim", "n_layers", "n_heads", "patch_size"):
                if required_key not in dit_kw:
                    raise KeyError(f"BVAwareScore(backbone='dit') dit_kwargs 缺少 '{required_key}'")
            # n_pde_types 优先以 dit_kwargs 为准, 缺省时用 BVAwareScore.n_pde_types
            dit_n_pde = dit_kw.pop("n_pde_types", n_pde_types)
            dit_dropout = dit_kw.pop("dropout", 0.0)
            self.phi_sm_net = DiT1D(
                in_channels=in_channels, out_channels=out_channels,
                n_pde_types=dit_n_pde, dropout=dit_dropout,
                **dit_kw,   # 剩余 Nx/dim/n_layers/n_heads/patch_size
            )
        else:
            raise ValueError(f"未知 backbone: {backbone}, 仅支持 'unet' | 'dit'")

        # 2. Shock signed distance phi_sh (论文 §3.2 role 2)
        #    输出 out_channels 通道 (系统级时每 component 一个 phi_sh)
        #    保留 conv stack: shock 检测是局部行为
        self.phi_sh_net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, out_channels, kernel_size=3, padding=1)
        )

        # 3. Jump amplitude kappa (论文 §3.2 role 3)
        #    out_channels 通道 (系统级时每 component 各自的跳幅)
        self.kappa_net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(16, out_channels, kernel_size=1),
            nn.Softplus()
        )

    def forward(self, x, sigma, pde_id: Optional[torch.Tensor] = None):
        """
        实现 Eq. 3.2 的建筑先验 (architectural prior).

        前向过程:
          1. x.requires_grad_(True) → 启用对输入的梯度追踪
          2. 计算三个子网络: phi_sm, phi_sh, kappa
          3. tanh(phi_sh / (2σ²)) → 编码 interfacial layer 的精确解析形式
          4. autograd.grad → 计算 ∇_u phi_sm 和 ∇_u phi_sh
          5. s_θ = ∇φ_sm + (κ/2)·tanh(φ_sh/(2σ²))·∇φ_sh  (Eq. 3.2)
          6. (可选) Tweedie 反演: D_x = x[:, :out_channels] + σ²·s_θ[:, :out_channels]

        关键: create_graph=True 是必须的 (loss 会对 s_θ 再次求导)
              inference 时用 torch.enable_grad() 突破外层 no_grad 限制

        pde_id: (B,) long 或 None — W5-C 新增, mixed-PDE 训练时透传给 DiT phi_sm.
                backbone='unet' 时忽略此参数, 保持单 PDE 行为.
        """
        # 保存原始噪声输入 (Tweedie 反演需要)
        x_noisy = x

        # 突破外层 torch.no_grad() (sampler 包裹) 以启用 autograd
        # 训练时外层无 no_grad, enable_grad 是空操作
        with torch.enable_grad():
            # 启用输入梯度 (对 x 原位修改, 不影响外部计算图)
            x.requires_grad_(True)

            # 2. Forward pass for potentials
            #    UNet:  传 c_noise = log(σ)/4   (现有行为)
            #    DiT:   传 raw σ (DiT1D.t_embedder 内部做 log/4 + sincos + MLP)
            if self.backbone == "unet":
                phi_sm = self.phi_sm_net(x, sigma.log() / 4.0)  # (B, out_C, Nx)
            else:
                # W5-C: DiT 接受 raw σ + pde_id; sigma → log/4 在 DiT1D.t_embedder 中
                phi_sm = self.phi_sm_net(x, sigma, pde_id=pde_id)  # (B, out_C, Nx)

            phi_sh = self.phi_sh_net(x)                      # (B, out_C, Nx)
            kappa = self.kappa_net(x) + 1e-4                 # (B, out_C, Nx), +ϵ 防除零

            # tanh 剖面: 直接嵌入网络结构
            tanh_factor = torch.tanh(
                phi_sh / (2 * (sigma**2).view(-1, 1, 1) + 1e-6)
            )

            # 3. 真梯度
            grad_phi_sm = torch.autograd.grad(
                phi_sm.sum(), x, create_graph=True
            )[0]
            grad_phi_sh = torch.autograd.grad(
                phi_sh.sum(), x, create_graph=True
            )[0]

            # 4. Eq. 3.2: 光滑背景 + 界面 tanh 层
            #    注意: grad_phi_sm 和 grad_phi_sh shape == x shape == (B, in_C, Nx)
            #          tanh_factor / kappa shape (B, out_C, Nx)
            #    系统级时 out_C < in_C (因 in_C 含 IC), 需要把 tanh_factor 广播或切片
            if self.out_channels != grad_phi_sm.shape[1]:
                # 切前 out_channels 通道 (与 noisy 部分对齐)
                grad_phi_sm = grad_phi_sm[:, :self.out_channels, :]
                grad_phi_sh = grad_phi_sh[:, :self.out_channels, :]
            s_theta = (
                grad_phi_sm
                + (kappa / 2.0) * tanh_factor * grad_phi_sh
            )

        # 5. Tweedie 反演到去噪器输出 (兼容 EDM loss / Heun sampler)
        #    D_x = x_noisy[:, :out_C] + σ² · s_θ[:, :out_C]
        #
        # W5-C 修复 (2026-05-01): 只对前 out_channels 通道做 Tweedie 反演.
        #   - 默认 out_channels=1 时, 与 StandardScore 输出 shape 一致 (B, 1, Nx)
        #   - 系统级 out_channels=3 时, 输出 (B, 3, Nx) 对应 Euler 三组分
        #   - ckpt-compatible: 模型权重不变, 仅输出切片
        if self.return_denoiser:
            D_x = (
                x_noisy[:, :self.out_channels, :]
                + (sigma**2).view(-1, 1, 1) * s_theta
            )
            return D_x
        # 不反演时仍返回完整 s_theta (debug / 高级用途)
        return s_theta
