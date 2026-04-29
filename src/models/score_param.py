import torch
import torch.nn as nn
from src.models.unet_1d import UNet1D

class StandardScore(nn.Module):
    """
    Standard EDM parameterization: D_theta(x, sigma) 
    predicts the denoised x_0, from which score is derived.
    """
    def __init__(self, in_channels=1, sigma_data=0.5):
        super().__init__()
        self.net = UNet1D(in_channels=in_channels, out_channels=in_channels)
        self.sigma_data = sigma_data

    def forward(self, x, sigma):
        """
        EDM preconditions.
        c_skip * x + c_out * F_theta(c_in * x, c_noise(sigma))
        """
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2)**0.5
        c_in = 1 / (self.sigma_data**2 + sigma**2)**0.5
        c_noise = sigma.log() / 4.0

        F_x = self.net(c_in[:, None, None] * x, c_noise)
        
        D_x = c_skip[:, None, None] * x + c_out[:, None, None] * F_x
        return D_x

class BVAwareScore(nn.Module):
    """
    BV-aware Score Parameterization for EntroDiff Theory (§3.2 Method).
    严格遵循 Eq. 3.2:
    S_theta = grad(phi_sm) + (kappa/2) * tanh(phi_sh / (2*sigma^2)) * grad(phi_sh)
    
    参数:
        in_channels: 输入通道数 (默认 1)
        dim: UNet 基础通道数 (PC=64, 服务器=128~256)
        return_denoiser: True → 输出 D_x = x + σ²·s_θ (兼容现有 loss/sampler)
                        False → 输出裸 s_θ
    """
    def __init__(self, in_channels=1, dim=64, return_denoiser=True):
        super().__init__()
        self.return_denoiser = return_denoiser
        
        # 1. Smooth background potential phi_sm (UNet backbone, 论文 §3.2 role 1)
        #    dim 控制模型容量: PC=64, 服务器=128/256 以提升 expressivity
        self.phi_sm_net = UNet1D(in_channels=in_channels, out_channels=in_channels, dim=dim)
        
        # 2. Shock signed distance phi_sh (论文 §3.2 role 2)
        #    轻量 Conv1d 网络, 编码 shock 几何位置
        self.phi_sh_net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, in_channels, kernel_size=3, padding=1)
        )
        
        # 3. Jump amplitude kappa (论文 §3.2 role 3)
        #    κ ≥ κ₀ > 0 通过 Softplus 强制为正, Rankine-Hugoniot 条件提供物理值
        self.kappa_net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(16, in_channels, kernel_size=1),
            nn.Softplus()  # 强制 κ > 0
        )
        
    def forward(self, x, sigma):
        """
        实现 Eq. 3.2 的建筑先验 (architectural prior).

        前向过程:
          1. x.requires_grad_(True) → 启用对输入的梯度追踪
          2. 计算三个子网络: phi_sm, phi_sh, kappa
          3. tanh(phi_sh / (2σ²)) → 编码 interfacial layer 的精确解析形式
          4. autograd.grad → 计算 ∇_u phi_sm 和 ∇_u phi_sh
          5. s_θ = ∇φ_sm + (κ/2)·tanh(φ_sh/(2σ²))·∇φ_sh  (Eq. 3.2)
          6. (可选) Tweedie 反演: D_x = x + σ²·s_θ

        关键: create_graph=True 是必须的 (loss 会对 s_θ 再次求导)
              inference 时用 torch.enable_grad() 突破外层 no_grad 限制
        """
        # 保存原始噪声输入 (Tweedie 反演需要)
        x_noisy = x

        # 突破外层 torch.no_grad() (sampler 包裹) 以启用 autograd
        # 训练时外层无 no_grad, enable_grad 是空操作
        with torch.enable_grad():
            # 启用输入梯度 (对 x 原位修改, 不影响外部计算图)
            x.requires_grad_(True)

            # 2. Forward pass for potentials
            #    sigma.log()/4.0 是 EDM 标准的噪声编码 (c_noise)
            phi_sm = self.phi_sm_net(x, sigma.log() / 4.0)  # (B, C, Nx)
            phi_sh = self.phi_sh_net(x)                      # (B, C, Nx)
            kappa = self.kappa_net(x) + 1e-4                 # (B, C, Nx), +ϵ 防除零

            # tanh 剖面: 直接嵌入网络结构
            # σ → ∞ 时 tanh → 0: shock 分量消失, 只剩光滑背景
            # σ → 0 时 tanh → step: shock 尖峰自然浮现
            tanh_factor = torch.tanh(
                phi_sh / (2 * (sigma**2).view(-1, 1, 1) + 1e-6)
            )

            # 3. 真梯度
            #    create_graph=True: 训练时必须 (loss 对 s_θ 再求导)
            #    inference 时 create_graph=True 也无害 (多余但兼容)
            grad_phi_sm = torch.autograd.grad(
                phi_sm.sum(), x, create_graph=True
            )[0]
            grad_phi_sh = torch.autograd.grad(
                phi_sh.sum(), x, create_graph=True
            )[0]

            # 4. Eq. 3.2: 光滑背景 + 界面 tanh 层
            s_theta = (
                grad_phi_sm
                + (kappa / 2.0) * tanh_factor * grad_phi_sh
            )

        # 5. Tweedie 反演到去噪器输出 (兼容 EDM loss / Heun sampler)
        #    D_x = x_noisy + σ² · s_θ(x_noisy, σ)
        if self.return_denoiser:
            D_x = x_noisy + (sigma**2).view(-1, 1, 1) * s_theta
            return D_x
        return s_theta
