# ============================================================================
# Post-hoc BV-aware Score Wrapper (方案 B · 冻结 backbone, 仅训 phi_sh + kappa)
#
# 论文对应: §5.6 修正叙事 — "Plug-and-play BV-aware modulation, fine-tuned"
#
# 设计:
#   1. plain_backbone: 已训 dit_plain 的 FoundationScore, 完全冻结 (不参与梯度)
#   2. phi_sh_net + kappa_net: 两个轻量 conv1d, ~50K 参数, 从头训
#   3. forward:
#        D_plain = plain_backbone(x, σ, pde_id)            [冻结, ~7.4M]
#        s_plain = (D_plain - x_noisy) / σ²                [按 EDM 反演]
#        phi_sh = phi_sh_net(x_input)                       [训]
#        kappa  = kappa_net(x_input) + 1e-4                 [训]
#        s_total = s_plain + (κ/2)·tanh(φ_sh/2σ²)·∇φ_sh    [post-hoc 修正]
#        D_total = x_noisy + σ²·s_total                     [Tweedie 反演]
#
# 训练:
#   - 仅优化 phi_sh_net + kappa_net 参数
#   - L_DSM(D_total, x_target) 单一 loss (不加 BV/time)
#   - 50 epoch 即可 (小网络, 数据高效)
#
# 兼容性:
#   - forward signature 与 StandardScore/BVAwareScore/FoundationScore 一致
#   - 接受 IC-conditioning + pde_id
#   - 输出 (B, 1, Nx) D_x, 与 sampler/loss 兼容
# ============================================================================

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.foundation_score import FoundationScore


class PostHocBVAwareScore(nn.Module):
    """
    Post-hoc BV-aware: 在 frozen plain backbone 上加可训的 phi_sh + kappa 修正.

    Args:
        plain_backbone:   已训的 FoundationScore (会被冻结)
        in_channels:      与 plain_backbone 一致 (默认 2 = noisy_u + IC)
        sigma_data:       EDM precondition 数据标准差 (默认 0.5)

    Forward:
        x:      (B, in_channels, Nx)
        sigma:  (B,) raw σ
        pde_id: (B,) long 或 None
    Returns:
        D_x: (B, 1, Nx) — Tweedie 反演后的去噪输出, 与 sampler/loss 兼容
    """

    def __init__(
        self,
        plain_backbone: FoundationScore,
        in_channels: int = 2,
        sigma_data: float = 0.5,
        phi_sh_dim: int = 32,           # W5 ext: 增大版用 64
        kappa_dim: int = 16,            # W5 ext: 增大版用 32
        depth: int = 2,                 # W5 ext: 默认 2 层 (= 小版); 3 层 = 大版
    ) -> None:
        super().__init__()
        self.plain_backbone = plain_backbone
        # 冻结 plain backbone 全部参数
        for p in self.plain_backbone.parameters():
            p.requires_grad_(False)
        self.plain_backbone.eval()                  # 关 dropout / BN running stats

        self.in_channels = in_channels
        self.sigma_data = sigma_data
        self.phi_sh_dim = phi_sh_dim
        self.kappa_dim = kappa_dim
        self.depth = depth

        # phi_sh: signed distance to shock
        if depth == 2:
            # 默认小版 (~321 params at dim=32)
            self.phi_sh_net = nn.Sequential(
                nn.Conv1d(in_channels, phi_sh_dim, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(phi_sh_dim, 1, kernel_size=3, padding=1),
            )
        else:
            # 大版 3 层 (~6.7K params at phi_sh_dim=64)
            self.phi_sh_net = nn.Sequential(
                nn.Conv1d(in_channels, phi_sh_dim, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(phi_sh_dim, phi_sh_dim // 2, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(phi_sh_dim // 2, 1, kernel_size=3, padding=1),
            )

        # kappa: jump amplitude
        if depth == 2:
            self.kappa_net = nn.Sequential(
                nn.Conv1d(in_channels, kappa_dim, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(kappa_dim, 1, kernel_size=1),
                nn.Softplus(),
            )
        else:
            self.kappa_net = nn.Sequential(
                nn.Conv1d(in_channels, kappa_dim, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(kappa_dim, kappa_dim // 2, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(kappa_dim // 2, 1, kernel_size=1),
                nn.Softplus(),
            )

        # 初始化为 small (修正初始为 0, 训练初期 = plain backbone)
        for m in [self.phi_sh_net, self.kappa_net]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv1d):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        pde_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:      (B, in_channels, Nx) — noisy_u + IC
            sigma:  (B,) raw σ
            pde_id: (B,) long 或 None
        Returns:
            D_x:    (B, 1, Nx)
        """
        # ---- 1. Plain backbone 给出 D_plain (冻结, 无梯度) ----
        with torch.no_grad():
            D_plain = self.plain_backbone(x, sigma, pde_id=pde_id)   # (B, 1, Nx)

        # ---- 2. 由 D_plain 反推 plain score ----
        x_noisy = x[:, :1, :]                                          # (B, 1, Nx)
        sigma_sq = (sigma ** 2).view(-1, 1, 1)                         # (B, 1, 1)
        s_plain = (D_plain - x_noisy) / (sigma_sq + 1e-8)               # (B, 1, Nx)

        # ---- 3. phi_sh + kappa (可训) ----
        # 启用梯度追踪 (不论外层是 train 还是 eval)
        with torch.enable_grad():
            x_for_grad = x.clone().detach().requires_grad_(True)        # (B, in_C, Nx)
            phi_sh = self.phi_sh_net(x_for_grad)                        # (B, 1, Nx)
            kappa = self.kappa_net(x_for_grad) + 1e-4                   # (B, 1, Nx)

            # ∇_x phi_sh (沿空间 x 的梯度, 但这里 phi_sh 也是空间网格, 用 autograd 拿对 x 的梯度)
            # 注意: x_for_grad shape (B, in_C, Nx), phi_sh shape (B, 1, Nx)
            # 我们要的 ∇φ_sh 是 phi_sh 对空间维度的梯度. 用中心差分更直接 (与 score_param.py 的 autograd 不同).
            # 这里因为 phi_sh 仅是 x 的小网络函数, 用空间中心差分效率更高.
            #
            # 简化策略: 直接用空间中心差分 (周期边界), 与 sampler 中的 detect_shock 一致
            # ∇φ_sh[i] ≈ (φ_sh[i+1] - φ_sh[i-1]) / (2 dx)
            Nx = phi_sh.shape[-1]
            dx = 2.0 * 3.141592653589793 / Nx
            phi_sh_left = torch.roll(phi_sh, shifts=1, dims=2)
            phi_sh_right = torch.roll(phi_sh, shifts=-1, dims=2)
            grad_phi_sh = (phi_sh_right - phi_sh_left) / (2.0 * dx)     # (B, 1, Nx)

        # ---- 4. tanh interfacial profile (论文 Eq. 3.2) ----
        tanh_factor = torch.tanh(phi_sh / (2.0 * sigma_sq + 1e-6))       # (B, 1, Nx)

        # ---- 5. BV-aware 修正项 ----
        bv_correction = (kappa / 2.0) * tanh_factor * grad_phi_sh        # (B, 1, Nx)

        # ---- 6. 总 score ----
        s_total = s_plain + bv_correction                                # (B, 1, Nx)

        # ---- 7. Tweedie 反演 → D_x ----
        D_x = x_noisy + sigma_sq * s_total                               # (B, 1, Nx)
        return D_x

    def trainable_parameters(self):
        """返回仅可训的参数 (phi_sh + kappa, 不含 plain backbone)."""
        return list(self.phi_sh_net.parameters()) + list(self.kappa_net.parameters())
