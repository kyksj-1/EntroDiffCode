# ============================================================================
# Learned Lambda Schedule for PostHoc-A v2
#
# 论文对应: §5.6 plug-and-play modulation 升级 — 自适应 σ + IC 的修正强度
#
# 设计动机:
#   PostHoc-A v1 用固定 λ(σ) = exp(-2σ); 不同 setting 最佳 bv_strength 不同 (0.5/1.0/2.0)
#   v2: 用小 MLP 学 λ(σ, IC_features), 单一方法在 5/5 setting 全胜 Plain
#
# 输入:
#   sigma: (B,) raw σ
#   ic:    (B, 1, Nx) IC, 用 4 维全局特征 (mean / std / max_grad / TV)
# 输出:
#   lam:   (B,) λ ∈ [0, lam_max], Sigmoid 保证有界
#
# 与现有 posthoc_bv_sampler 的兼容性:
#   - 旧: posthoc_bv_score_correction(D_x, sigma, bv_strength=lam_scalar, lambda_mode='exp_decay')
#   - 新: posthoc_bv_score_correction_v2(D_x, sigma, ic, lambda_module)  [本文件配套]
#   - sampler 接受可选 lambda_module 参数, 不传则退回 v1 行为
#
# 训练:
#   - 仅训 LearnedLambdaSchedule (~8K params)
#   - Plain backbone + shock detector 全冻结
#   - L_DSM only (不加 BV/time)
# ============================================================================

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigma_sincos_embedding(sigma: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    σ → sin/cos embedding, 与 DiT TimestepEmbedder 风格一致.

    Args:
        sigma:      (B,) raw σ > 0
        dim:        embedding 维度 (偶数)
        max_period: 频率范围

    Returns:
        (B, dim) embedding
    """
    # log/4 归一, 与 EDM/DiT 一致
    c_noise = sigma.log() / 4.0  # (B,)

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=sigma.device)  # (half,)
    args = c_noise[:, None].float() * freqs[None]  # (B, half)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def ic_global_features(ic: torch.Tensor, dx: float = None) -> torch.Tensor:
    """
    从 IC 提取 4 维全局特征 (mean / std / max_grad / TV).

    设计动机:
        - mean: 平均场强度, 影响 shock 强度
        - std: 振幅水平, 影响 shock 形成时间
        - max|∇ic|: shock 锐度的早期信号
        - TV(ic)/Nx: 全局变差, 与 BV 范数对齐 (论文 BV-aware 与之协同)

    Args:
        ic: (B, 1, Nx)
        dx: 空间步长, None 时取 2π/Nx
    Returns:
        (B, 4)
    """
    B, _, Nx = ic.shape
    if dx is None:
        dx = 2.0 * math.pi / Nx

    # 1. mean
    ic_mean = ic.mean(dim=(1, 2))  # (B,)
    # 2. std
    ic_std = ic.std(dim=(1, 2))    # (B,)
    # 3. max|∇ic| (中心差分, 周期边界)
    ic_left = torch.roll(ic, shifts=1, dims=2)
    ic_right = torch.roll(ic, shifts=-1, dims=2)
    grad = (ic_right - ic_left) / (2.0 * dx)
    max_grad = grad.abs().amax(dim=(1, 2))  # (B,)
    # 4. TV / Nx (单步差分总变差 / Nx, 类似 BV/length)
    tv = (ic_right - ic).abs().sum(dim=(1, 2)) / Nx  # (B,)

    feats = torch.stack([ic_mean, ic_std, max_grad, tv], dim=-1)  # (B, 4)
    return feats


class LearnedLambdaSchedule(nn.Module):
    """
    可学 λ(σ, IC) → λ ∈ [0, lam_max].

    Args:
        hidden:    隐藏维度 (默认 64)
        freq_dim:  σ 的 sin/cos 频率维度 (默认 64)
        lam_max:   λ 的上界 (默认 5.0; 与现有 bv_strength 范围对齐)
        zero_init: True → 训前 head 输出 0, lambda = lam_max * sigmoid(0) = lam_max/2 (≈2.5)
                   False → 默认 init, 输出依赖随机权重

    Forward:
        sigma: (B,) raw σ
        ic:    (B, 1, Nx) IC
    Returns:
        lam:   (B,) λ ≥ 0
    """

    def __init__(
        self,
        hidden: int = 64,
        freq_dim: int = 64,
        lam_max: float = 5.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        assert freq_dim % 2 == 0, "freq_dim 必须偶数"
        self.hidden = hidden
        self.freq_dim = freq_dim
        self.lam_max = lam_max

        # σ embedding
        self.sigma_emb = nn.Sequential(
            nn.Linear(freq_dim, hidden),
            nn.SiLU(),
        )
        # IC summary 维度 (固定 4)
        self.ic_summary_dim = 4
        # 合并 head
        self.head = nn.Sequential(
            nn.Linear(hidden + self.ic_summary_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        # 初始化
        if zero_init:
            # 让 head 最后一层输出 0 → sigmoid(0)=0.5 → λ=lam_max/2
            # 这与 PostHoc-A v1 的 exp_decay 中段值接近 (σ=1 时 λ≈0.13, λ_max=5 取中段过大)
            # 实际上 zero_init 让初始 λ = 2.5, 略大于 v1 中段, 训练初期更激进
            # 若想精确对齐 v1 行为, 用 init_to_v1 方法 (见下)
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    @torch.no_grad()
    def init_to_v1_behavior(
        self,
        sigma_range: tuple[float, float] = (0.05, 1.5),
        n_samples: int = 100,
        lr: float = 1e-3,
        steps: int = 500,
    ):
        """
        预训练 head 让 LearnedLambdaSchedule(σ, ic) ≈ exp(-2σ) on σ ∈ sigma_range.
        IC 用零张量 (训练时 ic 通道差异由后续 fine-tune 学到).

        这是可选的"暖启动", 让 v2 训练初期与 v1 行为一致.
        """
        # 暂用简单 exp_decay 作 target
        device = next(self.parameters()).device
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        # 启用梯度
        with torch.enable_grad():
            for step in range(steps):
                sigmas = torch.empty(n_samples, device=device).uniform_(*sigma_range)
                # IC 全 0 (Nx=128 默认, 可扩展)
                ic_dummy = torch.zeros(n_samples, 1, 128, device=device)
                target = torch.exp(-2.0 * sigmas)  # (n,) ∈ (0, 1) for σ ∈ [0.05, 1.5]
                pred = self.forward(sigmas, ic_dummy)
                loss = ((pred - self.lam_max * target) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        # 训完不返回梯度

    def forward(self, sigma: torch.Tensor, ic: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma: (B,) raw σ > 0
            ic:    (B, 1, Nx)
        Returns:
            lam: (B,) ∈ [0, lam_max]
        """
        s_freq = sigma_sincos_embedding(sigma, self.freq_dim)   # (B, freq_dim)
        s_emb = self.sigma_emb(s_freq)                            # (B, hidden)
        ic_feat = ic_global_features(ic)                          # (B, 4)
        merged = torch.cat([s_emb, ic_feat], dim=-1)              # (B, hidden+4)
        out = self.head(merged).squeeze(-1)                       # (B,)
        # Sigmoid 限定 [0, 1] → × lam_max → [0, lam_max]
        lam = self.lam_max * torch.sigmoid(out)
        return lam


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
