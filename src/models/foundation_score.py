# ============================================================================
# Foundation Score Model: DiT-Plain (W5-C)
#
# 论文对应: §3.2 Standard EDM parameterization (但 backbone 升级为 DiT-1D)
# 工程方案: Docs/black/W5_foundation_engineering_plan.md §2.3.1
#
# 与 StandardScore 的唯一区别:
#   - backbone: UNet1D → DiT1D
#   - forward 多接受 pde_id 参数 (mixed-PDE 训练时透传)
#   - 其他: EDM precondition 完全一致 (c_skip / c_out / c_in / c_noise)
#
# 兼容性铁律 (W5 plan §0.1):
#   - 输出 D_x: (B, 1, Nx) Tweedie 反演, 与 sampler / loss 完全兼容
#   - x_input 接受 channel-concat (noisy_u + IC + ...) 与 StandardScore 一致
#   - pde_id=None 时单 PDE 退化, 兼容现有训练 / eval 脚本
# ============================================================================

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.dit_1d import DiT1D


class FoundationScore(nn.Module):
    """
    DiT-Plain: 基础模型 baseline (与 StandardScore 同接口, 不走 BV-aware 分解).

    Args:
        in_channels:    输入通道数 (默认 2 = noisy_u + IC; W5 plan §1.1)
        Nx:             空间序列长度 (必须能被 patch_size 整除)
        dit_kwargs:     透传给 DiT1D 的架构参数 dict, 必含
                            dim, n_layers, n_heads, patch_size
                        可选:
                            n_pde_types (默认 1), dropout (默认 0.0)
        sigma_data:     EDM precondition 的数据标准差 (与 StandardScore 默认 0.5 一致)

    Forward signature (与 StandardScore / BVAwareScore 接口契约一致):
        forward(x, sigma, pde_id=None) → (B, 1, Nx)
        x:      (B, in_channels, Nx)
        sigma:  (B,) raw σ
        pde_id: (B,) long 或 None
    """

    def __init__(
        self,
        in_channels: int = 2,
        Nx: int = 128,
        dit_kwargs: Optional[dict] = None,
        sigma_data: float = 0.5,
    ) -> None:
        super().__init__()
        # dit_kwargs 解包 + 默认值兜底 (避免 None 调用)
        dit_kwargs = dict(dit_kwargs or {})  # 浅拷贝, 不污染调用方
        # 必填字段校验
        for required_key in ("dim", "n_layers", "n_heads", "patch_size"):
            if required_key not in dit_kwargs:
                raise KeyError(f"FoundationScore.dit_kwargs 缺少必填字段 '{required_key}'")
        # 可选字段默认值
        n_pde_types = dit_kwargs.pop("n_pde_types", 1)
        dropout = dit_kwargs.pop("dropout", 0.0)

        # 输出 1 通道 (与 StandardScore 一致, IC 仅作条件输入)
        self.net = DiT1D(
            in_channels=in_channels,
            out_channels=1,
            Nx=Nx,
            n_pde_types=n_pde_types,
            dropout=dropout,
            **dit_kwargs,  # 剩余 dim / n_layers / n_heads / patch_size
        )
        self.sigma_data = sigma_data
        # 保存供 introspection / save_load
        self.in_channels = in_channels
        self.Nx = Nx
        self.n_pde_types = n_pde_types

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        pde_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        EDM precondition (与 StandardScore.forward 完全一致):
            c_skip = σ_d² / (σ² + σ_d²)
            c_out  = σ·σ_d / sqrt(σ² + σ_d²)
            c_in   = 1 / sqrt(σ_d² + σ²)
            c_noise= log σ / 4   (内置在 DiT1D.t_embedder 中, 不需要这里再做)
            D_x = c_skip * x[:, :1] + c_out * F_θ(c_in * x, sigma, pde_id)

        Args:
            x:      (B, in_channels, Nx)  channel-concat 输入
            sigma:  (B,) raw σ (不取 log; DiT1D.t_embedder 内部做 log/4)
            pde_id: (B,) long 或 None

        Returns:
            D_x: (B, 1, Nx)
        """
        # ---- EDM precondition 系数 ----
        # 与 StandardScore.forward 逐字节对齐 (复用现有 sampler / loss 的预期)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2) ** 0.5
        # 注意: c_noise 由 DiT1D.t_embedder 内部计算 (sigma → log(σ)/4 → sin/cos),
        #       不需要在外层再次 log

        # ---- Forward 网络 (sigma 直接传 raw σ) ----
        # 输入 c_in[:, None, None] * x: (B, in_channels, Nx)
        F_x = self.net(c_in[:, None, None] * x, sigma, pde_id=pde_id)

        # ---- Tweedie 反演到去噪器输出 (与 StandardScore 一致) ----
        # 仅对 noisy_u 通道 (x[:, :1]) 做 skip connection
        # IC 通道仅作条件, 不参与 c_skip
        D_x = c_skip[:, None, None] * x[:, :1, :] + c_out[:, None, None] * F_x
        return D_x
