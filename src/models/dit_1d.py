# ============================================================================
# DiT-1D Backbone for EntroDiff Foundation Model (W5)
#
# 论文对应: §3.5 Foundation Model: DiT Backbone (Docs/black/path_A_method_skeleton.md)
# 工程方案: Docs/black/W5_foundation_engineering_plan.md §2.1
# 参考实现: facebookresearch/DiT (Peebles & Xie, ICCV 2023)
#           https://github.com/facebookresearch/DiT/blob/main/models.py (MIT License)
#
# 与官方 DiT 的关键差异:
#   1. PatchEmbed: Conv2d → Conv1d (1D 时空切片, 不是 2D 图像)
#   2. 位置编码: 1D sin/cos 固定不可学习 (Nx 维序列)
#   3. PDE Embedding: 新增 nn.Embedding(n_pde_types, D) 注入 AdaLN
#                     pde_id=None 时退化为单 PDE 行为 (输出零向量)
#   4. 兼容 EntroDiff 现有签名: model(x_input, sigma, pde_id=None) → (B, out_C, Nx)
#      x_input 是 channel-concat (noisy_u + IC), 与 StandardScore/BVAwareScore 一致
#
# 兼容性铁律 (W5 plan §0.1):
#   - 不使用 in-place 操作 (避免破坏 BVAwareScore 后续 autograd.grad create_graph=True)
#   - sigma 接收 raw σ, 内部做 c_noise = log(σ)/4
#   - 输出 (B, out_channels, Nx), 与现有 sampler/loss 完全兼容
# ============================================================================

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 1. Embedding 模块
# ----------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """
    将连续噪声水平 sigma (raw σ) 映射到 D 维 embedding.

    内部流程:
      sigma → c_noise = log(σ)/4   (EDM 标准编码, 与 StandardScore 一致)
      c_noise → SinusoidalPosEmbed(half_dim) → MLP(D → D)

    设计动机 (W5 plan §2.1):
      - 复用 unet_1d.py SinusoidalPositionEmbeddings 风格
      - log/4 编码使噪声水平的尺度归一化, 避免 σ→0 处梯度爆炸
    """

    def __init__(self, hidden_dim: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim   # 用于 sin/cos 频率分量数 (D 中间)
        # MLP: frequency_dim → hidden_dim → hidden_dim
        # 与官方 DiT 一致 (silu activation, 双层 Linear)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def sigma_embedding(sigma: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        """
        正弦余弦频率 embedding (与 unet_1d SinusoidalPositionEmbeddings 同思路).
        但接收 raw σ, 内部做 log/4 归一.

        Args:
            sigma: (B,) raw 噪声水平 (>0)
            dim:   输出频率维度 (建议偶数)
            max_period: 频率范围

        Returns:
            (B, dim) embedding
        """
        # EDM c_noise: 对数归一, 与 StandardScore.forward 中 c_noise = sigma.log()/4.0 一致
        # 注意 sigma 必须 >0; 训练/推断中 σ 范围一般 [1e-3, 80]
        c_noise = sigma.log() / 4.0  # (B,)

        # 标准 transformer sinusoidal positional encoding 公式
        half = dim // 2
        # 频率: 1 / max_period^(2i/dim), i = 0..half-1
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=sigma.device)  # (half,)
        # broadcast: (B, 1) * (1, half) = (B, half)
        args = c_noise[:, None].float() * freqs[None]
        # 拼接 cos / sin: (B, dim)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # 若 dim 奇数, 末尾 0 填充 (实践中 dim 都设偶数)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (B,) → (B, frequency_dim) → MLP → (B, hidden_dim)
        # 注意: 这里 dim 参数用 self.frequency_dim, 不是 hidden_dim
        # 因为 sinusoidal 提供"原始频率特征", MLP 再学习投影到 hidden_dim
        sig_emb = self.sigma_embedding(sigma, self.frequency_dim)  # (B, freq_dim)
        return self.mlp(sig_emb)  # (B, hidden_dim)


class PDEEmbedder(nn.Module):
    """
    PDE 类型 embedding.

    Args:
        n_pde_types: 支持的 PDE 类型数量 (≥1)
        hidden_dim:  与 TimestepEmbedder 输出对齐的维度 (用于相加注入)

    Forward:
        pde_id: (B,) long 或 None
            - None  → 返回 (B, hidden_dim) 零向量 (单 PDE 退化, 不影响 AdaLN)
            - Tensor→ nn.Embedding lookup → MLP → (B, hidden_dim)

    设计动机 (W5 plan §0.1):
        pde_id=None 时退化为零向量, 保证现有单 PDE 训练脚本切换到 DiT 后无需变动
    """

    def __init__(self, n_pde_types: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_pde_types = n_pde_types
        self.hidden_dim = hidden_dim
        # 即使 n_pde_types=1, embedding 也合法 (只用第 0 行)
        # 单 PDE 训练时 pde_id 总传 0 或 None
        self.embedding = nn.Embedding(n_pde_types, hidden_dim)
        # 投影 MLP 与 TimestepEmbedder 风格对齐 (silu + 双 Linear)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 初始化: embedding 表用 N(0, 0.02) (与官方 DiT class label 一致)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, pde_id: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if pde_id is None:
            # 单 PDE 退化: 返回零向量, 与 AdaLN 加和 = identity 效果
            return torch.zeros(batch_size, self.hidden_dim, device=device)
        # pde_id: (B,) long → (B, hidden_dim)
        emb = self.embedding(pde_id.long())
        return self.mlp(emb)


class PatchEmbed1D(nn.Module):
    """
    1D Patch Embedding (输入信号切块 + 线性投影).

    与官方 DiT (PatchEmbed2D) 的差异:
      Conv2d(C_in, D, kernel=patch, stride=patch)  →  Conv1d 同款 1D 化

    Args:
        in_channels: 输入通道数 (EntroDiff 中通常 2: noisy_u + IC)
        embed_dim:   token 维度 D
        patch_size:  每 patch 包含的网格点数 (W5 plan §2.1: 推荐 4)
        Nx:          输入序列总长 (必须能被 patch_size 整除)

    Forward:
        x: (B, in_channels, Nx) → (B, N_patch, D)  其中 N_patch = Nx // patch_size
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int, Nx: int) -> None:
        super().__init__()
        # 整除约束 (W5 plan §4 风险 2): Nx 必须能被 patch 整除
        assert Nx % patch_size == 0, \
            f"Nx={Nx} 必须被 patch_size={patch_size} 整除 (W5 plan §4 风险点 2)"
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.Nx = Nx
        self.n_patch = Nx // patch_size

        # Conv1d 实现 patch + 投影 (一次性完成, 与官方 DiT 一致)
        # kernel = stride = patch → 不重叠切块
        self.proj = nn.Conv1d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, Nx) → Conv1d → (B, embed_dim, n_patch) → 转置 → (B, n_patch, embed_dim)
        # 转置使其符合 Transformer 标准 (B, L, D) 输入
        x = self.proj(x)              # (B, D, n_patch)
        x = x.transpose(1, 2)         # (B, n_patch, D)
        return x


# ----------------------------------------------------------------------------
# 2. Positional Encoding (1D 固定 sin/cos)
# ----------------------------------------------------------------------------

def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> torch.Tensor:
    """
    1D 正余弦位置编码 (固定不可学习).

    Args:
        embed_dim: token 维度 D
        length:    序列长度 (= n_patch)

    Returns:
        (length, embed_dim) tensor

    与官方 DiT 2D 版本的差异:
        DiT 2D 是 sin/cos(grid_h) 和 sin/cos(grid_w) 各占一半, 然后 concat.
        1D 版本: 直接对单一位置坐标做 sin/cos.
    """
    assert embed_dim % 2 == 0, "embed_dim 必须偶数 (sin/cos 各占一半)"
    # 位置坐标: 0, 1, ..., length-1
    pos = np.arange(length, dtype=np.float64)
    # 频率: 1/10000^(2i/D), i = 0..D/2 - 1
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= (embed_dim / 2.)
    omega = 1. / 10000 ** omega   # (D/2,)
    # outer product: (length, D/2)
    out = np.einsum('m,d->md', pos, omega)
    # 拼接 sin/cos: (length, D)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return torch.from_numpy(emb).float()


# ----------------------------------------------------------------------------
# 3. AdaLN modulation 工具函数
# ----------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    AdaLN 调制: x * (1 + scale) + shift  (官方 DiT 标准)

    Args:
        x:     (B, L, D)
        shift: (B, D)
        scale: (B, D)

    Returns:
        (B, L, D)

    注意: 用 (1 + scale) 而非 scale, 是为了让 scale=0 初始化时 modulate=identity.
    这是 AdaLN-Zero 初始化的关键 (DiT 论文 §3.2).
    """
    # broadcast: (B, L, D) * (B, 1, D) + (B, 1, D)
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ----------------------------------------------------------------------------
# 4. DiT Block (AdaLN-Zero)
# ----------------------------------------------------------------------------

class ManualMultiheadAttention(nn.Module):
    """
    手写 Multi-Head Self-Attention (matmul + softmax, 二阶导全支持).

    设计动机 (W5 plan §0.1 兼容性铁律):
        nn.MultiheadAttention 在 CPU 路径上调用 scaled_dot_product_flash_attention,
        其反向不支持 create_graph=True (CPU)和某些 GPU 版本.
        BVAwareScore 集成时必须能 ∇_x phi_sm 再 ∇_θ loss → 手写最稳.

    序列长度 (Nx//patch=32~64) 极小, 手写性能差距 < 5%.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} 必须被 n_heads={n_heads} 整除"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        # 一次性投影 Q/K/V (官方 DiT/ViT 标准做法)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        # dropout 用 Functional 形式调用, 不存模块状态 (训练/推断分支由 self.training 控制)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        B, L, D = x.shape
        H = self.n_heads
        Hd = self.head_dim

        # qkv: (B, L, 3D) → (B, L, 3, H, Hd) → (3, B, H, L, Hd)
        qkv = self.qkv(x).reshape(B, L, 3, H, Hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # 各 (B, H, L, Hd)

        # 注意力分数: (B, H, L, L)
        # 用 matmul 而非 SDPA → 二阶导全支持
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        if self.dropout_p > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p, training=True)

        # 输出: (B, H, L, Hd) → (B, L, H, Hd) → (B, L, D)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        if self.dropout_p > 0 and self.training:
            out = F.dropout(out, p=self.dropout_p, training=True)
        return out


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with AdaLN-Zero conditioning.

    结构 (官方 DiT §3.2):
        x' = x + gate_msa * MSA(modulate(LN(x), shift_msa, scale_msa))
        x  = x' + gate_mlp * MLP(modulate(LN(x'), shift_mlp, scale_mlp))

    cond → adaLN_modulation (Linear) → 6 chunks: shift/scale/gate × (msa, mlp)

    AdaLN-Zero 初始化:
        adaLN_modulation 最后 Linear 全零 → shift=scale=gate=0 初始
        → 整个 block 初始化为 identity, 训练初期稳定
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        # LayerNorm: 不带 affine 参数 (因 AdaLN 取代 affine)
        # 官方 DiT: nn.LayerNorm(elementwise_affine=False, eps=1e-6)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # 手写 MSA (二阶导支持; W5 plan §0.1 - BVAwareScore 集成硬要求)
        self.attn = ManualMultiheadAttention(dim=dim, n_heads=n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # MLP: dim → mlp_ratio*dim → dim, 带 GELU (官方 DiT)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )
        # AdaLN modulation: cond → 6 个调制参数 (shift/scale/gate × msa/mlp)
        # SiLU + Linear 与官方 DiT 一致
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, L, D)
            cond: (B, D) 条件向量 (time_emb + pde_emb 之和)

        Returns:
            (B, L, D)
        """
        # cond → 6 chunks: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        # 每 chunk shape (B, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(cond).chunk(6, dim=-1)

        # ---- Self-Attention 子层 ----
        # 1) modulate(LN(x)): AdaLN
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        # 2) MSA: 手写 attention 直接返回 (B, L, D)
        attn_out = self.attn(h)
        # 3) gate * MSA + 残差 (避免 in-place +=, W5 plan §0.1)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # ---- MLP 子层 ----
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


# ----------------------------------------------------------------------------
# 5. Final Layer (AdaLN + Linear → 还原 patch)
# ----------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """
    最终输出层: AdaLN modulate → Linear → 还原回 (B, out_C, Nx).

    cond → 2 chunks: shift, scale (无 gate, 因 final 后无残差)
    """

    def __init__(self, dim: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # Linear: (B, L, D) → (B, L, patch_size * out_channels)
        # 然后 reshape 回 (B, out_C, Nx)
        self.linear = nn.Linear(dim, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, L=n_patch, D)
            cond: (B, D)
        Returns:
            (B, L, patch_size * out_channels)  (后续在 DiT1D 主类中 reshape)
        """
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ----------------------------------------------------------------------------
# 6. DiT1D 主类
# ----------------------------------------------------------------------------

class DiT1D(nn.Module):
    """
    1D Diffusion Transformer for EntroDiff (W5).

    论文对应: Docs/path_A_method_skeleton.md §3.5 Foundation Model: DiT Backbone
    工程方案: Docs/black/W5_foundation_engineering_plan.md §2.1

    Args:
        in_channels:   输入通道数 (EntroDiff: 通常 2 = noisy_u + IC)
        out_channels:  输出通道数 (通常 1)
        Nx:            输入序列长度 (必须能被 patch_size 整除)
        dim:           token 维度
        n_layers:      DiTBlock 数量
        n_heads:       MSA head 数
        patch_size:    每 patch 网格点数 (W5 推荐 4)
        n_pde_types:   PDE 类型数 (≥1; 单 PDE 时与 pde_id=None 配合使用)
        dropout:       MSA / MLP dropout

    Forward signature (与 EntroDiff 接口契约一致, W5 plan §0.1):
        forward(x, sigma, pde_id=None) → (B, out_channels, Nx)
        x:      (B, in_channels, Nx)
        sigma:  (B,) raw σ
        pde_id: (B,) long 或 None
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        Nx: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        patch_size: int,
        n_pde_types: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # 架构参数保存 (供测试 / debug 检查)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Nx = Nx
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.patch_size = patch_size
        self.n_pde_types = n_pde_types

        # 1) Patch Embedding (空间 patch 化)
        self.patch_embed = PatchEmbed1D(in_channels, dim, patch_size, Nx)
        self.n_patch = self.patch_embed.n_patch

        # 2) 位置编码 (固定 sin/cos, 注册为 buffer 以便随 .to(device) 迁移)
        pos_embed = get_1d_sincos_pos_embed(dim, self.n_patch)  # (n_patch, dim)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # (1, n_patch, dim)

        # 3) 条件 Embedding: time + pde
        self.t_embedder = TimestepEmbedder(dim)
        self.pde_embedder = PDEEmbedder(n_pde_types, dim)

        # 4) DiT Block 堆叠
        self.blocks = nn.ModuleList([
            DiTBlock(dim=dim, n_heads=n_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(n_layers)
        ])

        # 5) 最终输出层
        self.final_layer = FinalLayer(dim, patch_size, out_channels)

        # 初始化 (AdaLN-Zero, W5 plan §2.1)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """
        AdaLN-Zero 初始化 (DiT 官方做法):
          1. 标准 Xavier 初始化所有 Linear (基础)
          2. PatchEmbed Conv1d 用 Xavier (而非默认 kaiming)
          3. adaLN_modulation 最后 Linear 权重和偏置全置零
             → shift/scale/gate 初始化为 0
             → 所有 DiTBlock 初始化为 identity
             → FinalLayer 初始化为零输出 (训练初期不破坏 EDM precondition)
          4. FinalLayer 的最终 Linear 权重和偏置也置零
             (与官方 DiT 一致, 配合 EDM c_out=σ·σ_d/sqrt(...) 初始 D_x ≈ c_skip*x)
        """

        # 1) 基础初始化: 所有 Linear / Conv1d Xavier
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 2) PatchEmbed Conv1d Xavier (官方 DiT 做法)
        w = self.patch_embed.proj.weight.data
        # flatten 后 xavier_uniform 与 Linear 等价
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        # 3) TimestepEmbedder MLP 第一层用 N(0, 0.02), 与官方 DiT 一致
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 4) AdaLN-Zero: 所有 DiTBlock 的 adaLN_modulation 最后 Linear 全零
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 5) FinalLayer: adaLN_modulation 最后 Linear 全零 + 最终 Linear 也全零
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 (B, n_patch, patch_size * out_C) 还原为 (B, out_C, Nx).

        转换过程:
            (B, L, P*C)  →  reshape (B, L, P, C)  →  permute (B, C, L, P)
                          →  reshape (B, C, L*P) = (B, C, Nx)
        """
        B, L, _ = x.shape
        P = self.patch_size
        C = self.out_channels
        # reshape: (B, L, P*C) → (B, L, P, C)
        x = x.reshape(B, L, P, C)
        # permute: (B, C, L, P)
        x = x.permute(0, 3, 1, 2).contiguous()
        # reshape: (B, C, L*P=Nx)
        x = x.reshape(B, C, L * P)
        return x

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        pde_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:      (B, in_channels, Nx)  channel-concat 输入 [noisy_u | IC | ...]
            sigma:  (B,) raw 噪声水平 (>0)
            pde_id: (B,) long 或 None (None → 单 PDE 退化, 不影响 AdaLN)

        Returns:
            (B, out_channels, Nx)
        """
        B = x.shape[0]
        device = x.device

        # 1) Patch + 位置编码
        # (B, in_C, Nx) → (B, n_patch, dim)
        h = self.patch_embed(x)
        # 加 1D sin/cos 位置编码 (broadcast: (1, n_patch, dim) → (B, n_patch, dim))
        # 用 + 而非 += (避免 in-place, W5 plan §0.1)
        h = h + self.pos_embed

        # 2) 条件 embedding: time + pde
        t_emb = self.t_embedder(sigma)                          # (B, dim)
        p_emb = self.pde_embedder(pde_id, B, device)            # (B, dim) (None → 0)
        cond = t_emb + p_emb                                     # (B, dim)

        # 3) 堆叠 DiTBlock
        for block in self.blocks:
            h = block(h, cond)                                  # (B, n_patch, dim)

        # 4) 最终输出层
        h = self.final_layer(h, cond)                           # (B, n_patch, P*out_C)

        # 5) Unpatch 还原回空间维度
        out = self.unpatchify(h)                                # (B, out_C, Nx)
        return out
