# ============================================================================
# DiT-1D Backbone 单元测试 (W5-A)
#
# 工程方案: Docs/black/W5_foundation_engineering_plan.md §2.1 单元测试要求
#
# 4 个核心测试:
#   1. test_shape:        前向输出 shape 正确 (with pde_id)
#   2. test_pde_id_none:  pde_id=None 单 PDE 退化必须正常
#   3. test_grad_flow:    backward 后所有参数梯度非 None
#   4. test_create_graph: 二阶梯度可计算 (BVAwareScore 集成的硬要求)
#
# 跑测试:
#   python -m pytest PROJECT/black/tests/test_dit_1d.py -v
# ============================================================================

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# 让 tests 目录可以 import src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.dit_1d import (  # noqa: E402
    DiT1D,
    PatchEmbed1D,
    TimestepEmbedder,
    PDEEmbedder,
    DiTBlock,
    FinalLayer,
    get_1d_sincos_pos_embed,
    modulate,
)


# ----------------------------------------------------------------------------
# 配置: tiny 规模, 适合 CPU 跑测试
# ----------------------------------------------------------------------------

DIM = 64        # token 维度 (测试用小, 真实跑 256+)
N_LAYERS = 2    # block 数 (测试 2 即可)
N_HEADS = 4     # MSA head 数
PATCH = 4       # W5 推荐 patch=4
NX = 128        # 必须能被 PATCH 整除
B = 8           # batch size
IN_C = 2        # noisy_u + IC


# ----------------------------------------------------------------------------
# 必需测试 1-4
# ----------------------------------------------------------------------------

def test_shape() -> None:
    """前向 shape: (B, in_C, Nx) → (B, out_C, Nx), 带 pde_id."""
    model = DiT1D(
        in_channels=IN_C, out_channels=1, Nx=NX,
        dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
        patch_size=PATCH, n_pde_types=2, dropout=0.0,
    )
    x = torch.randn(B, IN_C, NX)
    sigma = torch.rand(B) * 10.0 + 1e-2  # σ ∈ [0.01, 10.01]
    pde_id = torch.randint(0, 2, (B,))
    y = model(x, sigma, pde_id=pde_id)
    assert y.shape == (B, 1, NX), f"输出 shape 错误: {y.shape}, 期望 ({B}, 1, {NX})"


def test_pde_id_none() -> None:
    """pde_id=None 时单 PDE 退化必须正常前向 (W5 plan §0.1)."""
    # n_pde_types=1 + pde_id=None: 现有单 PDE 训练脚本切换 DiT 后无需变动
    model = DiT1D(
        in_channels=IN_C, out_channels=1, Nx=NX,
        dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
        patch_size=PATCH, n_pde_types=1, dropout=0.0,
    )
    x = torch.randn(B, IN_C, NX)
    sigma = torch.rand(B) + 1e-2
    y = model(x, sigma, pde_id=None)
    assert y.shape == (B, 1, NX)

    # 同样验证 n_pde_types=2 + pde_id=None 也合法 (虽然不推荐使用)
    model2 = DiT1D(
        in_channels=IN_C, out_channels=1, Nx=NX,
        dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
        patch_size=PATCH, n_pde_types=2, dropout=0.0,
    )
    y2 = model2(x, sigma, pde_id=None)
    assert y2.shape == (B, 1, NX)


def test_grad_flow() -> None:
    """backward 后所有 trainable 参数 grad 非 None."""
    model = DiT1D(
        in_channels=IN_C, out_channels=1, Nx=NX,
        dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
        patch_size=PATCH, n_pde_types=2, dropout=0.0,
    )
    x = torch.randn(B, IN_C, NX)
    sigma = torch.rand(B) + 1e-2
    pde_id = torch.randint(0, 2, (B,))
    y = model(x, sigma, pde_id=pde_id)
    loss = y.pow(2).mean()
    loss.backward()

    # 注意: AdaLN-Zero 初始化使 final_layer.linear 输出全零, 且 adaLN_modulation 最后层全零
    # 这意味着 loss 第一次 backward 时, 部分参数的 grad 可能为 0 (但仍非 None)
    # 关键检查: trainable 参数都收到了 grad (非 None)
    no_grad_params = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            no_grad_params.append(name)
    assert len(no_grad_params) == 0, \
        f"以下 trainable 参数没有梯度: {no_grad_params}"


def test_create_graph() -> None:
    """
    二阶梯度可计算: autograd.grad(y, x, create_graph=True) 不报错.
    这是 BVAwareScore 集成的硬要求 (它会对 phi_sm 输出再求 ∇_x).
    """
    model = DiT1D(
        in_channels=IN_C, out_channels=1, Nx=NX,
        dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
        patch_size=PATCH, n_pde_types=2, dropout=0.0,
    )
    x = torch.randn(B, IN_C, NX, requires_grad=True)
    sigma = torch.rand(B) + 1e-2
    pde_id = torch.randint(0, 2, (B,))
    y = model(x, sigma, pde_id=pde_id)
    # 一阶梯度 ∇_x y, 保留计算图 (二阶可用)
    grad_x = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    assert grad_x.shape == x.shape, f"grad shape 错误: {grad_x.shape}"
    # 二阶梯度: ∇_θ ‖∇_x y‖² (loss 对模型参数求导)
    second_loss = grad_x.pow(2).mean()
    second_loss.backward()
    # 验证至少 patch_embed 参数收到了二阶梯度
    assert model.patch_embed.proj.weight.grad is not None


# ----------------------------------------------------------------------------
# 辅助测试: 子模块单独验证
# ----------------------------------------------------------------------------

def test_patch_embed_assert() -> None:
    """Nx 不能整除 patch_size 时必须 assert 失败."""
    with pytest.raises(AssertionError, match="必须被 patch_size"):
        PatchEmbed1D(in_channels=2, embed_dim=64, patch_size=5, Nx=128)


def test_pos_embed_shape() -> None:
    """1D sincos 位置编码 shape 正确."""
    pe = get_1d_sincos_pos_embed(embed_dim=64, length=32)
    assert pe.shape == (32, 64)


def test_modulate_identity_when_zero() -> None:
    """shift=0, scale=0 时 modulate 应为 identity (AdaLN-Zero 关键性质)."""
    x = torch.randn(2, 10, 64)
    shift = torch.zeros(2, 64)
    scale = torch.zeros(2, 64)
    y = modulate(x, shift, scale)
    assert torch.allclose(y, x), "modulate(x, 0, 0) 应等于 x (AdaLN-Zero identity)"


def test_pde_embedder_none_returns_zeros() -> None:
    """pde_id=None 时 PDEEmbedder 必须返回零向量."""
    emb = PDEEmbedder(n_pde_types=3, hidden_dim=64)
    out = emb(pde_id=None, batch_size=4, device=torch.device("cpu"))
    assert out.shape == (4, 64)
    assert torch.allclose(out, torch.zeros(4, 64))


def test_param_count_reference() -> None:
    """
    打印 dim=256, n_layers=6 配置下的参数量, 用于 W5-A 报告.
    (无 assert, 仅 print; pytest -s 时可见)
    """
    model = DiT1D(
        in_channels=2, out_channels=1, Nx=128,
        dim=256, n_layers=6, n_heads=4,
        patch_size=4, n_pde_types=2, dropout=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[DiT1D dim=256 n_layers=6] 参数量: {n_params:,} (~{n_params/1e6:.2f}M)")


def test_smaller_dim_works() -> None:
    """
    Sanity: tiny 配置 (dim=128, n_layers=4) 在 RTX 4060 (8GB) 上应该轻松跑.
    本地 CPU 也能正常前向 + backward.
    """
    model = DiT1D(
        in_channels=2, out_channels=1, Nx=128,
        dim=128, n_layers=4, n_heads=4,
        patch_size=4, n_pde_types=2, dropout=0.0,
    )
    x = torch.randn(4, 2, 128)
    sigma = torch.rand(4) + 1e-2
    pde_id = torch.tensor([0, 1, 0, 1])
    y = model(x, sigma, pde_id=pde_id)
    assert y.shape == (4, 1, 128)
    y.sum().backward()  # 不爆


if __name__ == "__main__":
    # 直接 python 运行也可
    pytest.main([__file__, "-v", "-s"])
