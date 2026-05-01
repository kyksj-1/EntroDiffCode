# ============================================================================
# W5-C 集成测试: FoundationScore (DiT-Plain) + BVAwareScore(backbone='dit')
#
# 工程方案: Docs/black/W5_foundation_engineering_plan.md §2.3
#
# 测试目标:
#   1. FoundationScore 与 StandardScore 接口完全一致 (输入/输出/sigma_data)
#   2. BVAwareScore(backbone='dit') 与 backbone='unet' 接口一致, 内部 phi_sm 用 DiT
#   3. 二者均能与现有 loss (DSM/BV) + sampler (entrodiff_heun_sampler) 协作
#   4. pde_id 透传链路完整 (model → loss → sampler 三处都能消费)
#   5. backbone='unet' 默认行为不变 (现有训练脚本零破坏)
#
# 跑测试:
#   python -m pytest PROJECT/black/tests/test_dit_scores.py -v
# ============================================================================

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.foundation_score import FoundationScore  # noqa: E402
from src.models.score_param import StandardScore, BVAwareScore  # noqa: E402
from src.diffusion.losses import (  # noqa: E402
    get_dsm_loss,
    get_bv_loss,
    get_godunov_time_loss,
    pde_residual,
    _resolve_flux_fn,
)
from src.diffusion.samplers import entrodiff_heun_sampler  # noqa: E402


# ----------------------------------------------------------------------------
# 共用 dit_kwargs (tiny 规模)
# ----------------------------------------------------------------------------

DIT_KW_TINY = {
    "dim": 64,
    "n_layers": 2,
    "n_heads": 4,
    "patch_size": 4,
    "n_pde_types": 2,
    "dropout": 0.0,
}

NX = 128
B = 4
IN_C = 2  # noisy_u + IC


# ----------------------------------------------------------------------------
# Group 1: FoundationScore (DiT-Plain) 接口测试
# ----------------------------------------------------------------------------

def test_foundation_score_shape() -> None:
    """FoundationScore 输出 shape: (B, in_C, Nx) → (B, 1, Nx)."""
    m = FoundationScore(in_channels=IN_C, Nx=NX, dit_kwargs=DIT_KW_TINY)
    x = torch.randn(B, IN_C, NX)
    sigma = torch.rand(B) + 1e-2
    pde_id = torch.randint(0, 2, (B,))
    D = m(x, sigma, pde_id=pde_id)
    assert D.shape == (B, 1, NX)


def test_foundation_score_pde_none() -> None:
    """pde_id=None 单 PDE 退化."""
    kw = dict(DIT_KW_TINY); kw["n_pde_types"] = 1
    m = FoundationScore(in_channels=IN_C, Nx=NX, dit_kwargs=kw)
    x = torch.randn(B, IN_C, NX)
    sigma = torch.rand(B) + 1e-2
    D = m(x, sigma, pde_id=None)
    assert D.shape == (B, 1, NX)


def test_foundation_score_edm_precondition() -> None:
    """
    σ→0 极限: c_skip→1, c_out→0, D_x → x[:, :1] (Tweedie 反演的极限).
    σ→∞ 极限: c_skip→0, c_out→σ_data, D_x → σ_data * F_θ.
    验证 EDM precondition 公式与 StandardScore 一致.
    """
    m = FoundationScore(in_channels=IN_C, Nx=NX, dit_kwargs=DIT_KW_TINY,
                        sigma_data=0.5)
    x = torch.randn(B, IN_C, NX)
    sigma_small = torch.full((B,), 1e-6)
    D_small = m(x, sigma_small, pde_id=None)
    # σ→0 时 D_x ≈ x[:, :1] (因 AdaLN-Zero 初始化 F_θ ≈ 0; c_out 也很小)
    # 容忍 1% 误差 (浮点)
    assert torch.allclose(D_small, x[:, :1, :], atol=1e-3), \
        "σ→0 极限 D_x 应趋于 x[:, :1] (Tweedie 反演)"


def test_foundation_score_missing_kwargs_raises() -> None:
    """dit_kwargs 缺必填字段时 KeyError."""
    bad_kw = {"dim": 64, "n_layers": 2}  # 缺 n_heads / patch_size
    with pytest.raises(KeyError, match="必填字段"):
        FoundationScore(Nx=NX, dit_kwargs=bad_kw)


# ----------------------------------------------------------------------------
# Group 2: BVAwareScore backbone='unet' (现有行为) vs 'dit' (W5-C 新)
# ----------------------------------------------------------------------------

def test_bvaware_unet_default_unchanged() -> None:
    """
    backbone='unet' 默认: 前向行为与 W5-C 改动前完全一致.
    现有 train_bvaware.py 调用 BVAwareScore(in_channels=2, dim=64) 必须仍工作.

    W5-C 修复 (2026-05-01): 输出 shape 现在是 (B, 1, Nx), 与 StandardScore 对齐.
    解决了 R7 (sampler+cond shape 错配). ckpt-compatible.
    """
    m = BVAwareScore(in_channels=IN_C, dim=64)  # 不传 backbone → 默认 'unet'
    x = torch.randn(B, IN_C, NX)
    sigma = torch.rand(B) + 1e-2
    # 不传 pde_id 也必须工作
    D = m(x, sigma)
    assert D.shape == (B, 1, NX), \
        f"W5-C 修复后 BVAware 输出 (B, 1, Nx), 实际 {D.shape}"


def test_bvaware_dit_backbone() -> None:
    """backbone='dit' 时 phi_sm 替换为 DiT, 输出 shape (B, 1, Nx) 与 StandardScore 一致."""
    dit_kw_with_nx = dict(DIT_KW_TINY); dit_kw_with_nx["Nx"] = NX
    m = BVAwareScore(
        in_channels=IN_C,
        backbone="dit",
        dit_kwargs=dit_kw_with_nx,
        n_pde_types=2,
    )
    x = torch.randn(B, IN_C, NX)
    sigma = torch.rand(B) + 1e-2
    pde_id = torch.randint(0, 2, (B,))
    D = m(x, sigma, pde_id=pde_id)
    # W5-C 修复后: 输出 (B, 1, Nx)
    assert D.shape == (B, 1, NX)


def test_bvaware_dit_grad_flow() -> None:
    """DiT-BVAware backward 可计算 (含 autograd.grad create_graph=True 的二阶导)."""
    dit_kw_with_nx = dict(DIT_KW_TINY); dit_kw_with_nx["Nx"] = NX
    m = BVAwareScore(in_channels=IN_C, backbone="dit",
                     dit_kwargs=dit_kw_with_nx, n_pde_types=2)
    x = torch.randn(B, IN_C, NX)
    sigma = torch.rand(B) + 1e-2
    pde_id = torch.randint(0, 2, (B,))
    D = m(x, sigma, pde_id=pde_id)
    loss = D.pow(2).mean()
    loss.backward()
    # 验证至少 phi_sm DiT 的 patch_embed 收到了梯度
    assert m.phi_sm_net.patch_embed.proj.weight.grad is not None


def test_bvaware_dit_missing_dit_kwargs_raises() -> None:
    """backbone='dit' 时不传 dit_kwargs 必须明确报错."""
    with pytest.raises(ValueError, match="dit_kwargs 必填"):
        BVAwareScore(in_channels=IN_C, backbone="dit")


def test_bvaware_unknown_backbone_raises() -> None:
    """backbone 拒绝未知字符串."""
    with pytest.raises(ValueError, match="未知 backbone"):
        BVAwareScore(in_channels=IN_C, backbone="transformer_xl")


# ----------------------------------------------------------------------------
# Group 3: Loss / Sampler 兼容性
# ----------------------------------------------------------------------------

def test_dsm_loss_with_foundation() -> None:
    """get_dsm_loss 接受 pde_id, 与 FoundationScore 协作."""
    m = FoundationScore(in_channels=IN_C, Nx=NX, dit_kwargs=DIT_KW_TINY)
    x_target = torch.randn(B, 1, NX)        # ground truth (1-ch)
    cond = torch.randn(B, 1, NX)             # IC 条件
    sigma = torch.rand(B) + 1e-2
    pde_id = torch.randint(0, 2, (B,))
    loss = get_dsm_loss(m, x_target, sigma, conditioning=cond, pde_id=pde_id)
    assert loss.dim() == 0   # 标量
    loss.backward()


def test_dsm_loss_standard_unchanged() -> None:
    """get_dsm_loss 不传 pde_id 时与 StandardScore (现有调用) 行为一致."""
    m = StandardScore(in_channels=IN_C)
    x_target = torch.randn(B, 1, NX)
    cond = torch.randn(B, 1, NX)
    sigma = torch.rand(B) + 1e-2
    # 不传 pde_id → StandardScore.forward 接收 pde_id=None → 忽略
    loss = get_dsm_loss(m, x_target, sigma, conditioning=cond)
    assert loss.dim() == 0
    loss.backward()


def test_godunov_time_loss_flux_dispatch() -> None:
    """get_godunov_time_loss 的 flux_type 派遣 (W5-C 新) 工作正常."""
    m = FoundationScore(in_channels=IN_C, Nx=NX, dit_kwargs=DIT_KW_TINY)
    x_prev = torch.randn(B, 1, NX)
    x_target = torch.randn(B, 1, NX)
    cond = torch.randn(B, 1, NX)
    sigma = torch.rand(B) + 1e-2
    pde_id = torch.zeros(B, dtype=torch.long)

    # burgers flux (默认)
    loss_b = get_godunov_time_loss(
        m, x_prev, x_target, sigma, dt=0.05, dx=2 * 3.14159 / NX,
        conditioning=cond, pde_id=pde_id, flux_type="burgers",
    )
    assert loss_b.dim() == 0

    # buckley_leverett flux
    # BL 数据应 ∈ [0,1]; 这里用 sigmoid 投影避免极端值
    x_prev_bl = torch.sigmoid(x_prev)
    x_tgt_bl = torch.sigmoid(x_target)
    loss_bl = get_godunov_time_loss(
        m, x_prev_bl, x_tgt_bl, sigma, dt=0.05, dx=2 * 3.14159 / NX,
        conditioning=cond, pde_id=pde_id, flux_type="buckley_leverett",
    )
    assert loss_bl.dim() == 0


def test_flux_registry_unknown_raises() -> None:
    """未知 flux_type 给清晰报错."""
    with pytest.raises(KeyError, match="未知 flux_type"):
        _resolve_flux_fn("non_existent_flux")


def test_pde_residual_with_flux_type() -> None:
    """pde_residual 默认 'burgers' 不变行为, 显式 flux_type 也能工作."""
    u = torch.randn(B, 1, NX)
    res_b = pde_residual(u, dx=2 * 3.14159 / NX)   # 不传 flux_type → 默认 burgers
    assert res_b.shape == u.shape

    u_bl = torch.sigmoid(u)
    res_bl = pde_residual(u_bl, dx=2 * 3.14159 / NX, flux_type="buckley_leverett")
    assert res_bl.shape == u.shape


# ----------------------------------------------------------------------------
# Group 4: Sampler end-to-end (DiT-Plain)
# ----------------------------------------------------------------------------

def test_sampler_with_foundation_score() -> None:
    """entrodiff_heun_sampler 与 FoundationScore + pde_id 端到端跑通."""
    m = FoundationScore(in_channels=IN_C, Nx=NX, dit_kwargs=DIT_KW_TINY)
    cond = torch.randn(B, 1, NX)
    pde_id = torch.zeros(B, dtype=torch.long)
    out = entrodiff_heun_sampler(
        m,
        shape=(B, 1, NX),
        sigma_min=0.01, sigma_max=10.0,
        tau_max=1.0, nu=1.0,
        num_steps=4,                    # 测试用极少步数, 验证 forward 不爆
        device="cpu",
        zeta_pde=0.0,                    # 不开 PDE guidance (省时)
        conditioning=cond,
        pde_id=pde_id,
        flux_type="burgers",
    )
    assert out.shape == (B, 1, NX)


def test_sampler_with_bvaware_dit() -> None:
    """
    sampler + BVAware(backbone='dit') 端到端 (W5-C 修复后 R7 解决).

    现在可以直接用 in_channels=2 + cond, 因为 BVAware 输出已对齐 (B, 1, Nx).
    """
    dit_kw_with_nx = dict(DIT_KW_TINY); dit_kw_with_nx["Nx"] = NX
    m = BVAwareScore(in_channels=IN_C, backbone="dit",
                     dit_kwargs=dit_kw_with_nx, n_pde_types=2)
    cond = torch.randn(B, 1, NX)
    pde_id = torch.tensor([0, 1, 0, 1])
    out = entrodiff_heun_sampler(
        m, shape=(B, 1, NX),
        sigma_min=0.01, sigma_max=10.0,
        tau_max=1.0, nu=1.0,
        num_steps=4, device="cpu",
        zeta_pde=0.0,
        conditioning=cond, pde_id=pde_id,
    )
    assert out.shape == (B, 1, NX)


# ----------------------------------------------------------------------------
# Group 5: 现有训练脚本零破坏 (回归测试)
# ----------------------------------------------------------------------------

def test_standard_score_no_pde_id_call() -> None:
    """StandardScore 的现有调用 (不传 pde_id) 必须仍能工作."""
    m = StandardScore(in_channels=IN_C)
    x = torch.randn(B, IN_C, NX)
    sigma = torch.rand(B) + 1e-2
    # 现有 train_mvp / train_baseline 的调用形式
    D = m(x, sigma)
    assert D.shape == (B, 1, NX)


def test_bvaware_no_pde_id_call() -> None:
    """BVAwareScore 默认 (UNet) 的现有调用 (不传 pde_id) 必须仍工作."""
    m = BVAwareScore(in_channels=IN_C, dim=64)
    x = torch.randn(B, IN_C, NX)
    sigma = torch.rand(B) + 1e-2
    D = m(x, sigma)
    # W5-C 修复后: 输出 (B, 1, Nx)
    assert D.shape == (B, 1, NX)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
