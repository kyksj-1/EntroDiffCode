# ============================================================================
# E3 Euler 训练管线测试 (W5-SA2)
# ============================================================================

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.euler_dataset import EulerDataset
from src.models.score_param import BVAwareScore


# ----------------------------------------------------------------------------
# Mock data fixture
# ----------------------------------------------------------------------------

@pytest.fixture
def mock_euler_npy(tmp_path: Path) -> Path:
    """生成 mock Euler 数据 (N=20, N_time=11, 3 components, Nx=128)."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(20, 11, 3, 128)).astype(np.float32)
    fp = tmp_path / "euler_mock.npy"
    np.save(fp, data)
    return fp


# ----------------------------------------------------------------------------
# EulerDataset 测试
# ----------------------------------------------------------------------------

def test_euler_dataset_loading(mock_euler_npy: Path) -> None:
    """加载 + train split 大小."""
    ds = EulerDataset(mock_euler_npy, mode="train", conditioning_type="ic")
    # 80% of 20 = 16
    assert len(ds) == 16


def test_euler_dataset_getitem_shape(mock_euler_npy: Path) -> None:
    """__getitem__ 返回 (N_time, 3, Nx) trajectory."""
    ds = EulerDataset(mock_euler_npy, mode="train")
    item = ds[0]
    assert item.shape == (11, 3, 128)
    assert item.dtype == torch.float32


def test_euler_dataset_get_conditioning(mock_euler_npy: Path) -> None:
    """get_conditioning 返回 (B, 3, Nx) 首帧."""
    ds = EulerDataset(mock_euler_npy, mode="train", conditioning_type="ic")
    # 模拟一个 batch (B=4)
    batch = torch.stack([ds[i] for i in range(4)])
    assert batch.shape == (4, 11, 3, 128)
    cond = ds.get_conditioning(batch)
    assert cond is not None
    assert cond.shape == (4, 3, 128)


def test_euler_dataset_conditioning_none(mock_euler_npy: Path) -> None:
    """conditioning='none' 返回 None."""
    ds = EulerDataset(mock_euler_npy, mode="train", conditioning_type="none")
    batch = torch.stack([ds[i] for i in range(2)])
    assert ds.get_conditioning(batch) is None


def test_euler_dataset_split_consistency(mock_euler_npy: Path) -> None:
    """train/val/test 切分大小."""
    train = EulerDataset(mock_euler_npy, mode="train")
    val = EulerDataset(mock_euler_npy, mode="val")
    test = EulerDataset(mock_euler_npy, mode="test")
    assert len(train) == 16
    assert len(val) == 2
    assert len(test) == 2


def test_euler_dataset_wrong_shape_raises(tmp_path: Path) -> None:
    """3D 数据 (Burgers shape) 应报错."""
    bad_data = np.random.randn(10, 11, 128).astype(np.float32)
    fp = tmp_path / "bad.npy"
    np.save(fp, bad_data)
    with pytest.raises(ValueError, match="应为 4D"):
        EulerDataset(fp, mode="train")


# ----------------------------------------------------------------------------
# BVAwareScore 系统级测试 (out_channels=3)
# ----------------------------------------------------------------------------

def test_bvaware_euler_shape() -> None:
    """BVAwareScore(out_channels=3) 输出 (B, 3, Nx)."""
    m = BVAwareScore(in_channels=6, dim=32, out_channels=3)
    x = torch.randn(4, 6, 128)
    sigma = torch.rand(4) + 1e-2
    D = m(x, sigma)
    assert D.shape == (4, 3, 128), f"Euler BVAware 输出 {D.shape}, 期望 (4, 3, 128)"


def test_bvaware_euler_grad_flow() -> None:
    """系统级 BVAware backward 不爆."""
    m = BVAwareScore(in_channels=6, dim=32, out_channels=3)
    x = torch.randn(2, 6, 128)
    sigma = torch.rand(2) + 1e-2
    D = m(x, sigma)
    D.pow(2).mean().backward()
    # 至少 phi_sm 收到梯度
    p = next(iter(m.phi_sm_net.parameters()))
    assert p.grad is not None


def test_bvaware_scalar_default_unchanged() -> None:
    """默认 out_channels=1 行为不变 (单 PDE Burgers 兼容)."""
    m = BVAwareScore(in_channels=2, dim=32)   # out_channels 默认 1
    x = torch.randn(4, 2, 128)
    sigma = torch.rand(4) + 1e-2
    D = m(x, sigma)
    assert D.shape == (4, 1, 128), \
        f"默认 out_channels=1 输出应为 (B, 1, Nx), 实际 {D.shape}"


def test_bvaware_euler_create_graph() -> None:
    """系统级 BVAware 二阶导支持 (训练时 loss 对 D_x 再求导)."""
    m = BVAwareScore(in_channels=6, dim=32, out_channels=3)
    x = torch.randn(2, 6, 128, requires_grad=True)
    sigma = torch.rand(2) + 1e-2
    D = m(x, sigma)
    grad_x = torch.autograd.grad(D.sum(), x, create_graph=True)[0]
    assert grad_x.shape == x.shape
    grad_x.pow(2).mean().backward()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
