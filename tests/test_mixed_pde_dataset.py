# ============================================================================
# MixedPDEDataset 单元测试 (W5-B)
#
# 工程方案: Docs/black/W5_foundation_engineering_plan.md §2.2
#
# 5 个核心测试 (用 mock npy 数据, 不依赖真实数据集):
#   1. test_two_pdes_loading
#   2. test_pde_id_consistent
#   3. test_collate_fn
#   4. test_split_consistency
#   5. test_extensibility (3 PDE 配置也能跑)
#
# 跑测试:
#   python -m pytest PROJECT/black/tests/test_mixed_pde_dataset.py -v
# ============================================================================

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.mixed_pde_dataset import MixedPDEDataset  # noqa: E402


# ----------------------------------------------------------------------------
# 共用 fixture: 创建 mock npy 数据 + 返回临时目录
# ----------------------------------------------------------------------------

@pytest.fixture
def mock_data_dir(tmp_path: Path) -> Path:
    """
    生成 3 个 PDE 的 mock 数据 (用 randn), 写入临时目录.

    数据形状 (与真实 generate_data.py 对齐):
        burgers:           (N=100, N_time=11, Nx=128)
        buckley_leverett:  (N=80,  N_time=11, Nx=128)
        euler_sod:         (N=60,  N_time=11, Nx=128)
    """
    rng = np.random.default_rng(seed=42)
    np.save(tmp_path / "burgers_mock.npy", rng.normal(size=(100, 11, 128)).astype(np.float32))
    np.save(tmp_path / "bl_mock.npy", rng.normal(size=(80, 11, 128)).astype(np.float32))
    np.save(tmp_path / "euler_mock.npy", rng.normal(size=(60, 11, 128)).astype(np.float32))
    return tmp_path


def _two_pdes_cfg() -> list[dict]:
    """2 个 PDE 的标准配置 (用于多数测试)."""
    return [
        {"name": "burgers", "data_file": "burgers_mock.npy",
         "flux_type": "burgers", "weight": 1.0},
        {"name": "buckley_leverett", "data_file": "bl_mock.npy",
         "flux_type": "buckley_leverett", "weight": 1.0},
    ]


# ----------------------------------------------------------------------------
# 5 个必需测试
# ----------------------------------------------------------------------------

def test_two_pdes_loading(mock_data_dir: Path) -> None:
    """
    2 个 PDE 配置, mode='train' → __len__ 等于 train splits 之和.

    burgers: 100 → train 80
    bl:      80  → train 64
    总计: 144
    """
    ds = MixedPDEDataset(
        pdes_config=_two_pdes_cfg(),
        data_dir=mock_data_dir,
        mode="train",
        conditioning_type="ic",
        mix_strategy="uniform",
    )
    assert len(ds) == 80 + 64, f"__len__={len(ds)}, 期望 144"
    # PDE 数与 num_pde_types 接口一致
    assert ds.num_pde_types == 2
    assert ds.pde_names == ["burgers", "buckley_leverett"]
    # flux 派遣接口
    assert ds.get_flux_type(0) == "burgers"
    assert ds.get_flux_type(1) == "buckley_leverett"


def test_pde_id_consistent(mock_data_dir: Path) -> None:
    """同 idx 多次访问, pde_id 必须保持一致 (无随机性)."""
    ds = MixedPDEDataset(
        pdes_config=_two_pdes_cfg(),
        data_dir=mock_data_dir,
        mode="train",
    )
    # 检查多个 idx 各访问 3 次, pde_id 都一致
    for idx in [0, 50, 100, 130]:
        ids = [ds[idx]["pde_id"] for _ in range(3)]
        assert len(set(ids)) == 1, f"idx={idx}: pde_id 不一致 {ids}"

    # 跨 PDE 边界: idx<80 必为 burgers (id=0), idx>=80 必为 bl (id=1)
    assert ds[0]["pde_id"] == 0 and ds[0]["pde_name"] == "burgers"
    assert ds[79]["pde_id"] == 0
    assert ds[80]["pde_id"] == 1 and ds[80]["pde_name"] == "buckley_leverett"
    assert ds[143]["pde_id"] == 1


def test_collate_fn(mock_data_dir: Path) -> None:
    """collate 后 shape 正确, pde_id 是 long Tensor, pde_name 是 list[str]."""
    ds = MixedPDEDataset(
        pdes_config=_two_pdes_cfg(),
        data_dir=mock_data_dir,
        mode="train",
    )
    loader = DataLoader(
        ds, batch_size=8, shuffle=False, num_workers=0,
        collate_fn=MixedPDEDataset.collate_fn,
    )
    batch = next(iter(loader))
    # shapes
    assert batch["trajectory"].shape == (8, 11, 128)
    assert batch["ic"].shape == (8, 1, 128)
    assert batch["x_target"].shape == (8, 1, 128)
    # pde_id: (B,) long
    assert batch["pde_id"].shape == (8,)
    assert batch["pde_id"].dtype == torch.long
    # pde_name: list[str], 长度 B
    assert isinstance(batch["pde_name"], list)
    assert len(batch["pde_name"]) == 8
    assert all(isinstance(n, str) for n in batch["pde_name"])

    # get_conditioning 通过 dataset 接口提取 (与 BurgersDataset 风格一致)
    cond = ds.get_conditioning(batch)
    assert cond is not None
    assert cond.shape == (8, 1, 128)


def test_split_consistency(mock_data_dir: Path) -> None:
    """train/val/test split 索引互不重叠 (在每个 PDE 内部)."""
    cfg = _two_pdes_cfg()
    train_ds = MixedPDEDataset(cfg, mock_data_dir, mode="train")
    val_ds = MixedPDEDataset(cfg, mock_data_dir, mode="val")
    test_ds = MixedPDEDataset(cfg, mock_data_dir, mode="test")

    # 每个 PDE 内: 80% / 10% / 10%
    # burgers: 80 / 10 / 10
    # bl:      64 / 8  / 8
    assert len(train_ds) == 80 + 64
    assert len(val_ds) == 10 + 8
    assert len(test_ds) == 10 + 8

    # 验证 train/val/test 在数据值上不重叠 (用每个 sample 的 hash 判断)
    # 取出每个 split 的 burgers 样本 (pde_id=0) 的首格值, 三个 split 应当无交集
    train_vals = {round(float(train_ds[i]["trajectory"][0, 0]), 6)
                  for i in range(len(train_ds)) if train_ds[i]["pde_id"] == 0}
    val_vals = {round(float(val_ds[i]["trajectory"][0, 0]), 6)
                for i in range(len(val_ds)) if val_ds[i]["pde_id"] == 0}
    test_vals = {round(float(test_ds[i]["trajectory"][0, 0]), 6)
                 for i in range(len(test_ds)) if test_ds[i]["pde_id"] == 0}
    assert train_vals.isdisjoint(val_vals)
    assert train_vals.isdisjoint(test_vals)
    assert val_vals.isdisjoint(test_vals)


def test_extensibility(mock_data_dir: Path) -> None:
    """3 PDE 配置: 验证不硬编码, 加 PDE 仅需配置."""
    cfg_3pde = [
        {"name": "burgers", "data_file": "burgers_mock.npy",
         "flux_type": "burgers", "weight": 1.0},
        {"name": "buckley_leverett", "data_file": "bl_mock.npy",
         "flux_type": "buckley_leverett", "weight": 1.0},
        {"name": "euler_sod", "data_file": "euler_mock.npy",
         "flux_type": "euler", "weight": 0.5},
    ]
    ds = MixedPDEDataset(
        pdes_config=cfg_3pde,
        data_dir=mock_data_dir,
        mode="train",
    )
    # 3 PDE: train splits = 80 + 64 + 48 = 192
    assert len(ds) == 80 + 64 + 48
    assert ds.num_pde_types == 3
    assert ds.pde_names == ["burgers", "buckley_leverett", "euler_sod"]
    # flux 派遣
    assert ds.get_flux_type(2) == "euler"

    # weighted_by_size 也能跑 (即使 3 PDE 大小不同)
    ds_w = MixedPDEDataset(
        pdes_config=cfg_3pde,
        data_dir=mock_data_dir,
        mode="train",
        mix_strategy="weighted_by_size",
    )
    # max n_split = 80 (burgers); weight 分别 1.0/1.0/0.5
    # → 期望: 80*1 + 80*1 + 80*0.5 = 200
    assert len(ds_w) == 80 + 80 + 40


# ----------------------------------------------------------------------------
# 辅助测试
# ----------------------------------------------------------------------------

def test_missing_file_raises(mock_data_dir: Path) -> None:
    """数据文件不存在时给清晰报错."""
    bad_cfg = [{"name": "x", "data_file": "nonexistent.npy", "flux_type": "burgers"}]
    with pytest.raises(FileNotFoundError, match="数据文件不存在"):
        MixedPDEDataset(bad_cfg, mock_data_dir, mode="train")


def test_missing_required_key_raises(mock_data_dir: Path) -> None:
    """缺必填字段时 KeyError."""
    bad_cfg = [{"name": "burgers", "data_file": "burgers_mock.npy"}]   # 少 flux_type
    with pytest.raises(KeyError, match="flux_type"):
        MixedPDEDataset(bad_cfg, mock_data_dir, mode="train")


def test_empty_pdes_raises(mock_data_dir: Path) -> None:
    """空 pdes_config 拒绝."""
    with pytest.raises(ValueError, match="不能为空"):
        MixedPDEDataset([], mock_data_dir, mode="train")


def test_unknown_mode_raises(mock_data_dir: Path) -> None:
    """未知 mode 拒绝."""
    with pytest.raises(ValueError, match="未知 mode"):
        MixedPDEDataset(_two_pdes_cfg(), mock_data_dir, mode="invalid")


def test_conditioning_none(mock_data_dir: Path) -> None:
    """conditioning='none' 时 get_conditioning 返回 None."""
    ds = MixedPDEDataset(
        pdes_config=_two_pdes_cfg(),
        data_dir=mock_data_dir,
        mode="train",
        conditioning_type="none",
    )
    loader = DataLoader(
        ds, batch_size=4, num_workers=0,
        collate_fn=MixedPDEDataset.collate_fn,
    )
    batch = next(iter(loader))
    assert ds.get_conditioning(batch) is None


def test_collate_rejects_mixed_nx() -> None:
    """batch 内 Nx 不一致时 collate 给清晰报错."""
    bad_batch = [
        {"trajectory": torch.randn(11, 128), "ic": torch.randn(1, 128),
         "x_target": torch.randn(1, 128), "pde_id": 0, "pde_name": "x"},
        {"trajectory": torch.randn(11, 64), "ic": torch.randn(1, 64),
         "x_target": torch.randn(1, 64), "pde_id": 1, "pde_name": "y"},
    ]
    with pytest.raises(ValueError, match="多种 Nx"):
        MixedPDEDataset.collate_fn(bad_batch)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
