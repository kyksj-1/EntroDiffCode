# ============================================================================
# Euler 1D Sod 数据集 (W5-SA2)
#
# 论文对应: §5.4 / A2.2 E3 Euler Sod 实验
#
# 数据格式 (与 generate_euler_data.py 输出对齐):
#   shape: (N, N_time, 3, Nx)  — 3 = 守恒变量 (ρ, ρu, E)
#
# 与 BurgersDataset 的关键区别:
#   - 多了一个 "通道" 维度 (3 conservative variables)
#   - get_conditioning 返回 3-通道 IC (而非单通道)
#   - x_target 也是 3 通道 (末帧守恒变量)
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class EulerDataset(Dataset):
    """
    1D Euler 系统数据集. 与 BurgersDataset 同款 80/10/10 切分.

    Args:
        data_path:           .npy 文件路径 (shape (N, N_time, 3, Nx))
        mode:                'train' | 'val' | 'test'
        conditioning_type:   'none' | 'ic' (复用 BurgersDataset 选项)
        n_components:        守恒变量数 (默认 3 for Euler: ρ, ρu, E)
                             显式参数让接口可扩展到其他系统 PDE (e.g. SW=3, MHD=8)

    返回 (per __getitem__):
        trajectory: (N_time, n_components, Nx)  完整时空轨迹
    """

    def __init__(
        self,
        data_path: Path | str,
        mode: str = "train",
        conditioning_type: str = "ic",
        n_components: int = 3,
    ) -> None:
        super().__init__()
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"未知 mode: {mode}")
        if conditioning_type not in {"none", "ic"}:
            raise ValueError(f"未知 conditioning_type: {conditioning_type}")

        self.data_path = str(data_path)
        self.mode = mode
        self.conditioning_type = conditioning_type
        self.n_components = n_components

        print(f"[EulerDataset] 加载 {self.data_path} mode={mode} cond={conditioning_type}")
        data = np.load(self.data_path)

        # 期望 shape (N, N_time, 3, Nx); 不一致就报错
        if data.ndim != 4:
            raise ValueError(
                f"Euler 数据应为 4D (N, N_time, n_comp, Nx), 实际 ndim={data.ndim}"
            )
        if data.shape[2] != n_components:
            raise ValueError(
                f"Euler 数据通道数 {data.shape[2]} 与 n_components={n_components} 不匹配"
            )

        # 80/10/10 切分 (与 BurgersDataset 完全一致)
        n_samples = data.shape[0]
        idx_train = int(0.8 * n_samples)
        idx_val = int(0.9 * n_samples)
        if mode == "train":
            self.data = data[:idx_train]
        elif mode == "val":
            self.data = data[idx_train:idx_val]
        else:
            self.data = data[idx_val:]

        print(f"    Loaded {self.data.shape}")

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        # 直接返回完整 trajectory (N_time, 3, Nx)
        # 训练脚本中按 first/last frame 分别取 IC / target
        return torch.tensor(self.data[idx], dtype=torch.float32)

    def get_conditioning(self, batch: torch.Tensor) -> Optional[torch.Tensor]:
        """
        提取条件张量.

        Args:
            batch: (B, N_time, n_components, Nx)
        Returns:
            (B, n_components, Nx) 首帧 IC, 或 None
        """
        if self.conditioning_type == "ic":
            # 首帧, 全部 n_components 通道
            return batch[:, 0, :, :]   # (B, n_comp, Nx)
        return None
