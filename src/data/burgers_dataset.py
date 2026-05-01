import torch
from torch.utils.data import Dataset
import numpy as np

class BurgersDataset(Dataset):
    """
    A PyTorch Dataset for the 1D Inviscid Burgers Equation data.
    Loads generated Godunov solutions (.npy format typically) and provides
    time-snapshot pairs (u(t), u) or entire trajectories for models.

    IC-conditioning support: 可通过 get_conditioning() 模块化提取
    初始条件通道，避免训练脚本中硬编码 batch[:, 0, :] 分散在多个位置。

    Args:
        data_path (str or Path): Path to the saved Godunov data.
        mode (str): 'train', 'val', or 'test'. Provides data splits if needed.
        conditioning_type (str): 'none' | 'ic' — 决定 get_conditioning() 的返回类型
    """
    def __init__(self, data_path, mode='train', conditioning_type='none'):
        super().__init__()
        self.data_path = str(data_path)
        self.mode = mode
        self.conditioning_type = conditioning_type  # 条件类型 (none/ic)
        
        # Load the data (shape depends on how generate_data.py saves it. 
        # Typically shape: [N_samples, N_time, N_x])
        
        print(f"[BurgersDataset] Loading data from {self.data_path} for mode: {mode}"
              f"  conditioning: {conditioning_type}")
        data = np.load(self.data_path)
        
        # Simple split: 80% train, 10% val, 10% test
        n_samples = data.shape[0]
        idx_train = int(0.8 * n_samples)
        idx_val = int(0.9 * n_samples)
        
        if mode == 'train':
            self.data = data[:idx_train]
        elif mode == 'val':
            self.data = data[idx_train:idx_val]
        elif mode == 'test':
            self.data = data[idx_val:]
        else:
            raise ValueError(f"Unknown mode {mode}")
            
        print(f"    Loaded {self.data.shape[0]} samples.")
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return the entire spatio-temporal trajectory [N_time, N_x]
        # Or you can modify according to specific diffusion training needs 
        # (e.g., returning the final time state u(T) as target distribution).
        
        item = self.data[idx]
        # Return as (C, V) where C is channel = 1, V is spatial dimension
        # Actually since the target distribution is just the state at some time T
        # usually Diffusion models denoise images [C, H, W] -> [1, Nx]
        
        trajectory = torch.tensor(item, dtype=torch.float32)
        return trajectory

    def get_conditioning(self, batch):
        """
        根据 self.conditioning_type 从 batch 中提取条件张量。
        
        batch: shape [B, N_time, N_x] 的完整时空轨迹
        返回:
            - 'ic' 类型: batch[:, 0, :].unsqueeze(1) → [B, 1, N_x] (首帧作为初始条件)
            - 'none' 类型: None
        返回值形状为 [B, 1, N_x] (含通道维), 可直接拼接到模型输入。
        """
        if self.conditioning_type == "ic":
            return batch[:, 0, :].unsqueeze(1)  # [B, 1, Nx] — 首帧 IC
        # 可扩展: "pde_embedding" 返回 t 值等
        return None
