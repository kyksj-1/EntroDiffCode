import torch
from torch.utils.data import Dataset
import numpy as np

class BurgersDataset(Dataset):
    """
    A PyTorch Dataset for the 1D Inviscid Burgers Equation data.
    Loads generated Godunov solutions (.npy format typically) and provides
    time-snapshot pairs (u(t), u) or entire trajectories for models.

    Args:
        data_path (str or Path): Path to the saved Godunov data.
        mode (str): 'train', 'val', or 'test'. Provides data splits if needed.
    """
    def __init__(self, data_path, mode='train'):
        super().__init__()
        self.data_path = str(data_path)
        self.mode = mode
        
        # Load the data (shape depends on how generate_data.py saves it. 
        # Typically shape: [N_samples, N_time, N_x])
        
        print(f"[BurgersDataset] Loading data from {self.data_path} for mode: {mode}")
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
