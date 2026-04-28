import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env
from src.data.burgers_dataset import BurgersDataset
from src.models.score_param import StandardScore
from src.diffusion.schedules import ViscosityMatchedSchedule
from src.diffusion.losses import get_dsm_loss, get_bv_loss

def train_mvp():
    """
    MVP Training Loop for EntroDiff: 
    1. Loads Burgers Data
    2. Uses StandardScore (1D U-Net)
    3. Optimizes L_DSM + L_BV (TV Proxy)
    """
    # 1. Configs
    device = torch.device(env.default_device)
    batch_size = env._config["hardware"].get("max_batch_size", 64)
    num_workers = env.num_workers
    data_path = env.data_dir / "burgers_1d_N5000_Nx128.npy"
    output_dir = env.output_dir / "mvp_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = 10  # MVP test scale
    lr = 2e-4
    nu = 0.01

    if not data_path.exists():
        print(f"Data not found at {data_path}. Run generate_data.py first.")
        return

    # 2. Data
    print("Loading Dataset...")
    train_dataset = BurgersDataset(data_path, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 3. Model & Schedule
    print("Initializing Model...")
    model = StandardScore(in_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    schedule = ViscosityMatchedSchedule(nu=nu, tau_max=1.0)
    
    lambda_bv = 0.1  # TV penalty (MVP proxy for entropy constraints)

    print(f"Starting Training on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_dsm = 0
        total_bv = 0
        
        for batch in train_loader:
            # target distribution rho_T(u). The paper locks target diffusion distribution
            # to physical states defined by the terminal time T = 0.5 sec.
            # We align precisely with this interpretation.
            x = batch[:, -1, :].unsqueeze(1).to(device) # Shape [B, 1, Nx]

            optimizer.zero_grad()
            
            # Sample continuous time tau ~ U[0, T_d]
            sigmas = schedule.sample_sigma(x.shape[0], device)
            
            # DSM Loss
            loss_dsm = get_dsm_loss(model, x, sigmas)
            
            # BV (Total-Variation) Loss
            loss_bv = get_bv_loss(model, x, sigmas)
            
            loss = loss_dsm + lambda_bv * loss_bv
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_dsm += loss_dsm.item()
            total_bv += loss_bv.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} "
              f"(DSM: {total_dsm/len(train_loader):.4f}, BV/TV: {total_bv/len(train_loader):.4f})")

        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = output_dir / f"entrodiff_mvp_ep{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    train_mvp()
