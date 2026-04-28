import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gnorm1 = nn.GroupNorm(8, out_channels)
        self.gnorm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.gnorm1(h)
        h = self.act(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t_emb)
        h = h + time_emb.unsqueeze(-1)
        
        h = self.conv2(h)
        h = self.gnorm2(h)
        h = self.act(h)
        return h

class UNet1D(nn.Module):
    """
    Lightweight 1D U-Net for Burgers PDE Score Estimation.
    Approximates the score function or the denoised state.
    """
    def __init__(self, in_channels=1, out_channels=1, dim=64):
        super().__init__()
        self.time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

        self.init_conv = nn.Conv1d(in_channels, dim, kernel_size=3, padding=1)
        
        # Down
        self.down1 = Block1D(dim, dim, self.time_dim)
        self.down2 = Block1D(dim, dim*2, self.time_dim)
        self.down3 = Block1D(dim*2, dim*4, self.time_dim)
        
        # Mid
        self.mid1 = Block1D(dim*4, dim*4, self.time_dim)
        self.mid2 = Block1D(dim*4, dim*4, self.time_dim)
        
        # Up
        self.up1 = Block1D(dim*4 + dim*4, dim*2, self.time_dim)
        self.up2 = Block1D(dim*2 + dim*2, dim, self.time_dim)
        self.up3 = Block1D(dim + dim, dim, self.time_dim)
        
        self.final_conv = nn.Conv1d(dim, out_channels, kernel_size=1)

    def forward(self, x, time):
        """
        x: (B, 1, Nx)
        time: (B,)
        """
        # Time embedding
        t = self.time_mlp(time)
        
        # Initial conv
        x0 = self.init_conv(x)
        
        # Down
        d1 = self.down1(x0, t)
        d1_pool = F.avg_pool1d(d1, 2)
        
        d2 = self.down2(d1_pool, t)
        d2_pool = F.avg_pool1d(d2, 2)
        
        d3 = self.down3(d2_pool, t)
        d3_pool = F.avg_pool1d(d3, 2)
        
        # Mid
        m1 = self.mid1(d3_pool, t)
        m2 = self.mid2(m1, t)
        
        # Up
        u1_up = F.interpolate(m2, scale_factor=2, mode='linear', align_corners=False)
        u1 = self.up1(torch.cat([u1_up, d3], dim=1), t)
        
        u2_up = F.interpolate(u1, scale_factor=2, mode='linear', align_corners=False)
        u2 = self.up2(torch.cat([u2_up, d2], dim=1), t)
        
        u3_up = F.interpolate(u2, scale_factor=2, mode='linear', align_corners=False)
        u3 = self.up3(torch.cat([u3_up, d1], dim=1), t)
        
        # Final
        out = self.final_conv(u3)
        return out
