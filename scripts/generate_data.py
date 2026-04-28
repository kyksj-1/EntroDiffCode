import os
import sys
import numpy as np
from pathlib import Path

# Setup paths to ensure src/ is discoverable
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.env_manager import env
from src.data.burgers_1d_solver import burgers_godunov_1d
from tqdm import tqdm

def generate_burgers_data(n_samples=1000, nx=256, nt=100, dt=0.001, dx=2*np.pi/256):
    """
    Generates a dataset of 1D inviscid Burgers solutions starting from a mix of sines
    and random smooth curves. Creates shocks over time.
    """
    print(f"Generating {n_samples} samples of 1D Burgers equation...")
    print(f"Grid: nx={nx}, nt={nt}, dx={dx:.4f}, dt={dt:.4f}")
    
    # CFL condition check for Godunov
    
    # Output structure
    data_hist = np.zeros((n_samples, nt + 1, nx), dtype=np.float32)
    
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    
    for i in tqdm(range(n_samples)):
        # Random initial condition: superposition of Fourier modes
        k_max = 5
        A = np.random.randn(k_max)
        B = np.random.randn(k_max)
        
        # Scale to avoid immediate explosion and huge CFL failures
        # u(x,0) values between [-1.5, 1.5] approx
        A /= (np.arange(1, k_max+1)**2)
        B /= (np.arange(1, k_max+1)**2)
        
        u0 = np.zeros(nx)
        for k in range(1, k_max+1):
            u0 += A[k-1] * np.sin(k * x) + B[k-1] * np.cos(k * x)
            
        u_max = np.max(np.abs(u0))
        cfl = u_max * dt / dx
        
        if cfl > 0.9:
            # Rescale to satisfy CFL
            u0 = u0 * (0.9 / cfl)
            
        # Solve
        u_hist = burgers_godunov_1d(u0, nx=nx, nt=nt, dx=dx, dt=dt)
        data_hist[i] = u_hist
        
    return data_hist

if __name__ == "__main__":
    # Settings for the MVP
    NX = 128
    NT = 100
    DT = 0.005 # Total time T = 0.5 sec
    DX = 2*np.pi / NX
    N_SAMPLES = 5000 
    
    output_path = env.data_dir / "burgers_1d_N5000_Nx128.npy"
    
    data = generate_burgers_data(n_samples=N_SAMPLES, nx=NX, nt=NT, dt=DT, dx=DX)
    
    print(f"Saving generated data to {output_path} (shape: {data.shape})")
    np.save(output_path, data)
    print("Done.")
