
import sys; sys.path.append('.')
import torch, math, numpy as np
from scipy.stats import wasserstein_distance
from src.utils.env_manager import env
from src.data.burgers_dataset import BurgersDataset
from src.models.score_param import StandardScore, BVAwareScore
from src.diffusion.samplers import entrodiff_heun_sampler
from src.diffusion.schedules import ViscosityMatchedSchedule

device = torch.device('cuda')
nu, tau_max = 1.0, 1.0
n_sample = 8
x_grid = np.linspace(0, 2*np.pi, 128, endpoint=False)
shock_radius = 10  # +/- grid points around shock center

# Load test data
ds = BurgersDataset(env.data_dir/"burgers_sharp_N5000_Nx128.npy", mode='test')
gt = ds.data[:n_sample, -1, :]

def shock_region_w1(gt_u, gen_u, radius=10):
    """W1 computed only around shock region (max |grad|)"""
    grad_gt = np.abs(np.gradient(gt_u, x_grid))
    shock_idx = np.argmax(grad_gt)
    lo = max(0, shock_idx - radius)
    hi = min(128, shock_idx + radius + 1)
    return wasserstein_distance(gt_u[lo:hi], gen_u[lo:hi])

def max_pointwise_err(gt_u, gen_u):
    return np.max(np.abs(gt_u - gen_u))

def tv_ratio(gt_u, gen_u):
    tv_gt = np.sum(np.abs(np.diff(gt_u)))
    tv_gen = np.sum(np.abs(np.diff(gen_u)))
    return tv_gen / (tv_gt + 1e-10)

# Load models
models = {}
for label, cls, ckpt, dim in [
    ("OURS", BVAwareScore, "output/experiments/ours_full/entrodiff_ours_full_20260429_223734_ep500.pt", 128),
    ("BASE", StandardScore, "output/experiments/mvp_baseline/entrodiff_mvp_baseline_20260429_223736_ep500.pt", None),
]:
    kwargs = {"in_channels": 1}
    if dim: kwargs["dim"] = dim; kwargs["return_denoiser"] = True
    m = cls(**kwargs).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval()
    models[label] = m
    print(f"Loaded {label}: {sum(p.numel()):,} params")

# Run comparison
print("\n========== Shock-Region Comparison (sharp IC) ==========")
print(f"{'Method':<6} {'Steps':>5} {'W1_glob':>8} {'W1_shock':>9} {'MaxErr':>8} {'TV_ratio':>8}")
print("-"*50)

for steps in [50, 25, 10, 5]:
    for label, model in models.items():
        gen = entrodiff_heun_sampler(model, (n_sample,1,128), 0.002,
                                      math.sqrt(2*nu*tau_max), tau_max, nu,
                                      num_steps=steps, device=device, zeta_pde=0.0)
        gen = gen.squeeze().cpu().numpy()
        
        w1_g = np.mean([wasserstein_distance(gt[i], gen[i]) for i in range(n_sample)])
        w1_s = np.mean([shock_region_w1(gt[i], gen[i], shock_radius) for i in range(n_sample)])
        me   = np.mean([max_pointwise_err(gt[i], gen[i]) for i in range(n_sample)])
        tv   = np.mean([tv_ratio(gt[i], gen[i]) for i in range(n_sample)])
        
        print(f"{label:<6} {steps:>5} {w1_g:>8.4f} {w1_s:>9.4f} {me:>8.4f} {tv:>8.4f}")

print("\nDone.")
