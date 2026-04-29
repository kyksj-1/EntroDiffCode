# 快速对比: time_loss on vs off, 10 epoch 训练 + eval
import sys; sys.path.append('.')
import torch, math, numpy as np, time
from src.utils.env_manager import env
from src.data.burgers_dataset import BurgersDataset
from src.models.score_param import StandardScore
from src.diffusion.schedules import ViscosityMatchedSchedule
from src.diffusion.losses import get_dsm_loss, get_bv_loss, get_godunov_time_loss
from src.diffusion.samplers import entrodiff_heun_sampler
from torch.utils.data import DataLoader
import torch.optim as optim
from scipy.stats import wasserstein_distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nu, tau_max = 1.0, 1.0
dt_p, dx_p, nx = 0.005, 2*3.141592653589793/128, 128

# Data
dp = env.data_dir / 'burgers_1d_N5000_Nx128.npy'
train_ds = BurgersDataset(dp, mode='train')
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
test_ds = BurgersDataset(dp, mode='test')
gt = test_ds.data[:4, -1, :]  # 4 test samples

schedule = ViscosityMatchedSchedule(nu=nu, tau_max=tau_max)

for lambda_t in [0.0, 1.0]:
    label = "ours+time" if lambda_t > 0 else "ours (no time)"
    print(f"\n--- {label} ---")
    model = StandardScore(in_channels=1).to(device)
    opt = optim.Adam(model.parameters(), lr=2e-4)
    
    t0 = time.time()
    for ep in range(10):
        model.train()
        for batch in train_loader:
            x_target = batch[:, -1, :].unsqueeze(1).to(device)
            sigmas = schedule.sample_sigma(x_target.shape[0], device)
            opt.zero_grad()
            loss = get_dsm_loss(model, x_target, sigmas) + 0.1 * get_bv_loss(model, x_target, sigmas)
            if lambda_t > 0:
                x_prev = batch[:, -2, :].unsqueeze(1).to(device)
                loss = loss + lambda_t * get_godunov_time_loss(model, x_prev, x_target, sigmas, dt=dt_p, dx=dx_p)
            loss.backward(); opt.step()
        print(f"  ep {ep+1}: loss={loss.item():.4f}", end='  \r')
    
    print(f"\n  train time: {time.time()-t0:.0f}s")
    
    # Eval
    model.eval()
    gen = entrodiff_heun_sampler(model, (4,1,nx), 0.002, math.sqrt(2*nu*tau_max),
                                  tau_max, nu, num_steps=50, device=device, zeta_pde=0.0)
    gen = gen.squeeze().cpu().numpy()
    
    w1s, l1s = [], []
    for i in range(4):
        w1s.append(wasserstein_distance(gt[i], gen[i]))
        l1s.append(np.linalg.norm(gen[i]-gt[i],1)/(np.linalg.norm(gt[i],1)+1e-10))
    print(f"  W1 avg: {np.mean(w1s):.4f}  L1 avg: {np.mean(l1s):.4f}")
    print(f"  Gen range: [{gen.min():.3f}, {gen.max():.3f}]  std={gen.std():.3f}")
