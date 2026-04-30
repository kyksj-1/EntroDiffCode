import sys; sys.path.append('.')
import torch, math, numpy as np, time
from src.utils.env_manager import env
from src.data.burgers_dataset import BurgersDataset
from src.models.score_param import StandardScore
from src.diffusion.schedules import ViscosityMatchedSchedule
from src.diffusion.losses import get_dsm_loss, get_bv_loss
from src.diffusion.samplers import entrodiff_heun_sampler
from torch.utils.data import DataLoader; import torch.optim as optim
from scipy.stats import wasserstein_distance

device=torch.device('cuda'); nu=1.0; nx=128
dp=env.data_dir/'burgers_1d_N5000_Nx128.npy'
train_ds=BurgersDataset(dp,mode='train'); test_ds=BurgersDataset(dp,mode='test')
gt=test_ds.data[:4,-1,:]; test_ic=test_ds.data[:4,0,:]
loader=DataLoader(train_ds,batch_size=32,shuffle=True,num_workers=0)

for label, use_ic in [('NO IC',False),('WITH IC',True)]:
    ch=2 if use_ic else 1; model=StandardScore(in_channels=ch).to(device)
    opt=optim.Adam(model.parameters(),lr=2e-4)
    sched=ViscosityMatchedSchedule(nu=nu,tau_max=1.0)
    t0=time.time()
    for ep in range(10):
        for batch in loader:
            xt=batch[:,-1,:].unsqueeze(1).to(device)
            icb=batch[:,0,:].unsqueeze(1).to(device) if use_ic else None
            sig=sched.sample_sigma(xt.shape[0],device); opt.zero_grad()
            loss=get_dsm_loss(model,xt,sig,ic=icb)
            if use_ic: loss=loss+0.1*get_bv_loss(model,xt,sig,ic=icb)
            loss.backward(); opt.step()
    print(f'{label}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)')
    model.eval()
    ic_t=torch.tensor(test_ic,device=device).unsqueeze(1) if use_ic else None
    gen=entrodiff_heun_sampler(model,(4,1,nx),0.002,math.sqrt(2*nu),1.0,nu,num_steps=50,device=device,ic=ic_t)
    gen=gen.squeeze().cpu().numpy()
    w1s=[wasserstein_distance(gt[i],gen[i]) for i in range(4)]
    print(f'  W1 avg={np.mean(w1s):.4f} range=[{gen.min():.3f},{gen.max():.3f}]')
    print()
