# ============================================================================
# E2 Buckley-Leverett 消融评估脚本
# 对比: (A) Baseline [EDM schedule, StdScore, L_DSM]
#        (B) Ours  [VM schedule, BV-aware, L_DSM + L_BV]
# 纸张对应: 05_experiments.tex §Ablation
# ============================================================================
import sys; sys.path.append('.')

def run_e2_ablation():
    import math, numpy as np
    from scipy.stats import wasserstein_distance
    import torch, torch.nn as nn, argparse
    from src.models.score_param import BVAwareScore, StandardScore
    from src.diffusion.samplers import entrodiff_heun_sampler

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='output/data/bl_1d_N5000_Nx128.npy')
    parser.add_argument('--ckpt_dir', type=str, default='output/experiments')
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--nu', type=float, default=1.0)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = np.load(args.data)
    n_test = int(0.9 * data.shape[0])
    gt = data[n_test:n_test + args.n, -1, :]
    nx = gt.shape[1]
    sigma_max = math.sqrt(2 * args.nu * 1.0)

    # Wrapper for models with IC=2 channels
    class W(nn.Module):
        def __init__(self, m, ic):
            super().__init__(); self.m = m; self.ic = ic
        def forward(self, x, sigma, **kw):
            if self.ic == 2 and x.shape[1] == 1:
                x = torch.cat([x, torch.zeros_like(x)], dim=1)
                o = self.m(x, sigma)
                return o[:, :1, :]
            return self.m(x, sigma)

    # --- Ablation Table ---
    configs = [
        # (label, path, cls, dim, ic_desc)
        ('E2_Baseline(EDM+StdScore+L_DSM)',
         f'{args.ckpt_dir}/e2_bl_baseline_200ep/entrodiff_e2_bl_baseline_200ep_20260502_131345_ep200.pt',
         StandardScore, 64, 'StandardScore'),

        ('E2_BV-aware(VM+BVAware+L_DSM+L_BV)',
         f'{args.ckpt_dir}/e2_bvaware_retrain/entrodiff_e2_bvaware_retrain_20260504_011311_ep200.pt',
         BVAwareScore, 128, 'BVAwareScore'),
    ]

    import os, glob
    # Also try to find any additional E2 models
    for d in sorted(os.listdir(args.ckpt_dir)):
        if not d.startswith('e2_'): continue
        pts = sorted(glob.glob(f'{args.ckpt_dir}/{d}/*ep*.pt'))
        if not pts: continue
        if any(pts[-1] == c[1] for c in configs): continue
        # Skip if directory already covered by explicit configs
        covered = False
        for c in configs:
            if d in c[1]: covered = True; break
        if covered: continue
        
        # Auto-detect model type from keys
        sd_check = torch.load(pts[-1], map_location='cpu')
        is_bv = any('phi_sm_net' in k for k in sd_check.keys())
        cls_check = BVAwareScore if is_bv else StandardScore
        dim_check = 128 if 'bv' in d or is_bv else 64
        configs.append((d, pts[-1], cls_check, dim_check))

    print(f'{"Model":<55} {"W1@50":>8} {"L1@50":>8} {"W1@25":>8} {"W1@10":>8}')
    print('-' * 95)
    results = {}

    for label, ckpt, cls, dim, _ in configs:
        if not os.path.exists(ckpt):
            print(f'{label:<55} {"SKIP":>8}')
            continue
        sd = torch.load(ckpt, map_location='cpu')
        # Auto-detect IC channels
        conv_k = [k for k in sd if 'init_conv.weight' in k or 'phi_sm_net.init_conv.weight' in k]
        ic = sd[conv_k[0]].shape[1] if conv_k else 1
        model = cls(in_channels=ic, dim=dim, return_denoiser=True).to(device) if cls == BVAwareScore else cls(in_channels=ic).to(device)
        model.load_state_dict(sd); model.eval()
        wrap = W(model, ic)

        w1s, l1s = {}, {}
        for ns in [10, 25, 50]:
            gen = entrodiff_heun_sampler(wrap, (args.n, 1, nx), 0.002, sigma_max, 1.0, args.nu, ns, device=device)
            gen = gen.squeeze().cpu().numpy()
            w1s[ns] = np.mean([wasserstein_distance(gt[i], gen[i]) for i in range(args.n)])
            l1s[ns] = np.mean([np.linalg.norm(gen[i] - gt[i], 1) / (np.linalg.norm(gt[i], 1) + 1e-10) for i in range(args.n)])

        line = f'{label:<55} {w1s[50]:>8.4f} {l1s[50]:>8.4f} {w1s[25]:>8.4f} {w1s[10]:>8.4f}'
        print(line)
        results[label] = {'w1': w1s, 'l1': l1s}

    # Ablation breakdown
    print('\n=== 消融分析 ===')
    base_w1 = results.get(list(results.keys())[0], {}).get('w1', {}).get(50, 1)
    for label, r in results.items():
        w1 = r['w1'].get(50, 1)
        delta = (base_w1 - w1) / base_w1 * 100
        print(f'{label[:60]}: W1={w1:.4f} (vs baseline: {delta:+.1f}%)')
    print('DONE')

if __name__ == '__main__':
    run_e2_ablation()
