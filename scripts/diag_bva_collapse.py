# ============================================================================
# 诊断 BVA Big/Small 同值异常 (W₁ = 0.0907 完全相同)
#
# 假设:
#   H1 模式塌陷:   两 BVA ckpt 在相同输入下 D_x 输出几乎一致 (cos sim ≈ 1)
#   H2 sampler attractor: 不同 noise 经 Heun 25 步收敛到同一不动点 (内部 std ≈ 0)
#   H3 BV-aware 退化: tanh(phi_sh/2σ²) 被压成 0, 实际等价于 Standard
#
# 流程:
#   1. 加载 3 ckpt 重建 model
#   2. 取 4 burgers + 4 bl test 样本, 同一 seed=42 给所有模型相同 noise
#   3. forward 拿 D_x, 算 cos / L2 / max abs
#   4. 跑 Heun sampler (num_steps=25) 各模型 8 个不同 noise → 输出 std (内部多样性)
#   5. (可选) hook BVAwareScore 内部三项, 看相对贡献
#
# 用法:
#   python scripts/diag_bva_collapse.py
#       [--num_steps 25]   # sampler 步数
# ============================================================================

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.data.mixed_pde_dataset import MixedPDEDataset
from src.diffusion.samplers import entrodiff_heun_sampler
from scripts.eval_foundation import build_model_from_ckpt_cfg


# 三 ckpt 路径 (服务器3 上, 也可在本地用相同结构)
CKPT_PATHS = {
    "BVA_Big":   "output/experiments/foundation_big/foundation_foundation_big_20260503_024957_ep200.pt",
    "BVA_Small": "output/experiments/foundation_small/foundation_foundation_small_20260503_024957_ep200.pt",
    "Plain":     "output/experiments/foundation_small_plain/foundation_foundation_small_plain_20260503_024957_ep200.pt",
}


def cosine_similarity_per_sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """逐样本 cos sim. a, b shape (B, 1, Nx) → (B,)."""
    a_flat = a.flatten(1)
    b_flat = b.flatten(1)
    return F.cosine_similarity(a_flat, b_flat, dim=1)


def load_models() -> dict[str, dict]:
    """加载 3 ckpt + 重建模型, 返回 {name: {model, cfg, state}}."""
    device = torch.device(env.default_device)
    out = {}
    for name, rel_path in CKPT_PATHS.items():
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            # 尝试 absolute fallback (在服务器上 PROJECT_ROOT 即项目根)
            full_path = Path(rel_path) if Path(rel_path).is_absolute() else full_path
            if not full_path.exists():
                raise FileNotFoundError(f"ckpt 不存在: {full_path}")
        print(f"[load] {name}  ←  {full_path}")
        state = torch.load(str(full_path), map_location=device, weights_only=False)
        cfg = state["config"]
        # 假设 n_pde_types=2 (与训练时 mixed PDE 一致)
        model = build_model_from_ckpt_cfg(cfg, n_pde_types=2, device=device)
        model.load_state_dict(state["model"])
        model.eval()
        out[name] = {"model": model, "cfg": cfg, "state": state}
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  type={cfg['model']['type']}  params={n_params/1e6:.2f}M")
    return out


def fetch_test_samples(n_per_pde: int = 4) -> dict:
    """取 burgers + bl 各 n_per_pde 个 test 样本. 返回 dict {pde_name: {ic, x_target}} ."""
    # 用 small 的 cfg 拿 PDE 配置
    state = torch.load(str(PROJECT_ROOT / CKPT_PATHS["BVA_Small"]),
                       map_location="cpu", weights_only=False)
    data_cfg = state["config"]["data"]
    test_dataset = MixedPDEDataset(
        pdes_config=data_cfg["pdes"],
        data_dir=env.data_dir,
        mode="test",
        conditioning_type=data_cfg.get("conditioning_type", "ic"),
        mix_strategy="uniform",
    )
    out = {}
    pde_names = test_dataset.pde_names
    for pid, pname in enumerate(pde_names):
        ic_list, gt_list = [], []
        cnt = 0
        for idx in range(len(test_dataset)):
            item = test_dataset[idx]
            if item["pde_id"] != pid:
                continue
            ic_list.append(item["ic"])
            gt_list.append(item["x_target"])
            cnt += 1
            if cnt >= n_per_pde:
                break
        out[pname] = {
            "ic":         torch.stack(ic_list, dim=0),         # (n, 1, Nx)
            "x_target":   torch.stack(gt_list, dim=0),         # (n, 1, Nx)
            "pde_id":     pid,
        }
        print(f"[data] {pname}: {cnt} samples")
    return out


def h1_test_d_x_collapse(models: dict, samples: dict, sigma_test: float = 1.0):
    """H1: 同一 (x_noisy, IC, σ, pde_id) 下三模型 D_x 是否塌陷."""
    print("\n" + "=" * 76)
    print(f"H1: D_x 输出 cosine / L2 / max-abs-diff (sigma={sigma_test})")
    print("=" * 76)
    device = torch.device(env.default_device)
    torch.manual_seed(42)

    # 拼 burgers + bl 各 4 个 = 8 个测试样本
    all_x_target = []
    all_ic = []
    all_pde_id = []
    for pname, batch in samples.items():
        all_x_target.append(batch["x_target"])
        all_ic.append(batch["ic"])
        n = batch["x_target"].shape[0]
        all_pde_id.extend([batch["pde_id"]] * n)
    x_target = torch.cat(all_x_target, dim=0).to(device)        # (8, 1, Nx)
    ic = torch.cat(all_ic, dim=0).to(device)                    # (8, 1, Nx)
    pde_id = torch.tensor(all_pde_id, dtype=torch.long, device=device)  # (8,)

    # 同一 noise (fixed seed 42)
    noise = torch.randn_like(x_target)
    sigma = torch.full((x_target.shape[0],), sigma_test, device=device)

    # x_noisy = x + σ·ε
    x_noisy = x_target + sigma.view(-1, 1, 1) * noise

    # 拼输入 (B, 2, Nx) = noisy_u + IC
    x_input = torch.cat([x_noisy, ic], dim=1)

    # 三模型 forward
    D_x = {}
    for name, m in models.items():
        with torch.no_grad():
            d = m["model"](x_input.clone(), sigma.clone(), pde_id=pde_id.clone())
        D_x[name] = d
        print(f"  {name}: D_x shape={tuple(d.shape)} mean={d.mean().item():.4f} "
              f"std={d.std().item():.4f}  range=[{d.min().item():.3f}, {d.max().item():.3f}]")

    # 两两 cosine + L2 + max abs diff
    pairs = [("BVA_Big", "BVA_Small"), ("BVA_Big", "Plain"), ("BVA_Small", "Plain")]
    print(f"\n  {'pair':<22} {'cos_mean':>10} {'cos_min':>10} {'L2_mean':>10} {'maxabs':>10}")
    for a, b in pairs:
        cos = cosine_similarity_per_sample(D_x[a], D_x[b])    # (8,)
        l2 = (D_x[a] - D_x[b]).flatten(1).norm(dim=1)         # (8,)
        maxabs = (D_x[a] - D_x[b]).abs().flatten(1).max(dim=1).values
        print(f"  {a:>9} vs {b:<9}  {cos.mean():>10.4f} {cos.min():>10.4f} "
              f"{l2.mean():>10.4f} {maxabs.mean():>10.4f}")

    # 判定
    bb_cos = cosine_similarity_per_sample(D_x["BVA_Big"], D_x["BVA_Small"]).mean().item()
    bp_cos = cosine_similarity_per_sample(D_x["BVA_Big"], D_x["Plain"]).mean().item()
    print(f"\n  *** H1 结论: ", end="")
    if bb_cos > 0.99 and bp_cos < 0.95:
        print(f"模式塌陷! (BVA 两者 cos={bb_cos:.4f} ≈ 1, BVA-Plain cos={bp_cos:.4f})")
    elif bb_cos > 0.99 and bp_cos > 0.99:
        print(f"三模型全部塌陷 (BVA-BVA={bb_cos:.4f}, BVA-Plain={bp_cos:.4f})")
    else:
        print(f"未塌陷 (BVA-BVA cos={bb_cos:.4f}, BVA-Plain cos={bp_cos:.4f})")


def h2_test_sampler_attractor(models: dict, samples: dict, num_steps: int = 25):
    """H2: 不同 noise 跑 Heun sampler, 看每模型输出内部多样性 (std)."""
    print("\n" + "=" * 76)
    print(f"H2: Heun sampler 多 noise 生成 → 内部 std (num_steps={num_steps})")
    print("=" * 76)
    device = torch.device(env.default_device)

    # 用 burgers 的第一个样本 IC 作为条件, 跑 8 个不同 noise
    burgers = samples["burgers"]
    ic_single = burgers["ic"][0:1].to(device)         # (1, 1, Nx)
    pde_id = torch.tensor([0], dtype=torch.long, device=device)  # burgers
    Nx = ic_single.shape[-1]
    n_noises = 8

    # 复制 8 份, 但每个 noise 是不同的 (sampler 内部用 randn_like 起点)
    ic_rep = ic_single.expand(n_noises, -1, -1).contiguous()
    pde_id_rep = pde_id.expand(n_noises)

    # 用每模型 cfg 的 nu/tau_max
    for name, m in models.items():
        cfg = m["cfg"]
        nu = float(cfg["experiment"].get("nu", 1.0))
        tau_max = float(cfg["experiment"].get("tau_max", 1.0))

        torch.manual_seed(42)   # 每模型同一 seed → noise 起点也相同
        gen = entrodiff_heun_sampler(
            m["model"],
            shape=(n_noises, 1, Nx),
            sigma_min=0.01, sigma_max=10.0,
            tau_max=tau_max, nu=nu,
            num_steps=num_steps,
            device=device,
            zeta_pde=0.0,
            conditioning=ic_rep,
            pde_id=pde_id_rep,
            flux_type="burgers",
        )
        # gen: (8, 1, Nx). std across noise dim 0
        std_per_x = gen.std(dim=0)                    # (1, Nx) — 每空间点 noise 间的 std
        std_mean = std_per_x.mean().item()
        std_max = std_per_x.max().item()
        # 还要看输出整体 mean / range
        print(f"  {name:>10}: noise-std mean={std_mean:.5f}  max={std_max:.5f}  "
              f"out range=[{gen.min().item():.3f}, {gen.max().item():.3f}]")

    print(f"\n  *** H2 结论: ", end="")
    print(f"  若 BVA std 远小于 Plain std → sampler 收敛到固定 attractor")


def h3_test_bv_components(models: dict, samples: dict, sigma_test: float = 1.0):
    """H3: BVAware forward 内部三项 (grad_phi_sm / kappa·tanh·grad_phi_sh) 各自范数."""
    print("\n" + "=" * 76)
    print(f"H3: BVAware 内部三项贡献占比 (sigma={sigma_test})")
    print("=" * 76)
    print("  (用 forward hook 监听 phi_sm/phi_sh/kappa 输出, 估算三项相对贡献)")
    device = torch.device(env.default_device)

    # 仅检查 BVA Big 和 BVA Small (Plain 没 BVAware 结构)
    burgers = samples["burgers"]
    ic = burgers["ic"][:4].to(device)
    x_target = burgers["x_target"][:4].to(device)
    sigma = torch.full((4,), sigma_test, device=device)

    torch.manual_seed(42)
    noise = torch.randn_like(x_target)
    x_noisy = x_target + sigma.view(-1, 1, 1) * noise
    x_input = torch.cat([x_noisy, ic], dim=1)
    pde_id = torch.zeros(4, dtype=torch.long, device=device)

    for name in ["BVA_Big", "BVA_Small"]:
        m = models[name]["model"]
        # 检查 m 有没有 BVAware 三件套子网
        if not hasattr(m, "phi_sm_net") or not hasattr(m, "phi_sh_net"):
            print(f"  {name}: 不是 BVAwareScore (跳过)")
            continue
        # 钩子记录 phi_sh, kappa 输出
        captured = {}

        def hook_phi_sh(module, inp, out):
            captured["phi_sh"] = out.detach()

        def hook_kappa(module, inp, out):
            captured["kappa"] = out.detach()

        h1 = m.phi_sh_net.register_forward_hook(hook_phi_sh)
        h2 = m.kappa_net.register_forward_hook(hook_kappa)
        try:
            with torch.no_grad():
                _ = m(x_input.clone(), sigma.clone(), pde_id=pde_id.clone())
        finally:
            h1.remove()
            h2.remove()

        if "phi_sh" not in captured or "kappa" not in captured:
            print(f"  {name}: hook 未捕获到 phi_sh/kappa (跳过)")
            continue

        phi_sh = captured["phi_sh"]              # (4, 1, Nx)
        kappa = captured["kappa"] + 1e-4         # 与 forward 内一致
        # tanh factor
        tanh_factor = torch.tanh(phi_sh / (2 * (sigma**2).view(-1, 1, 1) + 1e-6))
        # tanh 项的"幅度": (kappa/2) * tanh_factor 的 abs.mean
        bv_term_amp = ((kappa / 2.0) * tanh_factor).abs().mean().item()
        # 对比: phi_sh 范围 + kappa 范围
        print(f"  {name}:")
        print(f"    phi_sh:   abs.mean={phi_sh.abs().mean().item():.4f}  std={phi_sh.std().item():.4f}")
        print(f"    kappa:    abs.mean={kappa.abs().mean().item():.4f}  range=[{kappa.min().item():.3f},{kappa.max().item():.3f}]")
        print(f"    tanh_factor: abs.mean={tanh_factor.abs().mean().item():.4f}  (1.0=饱和, ~0=被压成 0)")
        print(f"    BV term ((κ/2)·tanh): abs.mean={bv_term_amp:.4f}")

    print(f"\n  *** H3 结论: 若 tanh_factor.abs() ≈ 0 → BV-aware 退化为 Standard")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=25,
                        help="H2 sampler 步数 (默认 25, 与论文 §5.6 一致)")
    parser.add_argument("--sigma_test", type=float, default=1.0,
                        help="H1/H3 测试用的 σ 值 (默认 1.0, 中间噪声水平)")
    args = parser.parse_args()

    print("=" * 76)
    print("BVA Big/Small 同值异常诊断")
    print(f"device: {env.default_device}")
    print("=" * 76)

    models = load_models()
    samples = fetch_test_samples(n_per_pde=4)

    # H1: D_x 塌陷
    h1_test_d_x_collapse(models, samples, sigma_test=args.sigma_test)
    # 不同 σ 也试一下
    h1_test_d_x_collapse(models, samples, sigma_test=0.1)
    h1_test_d_x_collapse(models, samples, sigma_test=5.0)

    # H2: sampler attractor
    h2_test_sampler_attractor(models, samples, num_steps=args.num_steps)

    # H3: BV-aware 内部
    h3_test_bv_components(models, samples, sigma_test=args.sigma_test)
    h3_test_bv_components(models, samples, sigma_test=0.1)
    h3_test_bv_components(models, samples, sigma_test=5.0)


if __name__ == "__main__":
    main()
