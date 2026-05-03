# ============================================================================
# E2 Buckley-Leverett 综合评估 + 消融脚本
# 论文对应: 05_experiments.tex §E2 · Cross-PDE
# 功能:
#   1. 对比 E2 Baseline vs Ours vs BV-aware (主表)
#   2. 步数消融 (10/25/50/100 Heun steps)
#   3. Loss 消融 (schedule + param 对比)
# ============================================================================
import os, sys, math, glob
import torch, numpy as np, argparse
from pathlib import Path
from scipy.stats import wasserstein_distance

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.env_manager import env
from src.data.burgers_dataset import BurgersDataset
from src.models.score_param import StandardScore, BVAwareScore
from src.diffusion.samplers import entrodiff_heun_sampler

MODEL_MAP = {"standard": StandardScore, "bvaware": BVAwareScore}
EPS = 1e-10


def load_model(ckpt_path, model_type, dim=128, device="cuda"):
    cls = MODEL_MAP[model_type]
    kwargs = {"in_channels": 1}
    if model_type == "bvaware":
        kwargs["dim"] = dim
        kwargs["return_denoiser"] = True
    model = cls(**kwargs).to(device)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()
    return model


def eval_samples(gt, gen):
    w1s = [wasserstein_distance(gt[i], gen[i]) for i in range(len(gt))]
    l1s = [
        np.linalg.norm(gen[i] - gt[i], 1) / (np.linalg.norm(gt[i], 1) + EPS)
        for i in range(len(gt))
    ]
    return np.mean(w1s), np.mean(l1s)


def run_e2_eval():
    parser = argparse.ArgumentParser(description="E2 Buckley-Leverett 评估")
    parser.add_argument("--ckpt_dir", type=str,
                        default="output/experiments",
                        help="checkpoint 根目录")
    parser.add_argument("--data_file", type=str,
                        default="bl_1d_N5000_Nx128.npy",
                        help="BL 数据文件名")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--tau_max", type=float, default=1.0)
    parser.add_argument("--zeta_pde", type=float, default=0.0)
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--steps", type=str, default="10,25,50,100",
                        help="逗号分隔的步数列表")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    step_list = [int(s) for s in args.steps.split(",")]

    # 加载 BL 测试数据
    data_path = env.data_dir / args.data_file
    if not data_path.exists():
        # 尝试绝对路径
        data_path = Path("/home/liuyanzhi/ljz/EntroDiffCode/output/data") / args.data_file
    print(f"[E2] 加载数据: {data_path}")
    ds = BurgersDataset(str(data_path), mode="test")
    gt = ds.data[: args.n_samples, -1]
    nx = gt.shape[1]
    print(f"[E2] test: {args.n_samples} samples, Nx={nx}")

    # ---------------------------------------------------------------
    # 自动发现 checkpoint
    # ---------------------------------------------------------------
    base = Path(args.ckpt_dir)
    ckpts = {}

    # 命名模式: e2_bl_baseline = 纯EDM, e2_bl_run = StandardScore+loss, e2_bvaware = BVAwareScore
    for label, subdir in [
        ("E2_Baseline(EDM)", "e2_bl_baseline"),
        ("E2_Baseline(EDM_200ep)", "e2_bl_baseline_200ep"),
        ("E2_Ours(StdScore+BV+time)", "e2_bl_run"),
        ("E2_Ours(BVAware_128d)", "e2_bvaware_run"),
        ("E2_Ours(BVAware_retrain)", "e2_bvaware_retrain"),
    ]:
        d = base / subdir
        if not d.exists():
            continue
        pts = sorted(d.glob("*ep*.pt"))
        if not pts:
            continue
        # 取最高 epoch
        ckpts[label] = str(pts[-1])
        print(f"[E2] 发现 {label}: {pts[-1].name}")

    if not ckpts:
        print("[E2] 未找到任何 checkpoint，退出")
        return

    # ---------------------------------------------------------------
    # 判定模型类型: bvaware 子目录 = BVAwareScore, 其他 = StandardScore
    # ---------------------------------------------------------------
    results = {}
    for label, ckpt_path in ckpts.items():
        mtype = "bvaware" if "bvaware" in label.lower() else "standard"
        dim = args.model_dim if mtype == "bvaware" else None

        try:
            model = load_model(ckpt_path, mtype, dim or 64, device)
        except Exception as e:
            print(f"[E2] 跳过 {label}: 模型加载失败 ({e})")
            continue

        # 默认 50 步
        print(f"[E2] 评估 {label} (50 steps)...")
        gen = (
            entrodiff_heun_sampler(
                model,
                (args.n_samples, 1, nx),
                sigma_min=0.002,
                sigma_max=math.sqrt(2 * args.nu * args.tau_max),
                tau_max=args.tau_max,
                nu=args.nu,
                num_steps=50,
                device=device,
                zeta_pde=args.zeta_pde,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        w1_50, l1_50 = eval_samples(gt, gen)
        results[label] = {"W1_50": w1_50, "L1_50": l1_50}
        print(f"  W1={w1_50:.4f}, L1={l1_50:.4f}")

        # 步数消融
        step_results = {}
        for ns in step_list:
            if ns == 50:
                step_results[ns] = (w1_50, l1_50)
                continue
            print(f"  {ns} steps...")
            gen_s = (
                entrodiff_heun_sampler(
                    model,
                    (args.n_samples, 1, nx),
                    0.002,
                    math.sqrt(2 * args.nu * args.tau_max),
                    args.tau_max,
                    args.nu,
                    num_steps=ns,
                    device=device,
                    zeta_pde=args.zeta_pde,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            w1_s, l1_s = eval_samples(gt, gen_s)
            step_results[ns] = (w1_s, l1_s)
            print(f"    W1={w1_s:.4f}, L1={l1_s:.4f}")
        results[label]["steps"] = step_results

    # ---------------------------------------------------------------
    # 打印汇总表
    # ---------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  E2 Buckley-Leverett 综合结果")
    print("=" * 75)
    print(f"{'Model':<35} {'W1@50':>8} {'L1@50':>8}")
    print("-" * 55)
    for label, r in results.items():
        print(f"{label:<35} {r['W1_50']:>8.4f} {r['L1_50']:>8.4f}")

    if any("steps" in r for r in results.values()):
        print("\n--- 步数消融 ---")
        header = f"{'Model':<35}"
        for ns in step_list:
            header += f" {'W1@'+str(ns):>9}"
        print(header)
        print("-" * (35 + 10 * len(step_list)))
        for label, r in results.items():
            if "steps" not in r:
                continue
            line = f"{label:<35}"
            for ns in step_list:
                if ns in r["steps"]:
                    line += f" {r['steps'][ns][0]:>9.4f}"
                else:
                    line += f" {'---':>9}"
            print(line)

    print("\n[DONE] E2 eval complete.")


if __name__ == "__main__":
    run_e2_eval()
