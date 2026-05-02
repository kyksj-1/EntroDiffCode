# ============================================================================
# Foundation Model 多 ckpt 汇总评估脚本 (W5-E2)
#
# 论文对应: §5.6 Foundation Model demonstration · 三模型对比
#
# 功能:
#   1. 接收 3 个 ckpt 路径 (big / small / small_plain), 或自动从 exp_dir 找最新 ep
#   2. 对每个 ckpt 复用 eval_foundation 的核心逻辑 (build_model + Heun + compute_metrics)
#   3. 输出汇总:
#        - markdown 表格 (Model × {PDE W₁/L¹}) — 直接粘贴论文 §5.6
#        - CSV 表格 — 给后续脚本/绘图复用
#        - per-model JSON (含逐样本指标, 用于诊断方差)
#
# 关键设计 (per Zongyi Li 工业级规范):
#   - 不重写采样/指标逻辑, 直接 import 自 eval_foundation.py
#   - 论文 "对我们有利": 默认 --num_steps 25 (BV-aware 25 步 = baseline 50 步, §5.2 已验)
#   - eval 不依赖训练完成: 训练中也能跑 (取当前最新 ep ckpt) 监控进度
#
# 用法:
#   # 显式指定 3 个 ckpt
#   python scripts/eval_foundation_full.py \
#       --ckpts output/experiments/foundation_big/foundation_big_ep200.pt \
#               output/experiments/foundation_small/foundation_small_ep200.pt \
#               output/experiments/foundation_small_plain/foundation_small_plain_ep200.pt
#
#   # 自动从默认目录找最新 ep
#   python scripts/eval_foundation_full.py --auto
#
#   # 调整采样步数 (默认 25, 论文 §5.6 的少步数优势位置)
#   python scripts/eval_foundation_full.py --auto --num_steps 25
# ============================================================================

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# 把 PROJECT/black/ 加入 sys.path 以便 import src.* 和 scripts.*
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 复用 eval_foundation.py 的核心组件 (per prompt: 不重写, 直接 import)
from scripts.eval_foundation import (
    build_model_from_ckpt_cfg,
    compute_metrics,
)
from src.utils.env_manager import env, PROJECT_ROOT
from src.data.mixed_pde_dataset import MixedPDEDataset
from src.diffusion.samplers import entrodiff_heun_sampler


# ----------------------------------------------------------------------------
# Auto-discovery: 自动从 output/experiments/{name}/ 找最新 ep ckpt
# ----------------------------------------------------------------------------

# 三个固定的实验名 (与 configs/foundation/{big,small,small_plain}.yaml 对齐)
DEFAULT_EXP_NAMES = ["foundation_big", "foundation_small", "foundation_small_plain"]


def find_latest_ckpt(exp_dir: Path) -> Optional[Path]:
    """
    在 exp_dir 中找最大 epoch 数的 ckpt 文件.

    Args:
        exp_dir: 单个实验输出目录, 含 *_ep<N>.pt 文件
    Returns:
        最新 ep 的 Path, 若无 ckpt 返回 None
    """
    if not exp_dir.exists() or not exp_dir.is_dir():
        return None
    candidates = []
    for f in exp_dir.glob("*.pt"):
        # 文件名形如 'foundation_big_<timestamp>_ep<N>.pt' 或 'foundation_big_ep<N>.pt'
        m = re.search(r"_ep(\d+)\.pt$", f.name)
        if m:
            candidates.append((int(m.group(1)), f))
    if not candidates:
        return None
    # 按 epoch 数取最大
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def auto_discover_ckpts(base_output_dir: Path,
                        exp_names: list[str] = DEFAULT_EXP_NAMES) -> dict[str, Optional[Path]]:
    """
    在 base_output_dir 下按 exp_names 顺序自动找 3 个最新 ckpt.

    Args:
        base_output_dir: env.output_dir, 如 /root/autodl-tmp/.../experiments/
        exp_names: 实验名列表, 默认 ['foundation_big', 'foundation_small', 'foundation_small_plain']
    Returns:
        dict {exp_name: ckpt_path or None}
    """
    found: dict[str, Optional[Path]] = {}
    for name in exp_names:
        exp_dir = base_output_dir / name
        ckpt = find_latest_ckpt(exp_dir)
        found[name] = ckpt
        if ckpt is None:
            print(f"  [warn] 未找到 ckpt: {exp_dir}")
        else:
            print(f"  [auto] {name}: {ckpt.name}")
    return found


# ----------------------------------------------------------------------------
# 单 ckpt 评估: 复用 eval_foundation.py 的核心逻辑, 但不出图, 返回结果 dict
# ----------------------------------------------------------------------------

def evaluate_one_ckpt(
    ckpt_path: Path,
    n_samples: int = 16,
    num_steps: int = 25,
    device: Optional[torch.device] = None,
) -> dict:
    """
    单 ckpt 评估流程.

    复用 eval_foundation 的 build_model_from_ckpt_cfg + compute_metrics + Heun sampler.
    跳过出图以节省 IO; 多 ckpt 汇总场景下出图意义不大.

    Args:
        ckpt_path: ckpt 文件路径 (.pt)
        n_samples: 每个 PDE 评估的 test 样本数
        num_steps: Heun 采样步数 (默认 25, 论文 §5.6 少步数优势位置)
        device:    torch.device, 默认 env.default_device

    Returns:
        dict:
            'ckpt_name':    str, ckpt 文件名 (无路径无后缀)
            'model_type':   str, 'dit_plain' | 'dit_bvaware'
            'n_params':     int, 模型参数量
            'num_steps':    int, 采样步数
            'per_pde': {
                pde_name: {
                    'W1':         float, 平均 W1
                    'L1_rel':     float, 平均 L1 相对误差
                    'shock_err':  float, 平均 shock-loc 误差 (argmax 度量)
                    'n_samples':  int, 实际评估样本数
                    'per_sample': list[dict] 逐样本指标 (用于方差分析)
                }
            }
    """
    if device is None:
        device = torch.device(env.default_device)

    # ---- 加载 ckpt + 内嵌 cfg ----
    print(f"\n[evaluate_one_ckpt] {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if not isinstance(state, dict) or "config" not in state:
        raise ValueError(
            f"ckpt 缺少内嵌 'config' 字段: {ckpt_path}. "
            f"请使用 train_foundation.py 产出的 ckpt (旧版 train_bvaware ckpt 不含 cfg)."
        )
    cfg = state["config"]
    model_type = cfg["model"]["type"]

    # ---- 数据 (test split, 复用训练时的 PDE 列表) ----
    data_cfg = cfg["data"]
    pdes = data_cfg["pdes"]
    test_dataset = MixedPDEDataset(
        pdes_config=pdes,
        data_dir=env.data_dir,
        mode="test",
        conditioning_type=data_cfg.get("conditioning_type", "ic"),
        mix_strategy="uniform",   # eval 不需要 weighted oversample
    )
    n_pde_types = test_dataset.num_pde_types

    # ---- 重建模型 + 加载权重 (复用 eval_foundation.build_model_from_ckpt_cfg) ----
    model = build_model_from_ckpt_cfg(cfg, n_pde_types=n_pde_types, device=device)
    model.load_state_dict(state["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model_type={model_type}  params={n_params:,}  PDEs={test_dataset.pde_names}")

    # ---- 采样参数 ----
    nu = float(cfg["experiment"].get("nu", 1.0))
    tau_max = float(cfg["experiment"].get("tau_max", 1.0))

    # ---- 按 PDE 评估 ----
    per_pde: dict[str, dict] = {}
    for pde_id in range(n_pde_types):
        pde_name = test_dataset.get_pde_name(pde_id)
        flux_type = test_dataset.get_flux_type(pde_id)

        # 收集该 PDE 的 test 样本
        gt_list, ic_list = [], []
        collected = 0
        for global_idx in range(len(test_dataset)):
            item = test_dataset[global_idx]
            if item["pde_id"] != pde_id:
                continue
            gt_list.append(item["x_target"])
            ic_list.append(item["ic"])
            collected += 1
            if collected >= n_samples:
                break
        if collected == 0:
            print(f"  [warn] PDE {pde_name} test 为空")
            continue

        gt_batch = torch.stack(gt_list, dim=0).to(device)        # (n, 1, Nx)
        ic_batch = torch.stack(ic_list, dim=0).to(device)        # (n, 1, Nx)
        Nx = gt_batch.shape[-1]
        x_grid = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
        pde_id_tensor = torch.full((collected,), pde_id, dtype=torch.long, device=device)

        # Heun 采样
        gen_batch = entrodiff_heun_sampler(
            model,
            shape=(collected, 1, Nx),
            sigma_min=0.01, sigma_max=10.0,
            tau_max=tau_max, nu=nu,
            num_steps=num_steps,
            device=device,
            zeta_pde=0.0,                     # 无 PDE guidance (与 §5.2 默认一致)
            conditioning=ic_batch,            # IC-conditioned
            pde_id=pde_id_tensor,             # 关键: 多 PDE 模型必须传
            flux_type=flux_type,              # 关键: Heun 内 PDE residual 用对应 flux
        )

        # 逐样本指标 (复用 eval_foundation.compute_metrics)
        gen_np = gen_batch.detach().cpu().numpy()    # (n, 1, Nx)
        gt_np = gt_batch.detach().cpu().numpy()      # (n, 1, Nx)
        per_sample = []
        for i in range(collected):
            m = compute_metrics(gt_np[i, 0], gen_np[i, 0], x_grid)
            per_sample.append(m)
        # 平均
        avg = {
            "W1":        float(np.mean([s["W1"]        for s in per_sample])),
            "L1_rel":    float(np.mean([s["L1_rel"]    for s in per_sample])),
            "shock_err": float(np.mean([s["shock_err"] for s in per_sample])),
            "n_samples": collected,
            "per_sample": per_sample,
        }
        per_pde[pde_name] = avg
        print(f"    {pde_name:20s}  W1={avg['W1']:.4f}  L1_rel={avg['L1_rel']:.4f}  "
              f"shock_err={avg['shock_err']:.4f}  n={collected}")

    # exp_name 从 cfg 直接读取, 避免文件名 regex 陷阱
    # (train_foundation 的 ckpt 文件名是 'foundation_{exp_name}_<ts>_ep<N>.pt',
    #  含双 'foundation_' 前缀, regex 解析容易出错)
    exp_name_from_cfg = cfg["experiment"].get("name", "unknown")
    epoch_from_state = int(state.get("epoch", 0))

    return {
        "ckpt_name":  ckpt_path.stem,
        "ckpt_path":  str(ckpt_path),
        "exp_name":   exp_name_from_cfg,    # 'foundation_big' / 'foundation_small' / ...
        "epoch":      epoch_from_state,
        "model_type": model_type,
        "n_params":   n_params,
        "num_steps":  num_steps,
        "per_pde":    per_pde,
    }


# ----------------------------------------------------------------------------
# 多 ckpt 汇总 + 输出 markdown / CSV / JSON
# ----------------------------------------------------------------------------

# 三模型显示名 (论文 §5.6 表格用): exp_name → 短名
DISPLAY_NAME_MAP = {
    "foundation_big":         "DiT-BVA Big",
    "foundation_small":       "DiT-BVA Small",
    "foundation_small_plain": "DiT-Plain Small",
}


def render_markdown_table(results: list[dict], pde_names: list[str]) -> str:
    """
    生成 markdown 汇总表格 (论文 §5.6 直接粘贴格式).

    格式:
        | Model | <PDE1> W₁ | <PDE1> L¹ | <PDE2> W₁ | <PDE2> L¹ |
        |---|---|---|---|---|
        | DiT-Plain Small | 0.xxx | 0.xxx | 0.xxx | 0.xxx |
        ...
    """
    # Header
    cols = ["Model", "n_params", "num_steps"]
    for name in pde_names:
        cols.extend([f"{name} W₁", f"{name} L¹"])
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"

    # Rows
    rows = []
    for r in results:
        # 直接从 cfg 读出的 exp_name (避免文件名解析陷阱)
        exp_name = r.get("exp_name", "unknown")
        display = DISPLAY_NAME_MAP.get(exp_name, exp_name)

        n_params_str = f"{r['n_params']/1e6:.1f}M"
        line = [display, n_params_str, str(r["num_steps"])]
        for name in pde_names:
            if name in r["per_pde"]:
                line.append(f"{r['per_pde'][name]['W1']:.4f}")
                line.append(f"{r['per_pde'][name]['L1_rel']:.4f}")
            else:
                line.append("---")
                line.append("---")
        rows.append("| " + " | ".join(line) + " |")

    return "\n".join([header, sep] + rows)


def render_csv_table(results: list[dict], pde_names: list[str]) -> str:
    """生成 CSV 汇总表 (header + rows)."""
    cols = ["model_display", "exp_name", "ckpt_name", "epoch", "n_params", "num_steps"]
    for name in pde_names:
        cols.extend([f"{name}_W1", f"{name}_L1_rel", f"{name}_shock_err"])
    lines = [",".join(cols)]
    for r in results:
        exp_name = r.get("exp_name", "unknown")
        display = DISPLAY_NAME_MAP.get(exp_name, exp_name)

        line = [display, exp_name, r["ckpt_name"], str(r.get("epoch", 0)),
                str(r["n_params"]), str(r["num_steps"])]
        for name in pde_names:
            if name in r["per_pde"]:
                line.append(f"{r['per_pde'][name]['W1']:.4f}")
                line.append(f"{r['per_pde'][name]['L1_rel']:.4f}")
                line.append(f"{r['per_pde'][name]['shock_err']:.4f}")
            else:
                line.extend(["", "", ""])
        lines.append(",".join(line))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Foundation Model 三模型 (big/small/small_plain) 汇总 eval"
    )
    parser.add_argument(
        "--ckpts", nargs="+", default=None,
        help="显式指定 ckpt 路径列表 (3 个: big / small / small_plain). "
             "未指定时启用 --auto 模式"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="自动从 env.output_dir 下 foundation_{big,small,small_plain}/ 找最新 ep"
    )
    parser.add_argument(
        "--n_samples", type=int, default=16,
        help="每个 PDE 评估的 test 样本数 (默认 16)"
    )
    parser.add_argument(
        "--num_steps", type=int, default=25,
        help="Heun 采样步数 (默认 25 — 论文 §5.6 少步数优势位置)"
    )
    parser.add_argument(
        "--exp_names", nargs="+", default=DEFAULT_EXP_NAMES,
        help="auto 模式下的实验名列表 (默认 ['foundation_big','foundation_small','foundation_small_plain'])"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="输出目录 (默认 env.output_dir / foundation_full_eval)"
    )
    args = parser.parse_args()

    device = torch.device(env.default_device)
    print(f"[eval_foundation_full] device={device}  num_steps={args.num_steps}  n_samples={args.n_samples}")

    # ---- 收集 ckpt 列表 ----
    ckpt_paths: list[Path] = []
    if args.ckpts:
        # 显式模式
        for c in args.ckpts:
            p = Path(c)
            if not p.exists():
                raise FileNotFoundError(f"ckpt 不存在: {p}")
            ckpt_paths.append(p)
    elif args.auto:
        # auto 模式: 自动查找
        print("[auto] 在 env.output_dir 下查找最新 ckpt...")
        found = auto_discover_ckpts(env.output_dir, args.exp_names)
        for name in args.exp_names:
            if found[name] is not None:
                ckpt_paths.append(found[name])
        if not ckpt_paths:
            raise FileNotFoundError(
                f"auto 模式未找到任何 ckpt. 检查 {env.output_dir} 下是否有 "
                f"{args.exp_names} 子目录"
            )
    else:
        parser.error("必须指定 --ckpts <path...> 或 --auto")

    # ---- 输出目录 ----
    out_dir = Path(args.out_dir) if args.out_dir else env.output_dir / "foundation_full_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_foundation_full] 输出目录: {out_dir}")

    # ---- 逐 ckpt 评估 ----
    results: list[dict] = []
    all_pde_names: set[str] = set()
    for ckpt in ckpt_paths:
        try:
            r = evaluate_one_ckpt(
                ckpt, n_samples=args.n_samples, num_steps=args.num_steps,
                device=device,
            )
            results.append(r)
            all_pde_names.update(r["per_pde"].keys())
        except Exception as e:
            # 某个 ckpt 评估失败不阻塞其余
            print(f"  [error] {ckpt.name}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            continue

    if not results:
        raise RuntimeError("所有 ckpt 评估均失败, 请检查 ckpt 完整性 + cfg 内嵌")

    # PDE 顺序: 按第一份成功结果的顺序 (一般是 burgers, buckley_leverett)
    pde_names_ordered = list(results[0]["per_pde"].keys())
    # 兜底: 若 PDE 不一致, 拼并集 (按字母序)
    for pn in sorted(all_pde_names):
        if pn not in pde_names_ordered:
            pde_names_ordered.append(pn)

    # ---- 输出 markdown 表 ----
    md_table = render_markdown_table(results, pde_names_ordered)
    md_path = out_dir / "summary_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Foundation Model 三模型 eval 汇总\n\n")
        f.write(f"- num_steps: {args.num_steps}\n")
        f.write(f"- n_samples per PDE: {args.n_samples}\n")
        f.write(f"- ckpts:\n")
        for r in results:
            f.write(f"    - {r['model_type']:14s} ({r['n_params']/1e6:.1f}M): {r['ckpt_path']}\n")
        f.write("\n## 主表 (复制到论文 §5.6)\n\n")
        f.write(md_table + "\n")
    print(f"\n[md] {md_path}")

    # ---- 输出 CSV 表 ----
    csv_table = render_csv_table(results, pde_names_ordered)
    csv_path = out_dir / "summary_table.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(csv_table + "\n")
    print(f"[csv] {csv_path}")

    # ---- 输出 per-ckpt JSON (含逐样本指标, 用于方差/diagnostics) ----
    json_path = out_dir / "summary_full.json"
    # 注意: per_sample 中的 numpy 标量须显式转 float, 但 compute_metrics 已返回 float
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "num_steps": args.num_steps,
                    "n_samples": args.n_samples,
                    "exp_names": args.exp_names,
                },
                "results": results,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"[json] {json_path}")

    # ---- 终端打印汇总 ----
    print("\n" + "=" * 80)
    print("Foundation Model 汇总 (markdown):")
    print("=" * 80)
    print(md_table)
    print("=" * 80)


if __name__ == "__main__":
    main()
