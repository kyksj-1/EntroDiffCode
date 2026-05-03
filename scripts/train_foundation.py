# ============================================================================
# Foundation Model 训练脚本 (W5-D)
#
# 论文对应: §3.5 Foundation Model: DiT Backbone (Docs/black/path_A_method_skeleton.md)
# 工程方案: Docs/black/W5_foundation_engineering_plan.md §2.4.1
#
# 与 train_bvaware.py 的关键区别:
#   1. 数据: MixedPDEDataset (混合 N 个 PDE)
#   2. 模型: FoundationScore (DiT-Plain) 或 BVAwareScore(backbone='dit')
#   3. Loss: get_godunov_time_loss 按 batch 内 pde_id 分桶派遣 flux
#   4. 配置: configs/foundation/{tiny,small,base}.yaml 配置驱动
#
# 兼容性 (W5 plan §0.1):
#   - 不修改 train_mvp.py / train_baseline.py / train_bvaware.py
#   - 复用现有 schedule / loss / sampler (仅利用其新增的 pde_id / flux_type 参数)
#
# 用法:
#   python scripts/train_foundation.py --config configs/foundation/tiny.yaml
#   python scripts/train_foundation.py --config configs/foundation/small.yaml --resume <ckpt>
# ============================================================================

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

# 添加 PROJECT/black 到 path 以便 import src.*
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.data.mixed_pde_dataset import MixedPDEDataset
from src.models.foundation_score import FoundationScore
from src.models.score_param import BVAwareScore
from src.diffusion.schedules import ViscosityMatchedSchedule
from src.diffusion.losses import get_dsm_loss, get_bv_loss, get_godunov_time_loss


def build_model(model_cfg: dict, n_pde_types: int, device: torch.device) -> torch.nn.Module:
    """
    根据 config 构建 score model.

    Args:
        model_cfg:    YAML 'model' 字段 (含 type, in_channels, dit)
        n_pde_types:  从 dataset 自动推导 (data.pdes 长度)
        device:       目标设备

    Returns:
        nn.Module — FoundationScore 或 BVAwareScore(backbone='dit')
    """
    # 复制 dit_kwargs 不污染原 cfg, 注入 n_pde_types
    dit_kwargs = dict(model_cfg["dit"])
    dit_kwargs["n_pde_types"] = n_pde_types

    model_type = model_cfg["type"]
    in_channels = int(model_cfg.get("in_channels", 2))
    Nx = int(dit_kwargs.get("Nx", 128))   # MixedPDEDataset 假设统一 Nx

    if model_type == "dit_plain":
        # FoundationScore = DiT-Plain (EDM precondition + DiT backbone)
        # 与 StandardScore 接口一致, 输出 (B, 1, Nx)
        # 注: dit_kwargs 不应含 Nx (FoundationScore 顶层吃 Nx)
        nx_for_dit = dit_kwargs.pop("Nx", Nx)
        if nx_for_dit != Nx:
            raise ValueError(f"dit.Nx={nx_for_dit} != model.Nx={Nx}")
        model = FoundationScore(
            in_channels=in_channels,
            Nx=Nx,
            dit_kwargs=dit_kwargs,
        )
    elif model_type == "dit_bvaware":
        # BVAwareScore(backbone='dit'): phi_sm 走 DiT, phi_sh/kappa 仍小 conv
        # 输出 (B, in_channels, Nx) (现有 BVAware 行为, sampler+cond 路径有预先问题但训练 OK)
        model = BVAwareScore(
            in_channels=in_channels,
            backbone="dit",
            dit_kwargs=dit_kwargs,
            n_pde_types=n_pde_types,
        )
    else:
        raise ValueError(f"未知 model.type='{model_type}', 可选: dit_plain | dit_bvaware")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [model] type={model_type}  params={n_params:,} ({n_params/1e6:.2f}M)")
    return model


def compute_loss_per_batch(
    model: torch.nn.Module,
    batch: dict,
    schedule: ViscosityMatchedSchedule,
    dataset: MixedPDEDataset,
    loss_cfg: dict,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    单 batch 的 loss 计算.

    L_total = λ_dsm·DSM + λ_bv·BV + λ_time·Σ_{pde}(time_loss_for_pde)

    L_time 的 flux 派遣:
      - 当 batch 内所有 sample 同 pde_id 时, 单次 flux 调用
      - 当 batch 异质时, 按 pde_id 分桶, 每桶单独算 time_loss 再加权平均

    Returns:
        (loss, metrics_dict)
        metrics_dict 包含 'loss_total', 'loss_dsm', 'loss_bv', 'loss_time'
    """
    x_target = batch["x_target"].to(device)         # (B, 1, Nx)
    cond = dataset.get_conditioning(batch)          # (B, 1, Nx) or None
    if cond is not None:
        cond = cond.to(device)
    pde_id = batch["pde_id"].to(device)             # (B,) long
    trajectory = batch["trajectory"].to(device)     # (B, N_time, Nx)

    B = x_target.shape[0]
    sigmas = schedule.sample_sigma(B, device)

    lambda_dsm = float(loss_cfg.get("lambda_dsm", 1.0))
    lambda_bv = float(loss_cfg.get("lambda_bv", 0.0))
    lambda_time = float(loss_cfg.get("lambda_time", 0.0))

    # ---- L_DSM ----
    loss_dsm = get_dsm_loss(model, x_target, sigmas, conditioning=cond, pde_id=pde_id)

    # ---- L_BV ----
    if lambda_bv > 0:
        loss_bv = get_bv_loss(model, x_target, sigmas, conditioning=cond, pde_id=pde_id)
    else:
        loss_bv = torch.tensor(0.0, device=device)

    # ---- L_time (Godunov 一步推进损失, 按 pde_id 分桶派遣 flux) ----
    if lambda_time > 0:
        # x_prev = trajectory[:, -2, :]; x_target = trajectory[:, -1, :] (倒数两帧)
        # 这与 train_bvaware.py 单 PDE 时的逻辑一致 (用最后两帧的时间一致性)
        x_prev = trajectory[:, -2:-1, :]   # (B, 1, Nx) — 注意 N_time ≥ 2 由 dataset 保证
        # dx 取 2π/Nx 与现有约定一致
        Nx = x_target.shape[-1]
        dx = 2.0 * 3.141592653589793 / Nx
        # dt 取 trajectory 时间步: T 总长 / (N_time-1); 这里用单步约定 0.05 (与 BL/Burgers 默认 dt 接近)
        # 工程上 dt 应来自数据生成 cfg, 但当前数据未带元数据 → 用默认值
        dt = 0.05

        # 按 pde_id 分桶
        # batch 内可能同时有 burgers / bl, 必须分别用各自的 flux
        loss_time_total = torch.tensor(0.0, device=device)
        n_buckets = 0
        unique_pde_ids = pde_id.unique().tolist()
        for pid in unique_pde_ids:
            mask = (pde_id == pid)
            if mask.sum() == 0:
                continue
            flux_type = dataset.get_flux_type(int(pid))
            xp_sub = x_prev[mask]
            xt_sub = x_target[mask]
            cond_sub = cond[mask] if cond is not None else None
            sig_sub = sigmas[mask]
            pid_sub = pde_id[mask]

            loss_time_b = get_godunov_time_loss(
                model, xp_sub, xt_sub, sig_sub,
                dt=dt, dx=dx,
                conditioning=cond_sub, pde_id=pid_sub,
                flux_type=flux_type,
            )
            loss_time_total = loss_time_total + loss_time_b
            n_buckets += 1
        loss_time = loss_time_total / max(n_buckets, 1)
    else:
        loss_time = torch.tensor(0.0, device=device)

    # ---- 总 loss ----
    loss = (
        lambda_dsm * loss_dsm
        + lambda_bv * loss_bv
        + lambda_time * loss_time
    )

    metrics = {
        "loss_total": float(loss.detach().item()),
        "loss_dsm": float(loss_dsm.detach().item()),
        "loss_bv": float(loss_bv.detach().item()) if lambda_bv > 0 else 0.0,
        "loss_time": float(loss_time.detach().item()) if lambda_time > 0 else 0.0,
    }
    return loss, metrics


def train(args: argparse.Namespace) -> None:
    """主训练函数."""
    # ---- 设备 + 配置加载 ----
    device = torch.device(env.default_device)
    config_path = (PROJECT_ROOT / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    data_cfg = cfg["data"]

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = exp_cfg["name"]
    seed_tag = f"_s{args.seed}" if hasattr(args, 'seed') and args.seed != 42 else ""
    output_dir = env.output_dir / (exp_name + seed_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(exp_cfg.get("epochs", 50))
    lr = float(exp_cfg.get("learning_rate", 2e-4))
    base_batch_size = int(exp_cfg.get("batch_size", 16))
    grad_accum_steps = int(exp_cfg.get("grad_accum_steps", 1))
    use_amp = bool(exp_cfg.get("use_amp", False))
    nu = float(exp_cfg.get("nu", 1.0))
    tau_max = float(exp_cfg.get("tau_max", 1.0))
    ckpt_every = int(exp_cfg.get("ckpt_every", 10))

    print(f"[train_foundation] config: {config_path}")
    print(f"  output_dir: {output_dir}")
    print(f"  epochs={epochs}  lr={lr}  batch={base_batch_size}  grad_accum={grad_accum_steps}  amp={use_amp}")

    # ---- 数据 ----
    pdes = data_cfg["pdes"]
    # Revisit 实验 (2026-05-04): yaml 顶层 data.data_limit 传给每 PDE
    global_data_limit = data_cfg.get("data_limit", None)
    if global_data_limit is not None:
        for p in pdes:
            p.setdefault("data_limit", global_data_limit)
        print(f"  [data_limit] 全局 train sample 限制: {global_data_limit}")
    train_dataset = MixedPDEDataset(
        pdes_config=pdes,
        data_dir=env.data_dir,
        mode="train",
        conditioning_type=data_cfg.get("conditioning_type", "ic"),
        mix_strategy=data_cfg.get("mix_strategy", "uniform"),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=base_batch_size,
        shuffle=True,
        num_workers=env.num_workers,
        collate_fn=MixedPDEDataset.collate_fn,
    )
    n_pde_types = train_dataset.num_pde_types

    # ---- 模型 + 优化器 + schedule ----
    model = build_model(model_cfg, n_pde_types=n_pde_types, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    schedule = ViscosityMatchedSchedule(nu=nu, tau_max=tau_max)

    # AMP scaler (BVAware 与 GradScaler 不兼容; use_amp=True 时仅 dit_plain 推荐)
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    # ---- Resume 支持 ----
    start_epoch = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume ckpt 不存在: {ckpt_path}")
        print(f"[resume] 加载 {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location=device)
        # 兼容两种 ckpt 格式: (a) 仅 state_dict; (b) {model, optimizer, epoch}
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state.get("epoch", 0))
        else:
            model.load_state_dict(state)
            # 从文件名推断 epoch
            import re
            m = re.search(r"_ep(\d+)", ckpt_path.name)
            start_epoch = int(m.group(1)) if m else 0
        print(f"[resume] 从 epoch {start_epoch} 继续")

    # ---- 日志文件 ----
    log_path = output_dir / f"train_log_{exp_name}_{run_timestamp}.txt"
    log_fp = open(str(log_path), "a" if args.resume else "w", encoding="utf-8")
    if not args.resume:
        log_fp.write(f"# Foundation Model Training Log\n")
        log_fp.write(f"# config: {config_path}\n")
        log_fp.write(f"# model.type: {model_cfg['type']}  PDEs: {train_dataset.pde_names}\n")
        log_fp.write(f"# epoch  loss_total  loss_dsm  loss_bv  loss_time\n")
    log_fp.flush()

    # ---- 训练主循环 ----
    print(f"[train] 开始 (epoch {start_epoch+1} → {epochs})")
    for epoch in range(start_epoch, epochs):
        model.train()
        accum_metrics = {"loss_total": 0.0, "loss_dsm": 0.0, "loss_bv": 0.0, "loss_time": 0.0}
        n_batches = 0

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            # AMP 上下文 (仅当 use_amp=True 且 cuda)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    loss, metrics = compute_loss_per_batch(
                        model, batch, schedule, train_dataset, loss_cfg, device,
                    )
                # grad accum: 缩放 loss 后 backward
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                loss, metrics = compute_loss_per_batch(
                    model, batch, schedule, train_dataset, loss_cfg, device,
                )
                (loss / grad_accum_steps).backward()

            # grad accumulation: 每 grad_accum_steps 才 step
            if (step + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            for k, v in metrics.items():
                accum_metrics[k] += v
            n_batches += 1

        # ---- Epoch 收尾: 平均指标 + 日志 ----
        avg = {k: v / max(n_batches, 1) for k, v in accum_metrics.items()}
        line = (
            f"Epoch {epoch+1}/{epochs} | "
            f"L: {avg['loss_total']:.6f}  "
            f"DSM: {avg['loss_dsm']:.6f}  "
            f"BV: {avg['loss_bv']:.6f}  "
            f"Time: {avg['loss_time']:.6f}"
        )
        print(line)
        log_fp.write(
            f"{epoch+1:4d}  {avg['loss_total']:.8f}  "
            f"{avg['loss_dsm']:.8f}  {avg['loss_bv']:.8f}  {avg['loss_time']:.8f}\n"
        )
        log_fp.flush()

        # 检查点
        if (epoch + 1) % ckpt_every == 0 or (epoch + 1) == epochs:
            ckpt_file = output_dir / f"foundation_{exp_name}_{run_timestamp}_ep{epoch+1}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "config": cfg,                  # 保存配置便于 eval 重建
                "pde_names": train_dataset.pde_names,
            }, str(ckpt_file))
            print(f"  [ckpt] {ckpt_file}")

    log_fp.write(f"# 训练完成. final_epoch={epochs}\n")
    log_fp.close()
    print(f"[train] 完成. 日志: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True,
        help="YAML 配置路径 (相对 PROJECT/black/), e.g. configs/foundation/tiny.yaml"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从 checkpoint 恢复训练 (.pt 路径)"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (revisit 5-seed 实验)")
    args = parser.parse_args()
    # 设置 seed
    import torch
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
