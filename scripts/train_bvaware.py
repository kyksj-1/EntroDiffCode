# ============================================================================
# EntroDiff BV-aware 训练脚本 (论文核心: §3.2 Eq. 3.2)
# 功能: 使用 BVAwareScore 参数化进行训练 — 这是 EntroDiff 的命门
#       与 StandardScore (EDM baseline) 的本质区别:
#       网络结构中硬编码了 tanh(phi_sh/(2σ²)) 的 interfacial profile,
#       使 shock 结构无需从数据中学, 直接继承 Score Shocks 的精确解析形式
# ============================================================================
import os, sys, re
import torch, torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import yaml, argparse

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.env_manager import env, PROJECT_ROOT
from src.data.burgers_dataset import BurgersDataset
from src.models.score_param import BVAwareScore  # 论文核心: Eq. 3.2 的 BV-aware 参数化
from src.diffusion.schedules import ViscosityMatchedSchedule
from src.diffusion.losses import get_dsm_loss, get_bv_loss  # BVAwareScore 输出 D_x, 兼容现有 loss

def train_bvaware():
    """
    BV-aware 训练主流程 (论文对应: §3.2 + §3.3).

    关键与 StandardScore 训练的区别:
      1. 模型: BVAwareScore 而非 StandardScore
         → 内部有 tanh(φ_sh/2σ²) 的建筑先验
         → 输出 D_x (通过 Tweedie 反演: D_x = x + σ²·s_θ)
         → 兼容现有 loss (get_dsm_loss, get_bv_loss) 和 Heun sampler
      2. 模型容量: dim=128 (vs 64), ~1.6M 参数 (vs 0.4M)
      3. 更多 epoch: 200 (vs 50) 以充分发挥建筑先验的优势
    """
    # ========== 0. 命令行参数 ==========
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment/mvp_burgers.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    parser.add_argument("--dim", type=int, default=128,
                        help="UNet 基础通道数 (PC=64, 服务器=128~256)")
    args = parser.parse_args()

    # ========== 1. 加载配置 ==========
    device = torch.device(env.default_device)
    batch_size = env._config["hardware"].get("max_batch_size", 64)
    num_workers = env.num_workers
    data_path = env.data_dir / "burgers_1d_N5000_Nx128.npy"
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f)["experiment"]

    exp_name = exp_cfg.get("name", "bvaware_run")
    output_dir = env.output_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = exp_cfg.get("epochs", 200)
    lr = float(exp_cfg.get("learning_rate", 2e-4))
    nu = float(exp_cfg.get("nu", 1.0))
    tau_max = float(exp_cfg.get("tau_max", 1.0))
    lambda_bv = float(exp_cfg.get("lambda_bv", 0.1))
    lambda_dsm = float(exp_cfg.get("lambda_dsm", 1.0))

    if not data_path.exists():
        print(f"数据不存在: {data_path}, 请先运行 generate_data.py")
        return

    # ========== 2. 数据加载 ==========
    print(f"加载数据: {data_path}")
    train_dataset = BurgersDataset(data_path, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    # ========== 3. 模型初始化 ==========
    # BVAwareScore: 论文 §3.2 Eq. 3.2 的完整实现
    #   dim 控制模型容量: PC=64 (~0.4M), 服务器=128 (~1.6M), 256 (~6M)
    #   return_denoiser=True → 输出 D_x, 兼容 get_dsm_loss / get_bv_loss / Heun sampler
    print(f"初始化 BVAwareScore (dim={args.dim}, nu={nu}, lambda_bv={lambda_bv}...)")
    model = BVAwareScore(in_channels=1, dim=args.dim, return_denoiser=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    schedule = ViscosityMatchedSchedule(nu=nu, tau_max=tau_max)

    # ---- 恢复训练 ----
    start_epoch = 0
    log_path = output_dir / f"train_log_{exp_name}_{run_timestamp}.txt"
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")
        print(f"[resume] 加载: {ckpt_path}")
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        m = re.search(r'_ep(\d+)', ckpt_path.name)
        start_epoch = int(m.group(1)) if m else 0
        print(f"[resume] 从 epoch {start_epoch} 继续")
        log_fp = open(str(log_path), "a", encoding="utf-8")
        log_fp.write(f"# [resume] epoch {start_epoch} at {run_timestamp}\n")
    else:
        log_fp = open(str(log_path), "w", encoding="utf-8")
        log_fp.write(f"# EntroDiff BV-aware Training Log\n")
        log_fp.write(f"# Model: BVAwareScore dim={args.dim}\n")
        log_fp.write(f"# Params: epochs={epochs} lr={lr} nu={nu} lambda_dsm={lambda_dsm} lambda_bv={lambda_bv}\n")
        log_fp.write(f"# Device: {device}  Batch: {batch_size}\n")
        log_fp.write(f"# epoch  loss_total  loss_dsm  loss_bv\n")
    log_fp.flush()
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 4. 训练主循环 ==========
    print(f"开始训练 (epoch {start_epoch+1} → {epochs}, device={device})...")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, total_dsm, total_bv = 0.0, 0.0, 0.0

        for batch in train_loader:
            # clean target 解 u(T, x), shape [B, 1, Nx]
            x = batch[:, -1, :].unsqueeze(1).to(device)
            optimizer.zero_grad()

            # 采样连续时间 σ ~ viscosity-matched schedule
            sigmas = schedule.sample_sigma(x.shape[0], device)

            # L_DSM: BVAwareScore 返回 D_x, 与 StandardScore 接口一致
            loss_dsm = get_dsm_loss(model, x, sigmas)

            # L_BV: TV 对 BVAwareScore 输出的约束 (tanh 层保证 shock 陡度)
            loss_bv = get_bv_loss(model, x, sigmas)

            loss = lambda_dsm * loss_dsm + lambda_bv * loss_bv
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dsm += loss_dsm.item()
            total_bv += loss_bv.item()

        avg_loss = total_loss / len(train_loader)
        avg_dsm = total_dsm / len(train_loader)
        avg_bv = total_bv / len(train_loader)
        line = (f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} "
                f"(DSM: {avg_dsm:.6f}, BV/TV: {avg_bv:.6f})")
        print(line)
        log_fp.write(f"{epoch+1:4d}  {avg_loss:.8f}  {avg_dsm:.8f}  {avg_bv:.8f}\n")
        log_fp.flush()

        if (epoch + 1) % 5 == 0:
            ckpt_path = output_dir / f"entrodiff_{exp_name}_{run_timestamp}_ep{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → ckpt: {ckpt_path}")

    log_fp.write(f"# 训练完成. Final epoch: {epochs}\n")
    log_fp.close()
    print(f"训练完成. 日志: {log_path}")

if __name__ == "__main__":
    train_bvaware()
