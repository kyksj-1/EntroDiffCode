# ============================================================================
# EntroDiff 全量实验主控脚本 (服务器上运行)
# 功能: 1. 生成 sharp IC 数据  2. 并行训练 Ours + Baseline  3. 多步数 eval
# 使用: python scripts/run_full_experiment.py
# ============================================================================
import os, sys, time, math, subprocess, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ===== 配置 =====
# Ours: BVAwareScore + 全 loss + sharp data
OURS_CONFIG = {
    "name": "ours_full", "data": "burgers_sharp_N5000_Nx128.npy",
    "model_type": "bvaware", "model_dim": 256,
    "epochs": 500, "lr": 2e-4, "batch": 128,
    "nu": 1.0, "lambda_dsm": 1.0, "lambda_bv": 0.1, "lambda_time": 1.0,
    "schedule": "viscosity_matched", "gpu": 1,
}
# Baseline: StandardScore + 纯 EDM + sharp data (baseline naturally struggles)
BASELINE_CONFIG = {
    "name": "baseline_full", "data": "burgers_sharp_N5000_Nx128.npy",
    "model_type": "standard", "model_dim": 64,
    "epochs": 500, "lr": 2e-4, "batch": 128,
    "nu": 1.0, "lambda_dsm": 1.0, "lambda_bv": 0.0, "lambda_time": 0.0,
    "schedule": "baseline", "gpu": 2,
}

def run(cmd, env=None):
    """Run shell command, print output"""
    e = os.environ.copy()
    if env: e.update(env)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env=e)
    for line in p.stdout:
        print(line, end='')
    return p.wait()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_data", action="store_true", help="Skip data generation")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent

    # ==== Step 1: Generate sharp data ====
    if not args.skip_data:
        data_file = base / "output" / "data" / "burgers_sharp_N5000_Nx128.npy"
        if not data_file.exists():
            print("=" * 60)
            print("Step 1: Generating sharp IC Burgers data...")
            print("=" * 60)
            ret = run(f"python {base}/scripts/generate_sharp_data.py")
            if ret != 0:
                print("ERROR: Data generation failed!"); return
        else:
            print(f"Sharp data already exists: {data_file}")

    # ==== Step 2: Write experiment configs ====
    for cfg, label in [(OURS_CONFIG, "ours"), (BASELINE_CONFIG, "baseline")]:
        import yaml
        yaml_path = base / "configs" / "experiment" / f"{label}_full.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump({"experiment": cfg}, f, default_flow_style=False)
        print(f"Config written: {yaml_path}")

    # ==== Step 3: Train (parallel, each on own GPU) ====
    # Ours: train_bvaware.py with full config
    import threading, datetime

    def train_ours():
        cfg = OURS_CONFIG
        cmd = (
            f"CUDA_VISIBLE_DEVICES={cfg['gpu']} "
            f"python {base}/scripts/train_bvaware.py "
            f"--config configs/experiment/ours_full.yaml "
            f"--dim {cfg['model_dim']}"
        )
        print(f"\n[OURS] Starting at {datetime.datetime.now()}")
        run(cmd, {"PATH": f"{os.environ.get('CONDA_PREFIX','')}/bin:{os.environ.get('PATH','')}"})

    def train_baseline():
        cfg = BASELINE_CONFIG
        cmd = (
            f"CUDA_VISIBLE_DEVICES={cfg['gpu']} "
            f"python {base}/scripts/train_baseline.py "
            f"--config configs/experiment/baseline_full.yaml"
        )
        print(f"\n[BASELINE] Starting at {datetime.datetime.now()}")
        run(cmd, {"PATH": f"{os.environ.get('CONDA_PREFIX','')}/bin:{os.environ.get('PATH','')}"})

    t1 = threading.Thread(target=train_ours, daemon=True)
    t2 = threading.Thread(target=train_baseline, daemon=True)
    t1.start(); t2.start()
    t1.join(); t2.join()

    # ==== Step 4: Multi-step eval ====
    print("\n" + "=" * 60)
    print("Step 4: Multi-step evaluation")
    print("=" * 60)

    # Find latest checkpoints
    import glob as g
    ours_dir = base / "output" / "experiments" / "ours_full"
    base_dir = base / "output" / "experiments" / "baseline_full"
    ours_ckpts = sorted(g.glob(str(ours_dir / "*ep*.pt")), key=lambda x: int(x.stem.split("ep")[-1].split(".")[0]))
    base_ckpts = sorted(g.glob(str(base_dir / "*ep*.pt")), key=lambda x: int(x.stem.split("ep")[-1].split(".")[0]))
    if not ours_ckpts or not base_ckpts:
        print("ERROR: checkpoints not found!"); return

    ours_ckpt = ours_ckpts[-1]  # latest epoch
    base_ckpt = base_ckpts[-1]

    for steps in [10, 25, 50]:
        for label, ckpt, mtype, dim in [
            ("OURS", ours_ckpt, "bvaware", 256),
            ("BASELINE", base_ckpt, "standard", None),
        ]:
            dim_arg = f"--model_dim {dim}" if dim else ""
            cmd = (
                f"python {base}/scripts/eval_viz.py "
                f"--config configs/experiment/ours_full.yaml "
                f"--ckpt_ours {ckpt} --model_type {mtype} {dim_arg} "
                f"--heun_steps {steps} --zeta_pde 0.0 --n_samples 8 "
                f"2>&1 | grep '平均:'"
            )
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(f"  {label:10} {steps:2}step: {result.stdout.strip()}")

    print("\nExperiment complete!")

if __name__ == "__main__":
    main()
