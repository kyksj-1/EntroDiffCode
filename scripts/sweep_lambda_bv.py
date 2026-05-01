"""
λ_BV 参数 sweep launcher (W5-SA4).

用法:
  python scripts/sweep_lambda_bv.py --base_config configs/experiment/bvaware_server.yaml \\
                                     --lambdas 0.05 0.1 0.2 0.5 \\
                                     [--execute]

dry_run 模式 (默认):
  - 为每个 λ 生成新 yaml: configs/experiment/.sweep/bvaware_lambda_{λ}.yaml
  - 输出 launch_commands.sh 列出所有 train 命令
  - 用户在服务器上 bash launch_commands.sh 即可

execute 模式 (--execute):
  - 同上 + 实际 subprocess 启动 (一般留给服务器 tmux 执行, 本地 dry_run 验证即可)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_config", type=str, required=True,
        help="基础 YAML 配置 (作为 sweep 起点)",
    )
    parser.add_argument(
        "--lambdas", type=float, nargs="+", required=True,
        help="λ_BV 值列表, 例如 --lambdas 0.05 0.1 0.2 0.5",
    )
    parser.add_argument(
        "--output_sweep_dir", type=str, default="configs/experiment/.sweep",
        help="生成的 sweep yaml 存放目录",
    )
    parser.add_argument(
        "--launch_script", type=str, default="launch_lambda_sweep.sh",
        help="生成的 launch 脚本路径",
    )
    parser.add_argument(
        "--train_script", type=str, default="scripts/train_bvaware.py",
        help="训练脚本路径 (sweep 中调用)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="实际启动训练 (默认 dry_run, 仅生成 yaml + sh)",
    )
    args = parser.parse_args()

    base_path = Path(args.base_config)
    if not base_path.exists():
        sys.exit(f"基础配置不存在: {base_path}")

    sweep_dir = Path(args.output_sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    with open(base_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # 为每个 λ 生成新 yaml + launch 命令
    launch_lines = ["#!/usr/bin/env bash", "# λ_BV sweep launch commands", "set -e", ""]
    for lam in args.lambdas:
        new_cfg = _deepcopy_yaml(base_cfg)
        # 修改 lambda_bv (兼容多种字段路径)
        if "experiment" in new_cfg:
            new_cfg["experiment"]["lambda_bv"] = float(lam)
            new_cfg["experiment"]["name"] = (
                f"{base_cfg['experiment'].get('name', 'sweep')}_lam{lam}"
            )
        else:
            new_cfg["lambda_bv"] = float(lam)

        sweep_yaml = sweep_dir / f"bvaware_lambda_{lam}.yaml"
        with open(sweep_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(new_cfg, f, sort_keys=False, allow_unicode=True)

        cmd = f"python {args.train_script} --config {sweep_yaml.as_posix()}"
        launch_lines.append(f"# λ_BV = {lam}")
        launch_lines.append(cmd)
        launch_lines.append("")

        print(f"[sweep] λ={lam} → {sweep_yaml}")

    # 写 launch 脚本
    launch_path = Path(args.launch_script)
    with open(launch_path, "w", encoding="utf-8") as f:
        f.write("\n".join(launch_lines))
    launch_path.chmod(0o755)
    print(f"\n[sweep] launch script → {launch_path}")
    print(f"  服务器执行: bash {launch_path}")

    if args.execute:
        print("\n[sweep] --execute 模式: 启动 subprocess")
        for line in launch_lines:
            if line.startswith("python "):
                print(f"  执行: {line}")
                subprocess.run(line, shell=True, check=False)


def _deepcopy_yaml(cfg: dict) -> dict:
    """yaml 安全深拷贝 (避免共享引用)."""
    return yaml.safe_load(yaml.safe_dump(cfg))


if __name__ == "__main__":
    main()
