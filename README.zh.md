# EntroDiff (MVP 代码库)

**EntroDiff: Entropy-aware diffusion for hyperbolic PDEs**（目标投稿 NeurIPS 2026）最小可行产品代码库。

本代码库专门设计为跨环境（PC、独立服务器、Colab、Kaggle）鲁棒运行。我们强调配置与代码分离，严格遵守确定性多环境设置。

## 1. 快速上手（4个基础步骤）

项目配置强调**配置与代码分离**，绝不硬编码绝对路径。请按以下 4 步在本地跑通最小可行产品（MVP）：

### 第 1 步：安装依赖包
在终端中进入 `PROJECT/black` 目录，并以开发模式安装项目及测试依赖：
```bash
cd PROJECT/black
pip install -e .[dev]
```

### 第 2 步：配置本地机器的专属路径
为了在多台电脑间无缝切换，我们需要一个不提交到 Git 的本地专属配置文件：
1. 复制模板文件：
   ```bash
   cp configs/env_config.yaml.example configs/env_config.yaml
   ```
2. 打开 `configs/env_config.yaml`，将 `data_dir` 和 `output_dir` 修改为你本地的绝对路径（例如 `D:/A-Nips-Diffussion/Output/black/data`）。
3. 如果在此电脑上没有 GPU 或显存太小，可设置 `device: "cpu"`。Windows 下建议设置 `num_workers: 0` 防止多进程加载报错。

### 第 3 步：生成真值数据（Data Generation）
我们需要用传统的数值求解器（Godunov 格式）算出 PDE 真实的“解”，作为模型学习的标准答案：
```bash
python scripts/generate_data.py
```
*运行成功后，你会在 `env_config.yaml` 所设定的 `data_dir` 中看到生成的数据文件。*

### 第 4 步：启动训练（Training）
运行端到端训练脚本，该脚本会将 1D U-Net 骨干与物理粘性匹配的噪声调度结合在一起：
```bash
python scripts/train_mvp.py
```
*模型输出、日志及 checkpoint 会保存在 `env_config.yaml` 设定的 `output_dir` 中。*

## 2. 目录结构

```text
PROJECT/black/
├── configs/                    # 配置文件
│   ├── env_config.yaml         # 你的本地环境配置（勿提交到 Git）
│   ├── default.yaml            # 默认实验配置
│   ├── e1_burgers.yaml         # E1 Burgers 实验配置
│   ├── e2_buckley_leverett.yaml
│   └── e3_euler_sod.yaml
├── src/
│   ├── data/                   # 数据生成与加载（Burgers 1D Godunov 求解器）
│   ├── models/                 # 神经网络架构（1D U-Net 骨干）
│   ├── diffusion/              # 扩散模型核心（粘性匹配调度、Godunov 采样器）
│   ├── entrodiff/              # EntroDiff 论文模块（待 W5+ 实装）
│   └── utils/                  # 环境管理器
├── scripts/
│   ├── generate_data.py        # 生成 Godunov 真值数据
│   ├── train_mvp.py            # 端到端训练脚本
│   └── eval_viz.py             # （待完成）评估与可视化
├── tests/                      # 单元测试（pytest）
├── pyproject.toml
├── requirements.txt
├── README.md                   # 英文文档
└── README.zh.md                # 本文件（中文文档）
```

## 3. 实验配置与开发建议

MVP 阶段在 1D 无粘 Burgers 方程上验证 EntroDiff 理论。实验对比纯 DSM 基线 + TV 正则 + 物理粘性匹配调度。

- **MVP 快速迭代**：在 `scripts/train_mvp.py` 或者 `configs/e1_burgers.yaml`（未来）中手动修改超参数，如 `epochs`、`lr` 和 `batch_size`。Windows 本地侧重功能测试而非性能。
- **环境安全**：不同环境（PC/Kaggle/Colab/Server）切换时，**只需修改本地环境专属配置 `configs/env_config.yaml`**，不要动实验配置的共享 YAML 文件，更不要在脚本里硬编码绝对路径。

## 4. 理论与实现差异说明

各模块与论文 `03_method.tex` 公式的对应关系与当前简化策略：

| 模块 | 文件 | 对应论文 | 当前状态 |
|---|---|---|---|
| 粘性匹配调度 | `src/diffusion/schedules.py` | Eq. 3.1: `σ²(τ) = 2ν·τ` | ✅ 精确实现 |
| 损失函数 | `src/diffusion/losses.py` | Eq. 3.6 损失家族 | `L_DSM` + `L_BV`（TV 代理）已实现；`L_ent`（Kruzhkov 熵正则）待实装 |
| Godunov 通量 | `src/diffusion/losses.py:godunov_flux()` | Eq. 3.9 | ✅ 精确实现（shock 用 max、rarefaction 用 min+sonic 检测） |
| BV-aware 参数化 | `src/models/score_param.py:BVAwareScore` | Eq. 3.2 | 🔶 三子网络框架已搭建；`autograd` 梯度接口已注释预留，当前使用 mock 前向 |
| 采样器 | `src/diffusion/samplers.py` | Algorithm 1 / Eq. 3.8 | 🔶 Heun 二阶积分已实现；PDE guidance 使用 Godunov 残差代理（非 `∇L_PDE`），注释中已给出正确梯度形式的实现方式 |
| Baseline U-Net | `src/models/unet_1d.py` | — | ✅ 完整实现 |
| EDM 预处理 | `src/models/score_param.py:StandardScore` | EDM (Karras 2022) | ✅ `c_skip/c_out/c_in/c_noise` 精确实现 |

## 5. 开发原则

- **三层解耦**：`src/`（库）、`scripts/`（CLI 入口）、`configs/`（参数 YAML）
- **所有路径从 `EnvManager` 获取**，绝不硬编码绝对路径
- **数据和代码彻底分离**：代码在 Git，数据在 `data_dir` 下
- **测试**：`tests/` 下用 `pytest`；核心模块完成一个写一个测试
- **Git**：原子化 commit，`feat/fix/refactor` 类型前缀
- **License**：MIT，作者 `kyksj-1`
