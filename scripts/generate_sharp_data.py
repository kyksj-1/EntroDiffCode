import os
import sys
import numpy as np
from pathlib import Path

# 将 PROJECT/black/ 加入 sys.path, 以确保 src/ 可被发现
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.env_manager import env
from src.data.burgers_1d_solver import burgers_godunov_1d
from tqdm import tqdm


def generate_sharp_burgers_data(
    n_samples=5000,
    nx=128,
    nt=100,
    dt=0.005,
    dx=2 * np.pi / 128,
    k_max=10,
):
    """
    生成具有更陡峭激波 (sharper shock) 的 1D Burgers 方程数据集。

    与标准 generate_data.py 的关键区别:
        1. k_max=10 (标准: 5)      → 更多高频傅里叶模态
        2. 系数缩放 1/k (标准: 1/k²) → 高频模态振幅显著增大, 不抑制高频
        3. 结果是初始条件梯度更大, 激波形成更早、更尖锐

    物理直觉 —— 为何 sharper ICs 对 baseline 更难:
        (a) Burgers 非线性项 u·u_x 将梯度指数放大: 高初始 ∇u → 更快进入激波
        (b) Godunov 通量在 uL ≥ uR 区间频繁触发 → 解的 BV 半范数 |u|_BV 增大
        (c) 标准 EDM baseline: score 网络学的是从 p(data) 到 p(noise) 的扩散路径,
            但 sharper shock 的高阶 BV 变化使得该路径中 score 的 Lipschitz 常数更大,
            网络需要更多去噪步数来补偿, 少步时 PDE residual 爆炸
        (d) 中央差分 (baseline PDE guidance 的 proxy) 在激波处截断误差 ∝ Δx·|u'''|,
            尖锐激波 → |u'''| 极大 → 残差巨大, 推往非物理解
        (e) 相比之下 EntroDiff 的 BVAwareScore 硬编码了 tanh(φ_sh/2σ²) 界面剖面,
            对尖锐梯度具有内在鲁棒性 —— 即便在少步采样下也能恢复激波结构
    """

    print(f"Generating {n_samples} samples of 1D Burgers with sharper shocks...")
    print(f"Grid: nx={nx}, nt={nt}, dx={dx:.4f}, dt={dt:.4f}, k_max={k_max}")
    print(f"IC scaling: 1/k (standard: 1/k^2) → higher amplitude HF modes")

    # 存储所有样本: (n_samples, nt+1, nx)
    data_hist = np.zeros((n_samples, nt + 1, nx), dtype=np.float32)

    # 空间网格: [0, 2π) 周期性
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)

    for i in tqdm(range(n_samples)):
        # ---- 步骤 1: 构造更尖锐的随机初始条件 ----
        # k_max=10 傅里叶模态 (标准为 5)
        A = np.random.randn(k_max)  # 正弦系数
        B = np.random.randn(k_max)  # 余弦系数

        # ★ 关键区别: 系数缩放用 1/k 而非 1/k²
        # - 1/k² 会强烈压制高频 → 初始条件平滑, 激波缓慢发展
        # - 1/k 仅弱衰减高频 → 高频分量振幅显著增大, 梯度更陡
        # 数值示例 (k=10 模态):
        #   scale=1/k² → 振幅 ~0.01·N(0,1)
        #   scale=1/k  → 振幅 ~0.1·N(0,1) → 10× 更大的高频贡献
        A /= (np.arange(1, k_max + 1))  # 1/k 缩放
        B /= (np.arange(1, k_max + 1))  # 1/k 缩放

        # 叠加所有模态构建 u(x,0)
        u0 = np.zeros(nx)
        for k in range(1, k_max + 1):
            u0 += A[k - 1] * np.sin(k * x) + B[k - 1] * np.cos(k * x)

        # ---- 步骤 2: CFL 条件检查与自适应缩放 ----
        # Godunov 显式格式的 CFL 条件: max|u| * dt/dx < 1
        # 由于 1/k 缩放使振幅增大, CFL 违反概率上升, 但 Godunov 对弱 CFL
        # 违反仍有一定鲁棒性; 为安全起见当 CFL > 0.9 时进行全局缩放
        u_max = np.max(np.abs(u0))
        cfl = u_max * dt / dx

        if cfl > 0.9:
            # 等比例缩放使得新 CFL = 0.9, 保持激波结构的相对形状不变
            u0 = u0 * (0.9 / cfl)

        # ---- 步骤 3: Godunov 有限体积法求解 Burgers 方程 ----
        # 在尖锐 IC 下 Godunov 通量频繁触发 max/min 分支, 精确捕捉激波速度
        u_hist = burgers_godunov_1d(u0, nx=nx, nt=nt, dx=dx, dt=dt)
        data_hist[i] = u_hist

    return data_hist


if __name__ == "__main__":
    # ---- 网格参数 (与标准 generate_data.py 对齐, 便于对照) ----
    NX = 128  # 空间网格点数
    NT = 100  # 时间步数 → 总时间 T = NT * DT = 0.5
    DT = 0.005  # 时间步长
    DX = 2 * np.pi / NX  # 空间步长 (周期性域 [0, 2π))
    K_MAX = 10  # ★ 傅里叶模态数: 10 (标准: 5), 引入更多高频
    N_SAMPLES = 5000  # 样本数 (与标准数据集对齐)

    # ---- 输出路径: 使用 env.data_dir, 文件名与标准数据区分 ----
    output_path = env.data_dir / "burgers_sharp_N5000_Nx128.npy"

    # ---- 检查环境配置中的数据目录是否存在 ----
    env.data_dir  # 触发 mkdir(parents=True, exist_ok=True) 在 EnvManager.data_dir 属性中

    # ---- 生成数据 ----
    data = generate_sharp_burgers_data(
        n_samples=N_SAMPLES,
        nx=NX,
        nt=NT,
        dt=DT,
        dx=DX,
        k_max=K_MAX,
    )

    # ---- 保存 ----
    print(f"Saving sharp Burgers data to {output_path} (shape: {data.shape})")
    np.save(output_path, data)
    print("Done.")
