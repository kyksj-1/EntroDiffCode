# ============================================================================
# Robust Shock Location Metrics (W5 后 SA3)
#
# 背景: 当前 eval 用 argmax(|∇u|) 定位 shock, 对噪声敏感.
#   - 传统方法 (Lax-F / MacCormack / Central): shock_err ∈ [0.07, 0.17]
#   - Foundation DiT-BVAware:                  shock_err ≈ 2.05  (~30× 差距)
#   差距太大暗示 argmax 跳到错位置. 提供 3 种鲁棒度量供选择.
#
# 设计原则:
#   - 默认 method='argmax' 保留旧行为, 不破坏现有 eval
#   - 新加 'topk_mean' / 'threshold_centroid' 两种鲁棒度量
#   - 统一接口 compute_shock_err_robust 通过 method 字符串派遣
#   - 单元测试覆盖噪声鲁棒性
# ============================================================================

from __future__ import annotations

from typing import Literal

import numpy as np

ShockMethod = Literal["argmax", "topk_mean", "threshold_centroid"]


def shock_location_argmax(u: np.ndarray, x_grid: np.ndarray) -> float:
    """
    Argmax-based shock 位置: 最大 |∇u| 处的 x 坐标.

    Args:
        u:      (Nx,) 1D 解
        x_grid: (Nx,) 空间网格坐标
    Returns:
        shock 位置 (float, 单位与 x_grid 一致)

    脆弱性: 如果 u 在远离 shock 处有低幅噪声残留, argmax 可能跳到噪声位置.
    """
    grad_u = np.abs(np.gradient(u, x_grid))
    return float(x_grid[np.argmax(grad_u)])


def shock_location_topk_mean(
    u: np.ndarray,
    x_grid: np.ndarray,
    k: int = 3,
) -> float:
    """
    Top-k 个 |∇u| 最大点的位置平均, 抗噪声.

    机理:
      - 单 shock 信号: top-k 都在 shock 邻域 → 平均 ≈ shock 位置
      - 噪声信号: top-1 可能在噪声, 但 top-k 中真 shock 占多数 → 平均偏向真 shock

    Args:
        u:      (Nx,) 解
        x_grid: (Nx,) 网格
        k:      取 top-k 个最大梯度位置 (默认 3)

    Returns:
        位置平均
    """
    if k < 1:
        raise ValueError(f"k 必须 ≥ 1, 实际 {k}")
    grad_u = np.abs(np.gradient(u, x_grid))
    # argpartition 求 top-k 索引 (无序), 比 argsort 快
    top_k_idx = np.argpartition(grad_u, -k)[-k:]
    return float(np.mean(x_grid[top_k_idx]))


def shock_location_threshold_centroid(
    u: np.ndarray,
    x_grid: np.ndarray,
    threshold_ratio: float = 0.5,
) -> float:
    """
    阈值过滤后的加权重心: 取 |∇u| > t·max(|∇u|) 的点的梯度加权平均位置.

    机理:
      - shock 区域形成连通的高梯度块 → 重心稳定在 shock 中心
      - 噪声幅度小于阈值会被过滤掉
      - 比 topk_mean 更平滑, 抗 outlier 也更好

    Args:
        u:               (Nx,) 解
        x_grid:          (Nx,) 网格
        threshold_ratio: 阈值占最大梯度的比例 (默认 0.5)

    Returns:
        加权重心位置
    """
    if not (0.0 < threshold_ratio <= 1.0):
        raise ValueError(f"threshold_ratio 应在 (0, 1], 实际 {threshold_ratio}")
    grad_u = np.abs(np.gradient(u, x_grid))
    threshold = threshold_ratio * grad_u.max()
    mask = grad_u > threshold
    if not mask.any():
        # 退化: 没有点超阈值 (近似常数解), 退回 argmax
        return float(x_grid[np.argmax(grad_u)])
    weights = grad_u[mask]
    positions = x_grid[mask]
    return float(np.sum(positions * weights) / np.sum(weights))


def compute_shock_err_robust(
    u_pred: np.ndarray,
    u_gt: np.ndarray,
    x_grid: np.ndarray,
    method: ShockMethod = "argmax",
    **kwargs,
) -> float:
    """
    统一 shock-err 计算接口: |x_shock_pred - x_shock_gt| 在指定 metric 下.

    Args:
        u_pred:  (Nx,) 预测解
        u_gt:    (Nx,) 真值
        x_grid:  (Nx,) 网格
        method:  'argmax' | 'topk_mean' | 'threshold_centroid'
        **kwargs: 透传给具体 method (e.g. k=3 给 topk_mean, threshold_ratio=0.5)
    Returns:
        |Δx_shock| (与 x_grid 同单位)

    Raises:
        ValueError: method 未知
    """
    dispatch = {
        "argmax": shock_location_argmax,
        "topk_mean": shock_location_topk_mean,
        "threshold_centroid": shock_location_threshold_centroid,
    }
    if method not in dispatch:
        raise ValueError(
            f"未知 shock method '{method}'. 可用: {list(dispatch.keys())}"
        )
    fn = dispatch[method]
    # 滤掉 dispatch 不接受的 kwargs (e.g. k 不应给 argmax)
    valid_kwargs = _filter_kwargs(fn, kwargs)
    x_pred = fn(u_pred, x_grid, **valid_kwargs)
    x_gt = fn(u_gt, x_grid, **valid_kwargs)
    return float(abs(x_pred - x_gt))


def _filter_kwargs(fn, kwargs: dict) -> dict:
    """仅保留 fn 签名接受的 kwargs."""
    import inspect
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def compute_all_shock_errs(
    u_pred: np.ndarray,
    u_gt: np.ndarray,
    x_grid: np.ndarray,
) -> dict:
    """
    一次返回 3 种 shock_err. 用于 eval 输出表格的 multi-metric 列.

    Returns:
        {
            "shock_argmax":   float,
            "shock_topk3":    float,
            "shock_threshold": float,
        }
    """
    return {
        "shock_argmax": compute_shock_err_robust(u_pred, u_gt, x_grid, method="argmax"),
        "shock_topk3": compute_shock_err_robust(
            u_pred, u_gt, x_grid, method="topk_mean", k=3
        ),
        "shock_threshold": compute_shock_err_robust(
            u_pred, u_gt, x_grid, method="threshold_centroid", threshold_ratio=0.5
        ),
    }
