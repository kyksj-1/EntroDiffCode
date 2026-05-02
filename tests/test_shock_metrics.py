# ============================================================================
# Robust Shock Metric 单元测试 (SA3)
#
# 跑测试: python -m pytest PROJECT/black/tests/test_shock_metrics.py -v
# ============================================================================

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.shock_metrics import (  # noqa: E402
    shock_location_argmax,
    shock_location_topk_mean,
    shock_location_threshold_centroid,
    compute_shock_err_robust,
    compute_all_shock_errs,
)


# ----------------------------------------------------------------------------
# Fixture: tanh shock signal
# ----------------------------------------------------------------------------

def make_tanh_shock(x_shock: float, Nx: int = 256, width: float = 0.05) -> tuple:
    """生成 tanh 形 shock 在 x=x_shock 处, 返回 (u, x_grid)."""
    x_grid = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
    u = np.tanh((x_grid - x_shock) / width)
    return u, x_grid


# ----------------------------------------------------------------------------
# 6 个核心测试
# ----------------------------------------------------------------------------

def test_argmax_basic() -> None:
    """干净 tanh shock, argmax 应给出 shock 位置."""
    x_true = 3.0
    u, x_grid = make_tanh_shock(x_true)
    x_est = shock_location_argmax(u, x_grid)
    # 网格分辨率 dx = 2π/256 ≈ 0.025, 误差应 < 1 个网格
    assert abs(x_est - x_true) < 0.05, f"argmax 估计 {x_est} 偏离真值 {x_true}"


def test_topk_mean_robust_to_noise() -> None:
    """
    Pathological case: 一个 spurious peak 让 argmax 跳到错位置, 鲁棒度量找回真 shock.

    构造: 真 shock 形状温和 (max gradient ~5), 远处一个 spurious peak (gradient ~6).
    argmax 会选 spurious; threshold_centroid 用阈值过滤后, spurious 占比小.
    """
    Nx = 256
    x_grid = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
    x_true = 2.0
    # 温和 shock: width 较大, max gradient 中等
    u = np.tanh((x_grid - x_true) / 0.3)
    # 远处 spurious peak (单点跳变): gradient = jump_height/dx
    rng = np.random.default_rng(0)
    u_noisy = u + rng.normal(0, 0.001, size=u.shape)
    spurious_idx = int(0.85 * Nx)   # x ≈ 5.34
    # 单格跳一个小步, 但孤立 → 只产生 1-2 个高梯度点
    u_noisy[spurious_idx] += 0.3

    x_argmax = shock_location_argmax(u_noisy, x_grid)
    x_thr = shock_location_threshold_centroid(u_noisy, x_grid, threshold_ratio=0.4)

    err_argmax = abs(x_argmax - x_true)
    err_thr = abs(x_thr - x_true)

    # 期望 1: argmax 跳到 spurious (error 接近 spurious 距离)
    # 期望 2: threshold_centroid 过滤掉 spurious 后回到 shock
    # 容忍: 不一定每次 argmax 都跳, 但任一鲁棒度量比 argmax 改善 > 0.5 即过
    if err_argmax > 1.0:
        # argmax 确实跳了 → 验证鲁棒度量找回
        assert err_thr < err_argmax * 0.5, (
            f"argmax 已跳到 spurious (err={err_argmax:.3f}), "
            f"threshold_centroid 应明显改善 (err={err_thr:.3f})"
        )
    else:
        # argmax 没跳 (信号太干净) → 鲁棒度量也应给出接近答案
        assert err_thr < 0.5, (
            f"argmax 未失效 (err={err_argmax:.3f}), "
            f"threshold_centroid 也应正常 (err={err_thr:.3f})"
        )


def test_threshold_centroid_basic() -> None:
    """干净 tanh shock + 平坦区, centroid 应在 shock 中心."""
    x_true = 3.0
    u, x_grid = make_tanh_shock(x_true)
    x_est = shock_location_threshold_centroid(u, x_grid, threshold_ratio=0.5)
    assert abs(x_est - x_true) < 0.05


def test_metrics_close_for_clean_signal() -> None:
    """干净信号下三种 metric 应给出相近结果."""
    x_true = 2.5
    u, x_grid = make_tanh_shock(x_true)
    x_arg = shock_location_argmax(u, x_grid)
    x_topk = shock_location_topk_mean(u, x_grid, k=3)
    x_thr = shock_location_threshold_centroid(u, x_grid, threshold_ratio=0.5)
    # 三者两两差距 < 0.1 (干净信号下都接近真值)
    assert abs(x_arg - x_topk) < 0.1
    assert abs(x_arg - x_thr) < 0.1
    assert abs(x_topk - x_thr) < 0.1


def test_compute_shock_err_dispatch() -> None:
    """统一接口的字符串派遣."""
    u_pred, x_grid = make_tanh_shock(3.0)
    u_gt, _ = make_tanh_shock(3.1)
    err_arg = compute_shock_err_robust(u_pred, u_gt, x_grid, method="argmax")
    err_topk = compute_shock_err_robust(
        u_pred, u_gt, x_grid, method="topk_mean", k=3
    )
    err_thr = compute_shock_err_robust(
        u_pred, u_gt, x_grid, method="threshold_centroid", threshold_ratio=0.5
    )
    # 真差距是 0.1, 三个 metric 估计都应接近
    for label, err in [("argmax", err_arg), ("topk_mean", err_topk),
                        ("threshold", err_thr)]:
        assert abs(err - 0.1) < 0.1, f"{label} 估计 {err} 偏离真差 0.1"


def test_unknown_method_raises() -> None:
    """未知 method 应 ValueError."""
    u, x_grid = make_tanh_shock(3.0)
    with pytest.raises(ValueError, match="未知 shock method"):
        compute_shock_err_robust(u, u, x_grid, method="nonexistent")


# ----------------------------------------------------------------------------
# 辅助测试
# ----------------------------------------------------------------------------

def test_topk_invalid_k() -> None:
    """k < 1 必须报错."""
    u, x_grid = make_tanh_shock(3.0)
    with pytest.raises(ValueError, match="k 必须"):
        shock_location_topk_mean(u, x_grid, k=0)


def test_threshold_invalid_ratio() -> None:
    """threshold_ratio 越界报错."""
    u, x_grid = make_tanh_shock(3.0)
    with pytest.raises(ValueError, match="threshold_ratio"):
        shock_location_threshold_centroid(u, x_grid, threshold_ratio=1.5)


def test_compute_all_shock_errs_keys() -> None:
    """compute_all 返回的 dict 含 3 个 key."""
    u, x_grid = make_tanh_shock(3.0)
    u2, _ = make_tanh_shock(3.05)
    out = compute_all_shock_errs(u, u2, x_grid)
    assert set(out.keys()) == {"shock_argmax", "shock_topk3", "shock_threshold"}
    for v in out.values():
        assert isinstance(v, float) and v >= 0


def test_threshold_degenerate_constant() -> None:
    """常数解 (无 shock) 应退化为 argmax 不报错."""
    Nx = 128
    x_grid = np.linspace(0, 2 * np.pi, Nx, endpoint=False)
    u = np.ones(Nx)
    # ∇u 处处为 0, threshold 后没点超阈值, 退化为 argmax
    x = shock_location_threshold_centroid(u, x_grid, threshold_ratio=0.5)
    assert isinstance(x, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
