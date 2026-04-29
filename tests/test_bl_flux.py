# ============================================================================
# TDD Step 1: Buckley–Leverett flux 单元测试
# 先写测试，确保 flux 函数数学正确后再实现 solver
# ============================================================================
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 测试目标: src/pdes/bl_flux.py 中的 flux 和 Godunov 通量
from src.pdes.bl_flux import bl_flux, bl_godunov_flux


def test_flux_boundary():
    """边界值: f(0)=0, f(1)=1"""
    assert abs(bl_flux(0.0)) < 1e-10, f"f(0) should be 0, got {bl_flux(0.0)}"
    assert abs(bl_flux(1.0) - 1.0) < 1e-10, f"f(1) should be 1, got {bl_flux(1.0)}"


def test_flux_midpoint():
    """中点: f(0.5)=0.5"""
    assert abs(bl_flux(0.5) - 0.5) < 1e-10, f"f(0.5) should be 0.5, got {bl_flux(0.5)}"


def test_flux_monotonic():
    """单调性: u 递增 → f(u) 递增"""
    u = np.linspace(0.0, 1.0, 100)
    f = np.array([bl_flux(ui) for ui in u])
    assert np.all(np.diff(f) >= -1e-12), "f(u) should be monotonically non-decreasing"


def test_flux_range():
    """值域: f(u) ∈ [0,1] for u ∈ [0,1]"""
    for u in np.linspace(0, 1, 50):
        f = bl_flux(u)
        assert 0.0 <= f <= 1.0, f"f({u:.2f}) = {f:.4f}, out of [0,1]"


def test_godunov_flux_shock():
    """shock 情况: uL > uR, 选取较小通量 (迎风)"""
    # 对单调通量, uL > uR 意味着 shock, Godunov = max_{u∈[uR,uL]} f(u) = f(uL)
    # 因为 f 单调增
    ul, ur = 0.8, 0.2
    F = bl_godunov_flux(ul, ur)
    # 期望: F 在 [f(ur), f(ul)] ≈ [0.059, 0.5] 之间 (大致)
    assert bl_flux(ur) <= F <= bl_flux(ul), \
        f"Godunov flux {F:.4f} should be between f(ur)={bl_flux(ur):.4f} and f(ul)={bl_flux(ul):.4f}"


def test_godunov_flux_rarefaction():
    """rarefaction 情况: uL < uR, 选取较大通量"""
    ul, ur = 0.2, 0.8
    F = bl_godunov_flux(ul, ur)
    assert bl_flux(ul) <= F <= bl_flux(ur), \
        f"Godunov flux {F:.4f} should be between f(ul)={bl_flux(ul):.4f} and f(ur)={bl_flux(ur):.4f}"


def test_godunov_flux_symmetry():
    """反对称性: Godunov flux (ul,ur) 应在合理范围内"""
    ul, ur = 0.3, 0.7
    F1 = bl_godunov_flux(ul, ur)
    F2 = bl_godunov_flux(ur, ul)
    # 不同方向的通量应不同 (迎风特性)
    assert F1 != F2, "Godunov flux should be upwind (direction-dependent)"


def test_godunov_flux_consistency():
    """一致性: uL = uR 时 Godunov flux = f(u)"""
    for u in [0.0, 0.3, 0.5, 0.7, 1.0]:
        F = bl_godunov_flux(u, u)
        assert abs(F - bl_flux(u)) < 1e-10, \
            f"Consistency failed at u={u}: F={F:.4f}, f(u)={bl_flux(u):.4f}"


if __name__ == "__main__":
    # 运行所有测试
    tests = [
        test_flux_boundary,
        test_flux_midpoint,
        test_flux_monotonic,
        test_flux_range,
        test_godunov_flux_shock,
        test_godunov_flux_rarefaction,
        test_godunov_flux_symmetry,
        test_godunov_flux_consistency,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
