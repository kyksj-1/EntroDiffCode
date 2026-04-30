# ============================================================================
# TDD: Euler Sod flux + HLLC Riemann solver 单元测试
# 确保 flux 函数数学正确后再用 solver 生成数据
# ============================================================================
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pdes.euler_sod import (
    euler_prim_to_cons,
    euler_cons_to_prim,
    euler_flux,
    euler_hllc_flux,
    euler_max_wavespeed,
    sod_initial_condition,
    euler_godunov_step,
    solve_euler_1d,
)

GAMMA = 1.4

# === 基本通量测试 ===

def test_prim_cons_roundtrip():
    """原始 ↔ 守恒变量互逆."""
    rho, u, p = 1.2, 0.3, 1.0
    U = euler_prim_to_cons(np.array(rho), np.array(u), np.array(p), GAMMA)
    rho2, u2, p2 = euler_cons_to_prim(U, GAMMA)
    assert abs(float(rho2) - rho) < 1e-10, f"rho roundtrip: {rho2} != {rho}"
    assert abs(float(u2) - u) < 1e-10, f"u roundtrip: {u2} != {u}"
    assert abs(float(p2) - p) < 1e-10, f"p roundtrip: {p2} != {p}"


def test_flux_uniform_state():
    """均匀态: F(U) 应与解析一致."""
    rho, u, p = 1.0, 2.0, 1.5
    U = euler_prim_to_cons(np.array(rho), np.array(u), np.array(p), GAMMA)
    F = euler_flux(U, GAMMA)
    # F[0] = ρu = 2.0
    assert abs(float(F[0]) - 2.0) < 1e-8, f"F[0]={F[0]}"
    # F[1] = ρu² + p = 1*4 + 1.5 = 5.5
    assert abs(float(F[1]) - 5.5) < 1e-8, f"F[1]={F[1]}"
    # F[2] = u(E+p), E = p/(γ-1) + ½ρu² = 1.5/0.4 + 0.5*1*4 = 3.75+2=5.75
    # F[2] = 2*(5.75+1.5) = 14.5
    assert abs(float(F[2]) - 14.5) < 1e-6, f"F[2]={F[2]}"


def test_hllc_identical_states():
    """相同左右状态 → HLLC 通量 = 物理通量 (一致性)."""
    U = euler_prim_to_cons(np.ones(3) * 1.0, np.ones(3) * 0.0, np.ones(3) * 1.0, GAMMA)
    F_hllc = euler_hllc_flux(U, U, GAMMA)
    F_phys = euler_flux(U, GAMMA)
    assert np.allclose(F_hllc, F_phys, atol=1e-10), "HLLC ≠ physical flux for identical states"


def test_hllc_sod_problem():
    """经典 Sod 激波管: HLLC 应给出合理的中间通量."""
    # 左: (ρ=1, u=0, p=1), 右: (ρ=0.125, u=0, p=0.1)
    UL = euler_prim_to_cons(np.array(1.0), np.array(0.0), np.array(1.0), GAMMA)[:, np.newaxis]
    UR = euler_prim_to_cons(np.array(0.125), np.array(0.0), np.array(0.1), GAMMA)[:, np.newaxis]

    F = euler_hllc_flux(UL, UR, GAMMA)

    # 检查 F 各分量均为有限值
    assert np.all(np.isfinite(F)), f"NaN/inf in HLLC flux: {F}"
    # Sod 问题初始膜处, 两侧速度均为 0, 但 p_L >> p_R,
    # 整体波系向右 → F[2] (能量通量) 应 > 0
    assert float(F[2, 0]) > 0.0, f"Energy flux should be positive for Sod: {F[2]}"


def test_hllc_vectorized():
    """向量化 HLLC: 多个界面同时计算应 OK."""
    N = 10
    UL = euler_prim_to_cons(
        np.random.uniform(0.5, 2.0, N),
        np.random.uniform(-0.5, 0.5, N),
        np.random.uniform(0.5, 2.0, N),
        GAMMA,
    )
    UR = euler_prim_to_cons(
        np.random.uniform(0.5, 2.0, N),
        np.random.uniform(-0.5, 0.5, N),
        np.random.uniform(0.5, 2.0, N),
        GAMMA,
    )
    F = euler_hllc_flux(UL, UR, GAMMA)
    assert F.shape == (3, N), f"Shape mismatch: {F.shape}"
    assert np.all(np.isfinite(F)), "NaN/inf in vectorized HLLC"


def test_max_wavespeed():
    """max|u|+c 应 ≥ 0 且为有限值."""
    nx = 16
    U0 = sod_initial_condition(nx=nx, gamma=GAMMA)
    w = euler_max_wavespeed(U0, GAMMA)
    assert w > 0, f"Wavespeed should be positive: {w}"
    assert np.isfinite(w), f"Wavespeed should be finite: {w}"


def test_sod_ic_values():
    """Sod IC: 左半 ρ=1, p=1; 右半 ρ=0.125, p=0.1."""
    nx = 128
    U0 = sod_initial_condition(nx=nx, gamma=GAMMA)
    rho, u, p = euler_cons_to_prim(U0, GAMMA)
    mid = nx // 2
    # 左侧
    assert abs(float(rho[0]) - 1.0) < 1e-6
    assert abs(float(p[0]) - 1.0) < 1e-6
    # 右侧
    assert abs(float(rho[-1]) - 0.125) < 1e-6
    assert abs(float(p[-1]) - 0.1) < 1e-6


def test_godunov_step_stability():
    """单步 Godunov 推进: 不崩溃, 密度/压力维持正值."""
    nx = 64
    U0 = sod_initial_condition(nx=nx, gamma=GAMMA)
    dx = 1.0 / nx
    w_max = euler_max_wavespeed(U0, GAMMA)
    dt = 0.4 * dx / w_max

    U1 = euler_godunov_step(U0, dt, dx, GAMMA)
    rho1, _, p1 = euler_cons_to_prim(U1, GAMMA)
    assert np.all(rho1 > 0), f"Negative density after 1 step"
    assert np.all(p1 > 0), f"Negative pressure after 1 step"


def test_solver_small_run():
    """小规模全时间积分: 不崩溃, 守恒量大体保持."""
    nx = 64
    nt = 50
    U0 = sod_initial_condition(nx=nx, gamma=GAMMA)
    dx = 1.0 / nx
    w_max = euler_max_wavespeed(U0, GAMMA)
    dt = 0.3 * dx / w_max

    U_hist = solve_euler_1d(U0, nx, nt, dt, dx, GAMMA)
    assert U_hist.shape == (3, nt + 1, nx), f"Shape: {U_hist.shape}"
    # 总质量应大致守恒 (周期 BC 下精确; 透射 BC 下近似)
    mass0 = U_hist[0, 0, :].sum()
    mass_final = U_hist[0, -1, :].sum()
    rel_change = abs(mass_final - mass0) / mass0
    assert rel_change < 0.10, f"Mass conservation violated: {rel_change:.4f}"


if __name__ == "__main__":
    tests = [
        test_prim_cons_roundtrip,
        test_flux_uniform_state,
        test_hllc_identical_states,
        test_hllc_sod_problem,
        test_hllc_vectorized,
        test_max_wavespeed,
        test_sod_ic_values,
        test_godunov_step_stability,
        test_solver_small_run,
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
