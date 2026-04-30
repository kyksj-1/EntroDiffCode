# Buckley–Leverett Godunov solver (pdes package)
from .bl_flux import bl_flux, bl_godunov_flux

# 1D Euler Sod + HLLC Riemann solver (pdes package)
from .euler_sod import (
    euler_prim_to_cons,
    euler_cons_to_prim,
    euler_flux,
    euler_hllc_flux,
    euler_max_wavespeed,
    euler_godunov_step,
    solve_euler_1d,
    sod_initial_condition,
)

# 2D Shallow-Water + HLL dimensional splitting solver (pdes package)
from .shallow_water import (
    sw_cons_to_prim,
    sw_flux_x,
    sw_flux_y,
    sw_hll_flux_1d,
    sw_x_sweep,
    sw_y_sweep,
    solve_shallow_water_2d,
    dam_break_ic,
    sw_max_wavespeed,
)
