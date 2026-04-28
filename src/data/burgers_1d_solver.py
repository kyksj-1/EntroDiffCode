import numpy as np

def burgers_godunov_1d(u0, nx, nt, dx, dt):
    """
    Solves the 1D inviscid Burgers' equation u_t + (0.5 * u^2)_x = 0
    using the Godunov finite volume scheme.

    Args:
        u0 (np.ndarray): Initial condition array of size (nx,)
        nx (int): Number of spatial grid points
        nt (int): Number of time steps
        dx (float): Spatial step size
        dt (float): Time step size

    Returns:
        u_hist (np.ndarray): Solution history of size (nt+1, nx)
    """
    
    u = u0.copy()
    u_hist = np.zeros((nt + 1, nx))
    u_hist[0, :] = u

    for t in range(nt):
        f = np.zeros(nx + 1)
        
        # Calculate Godunov numerical flux at cell interfaces
        for i in range(1, nx):
            ul = u[i - 1]
            ur = u[i]
            
            # The flux function is F(u) = 0.5 * u^2
            # Godunov flux for f(u) = 1/2 u^2 (convex flux):
            if ul >= ur:
                f[i] = max(0.5 * ul**2, 0.5 * ur**2)
            else:
                if ul <= 0.0 and ur >= 0.0:
                    f[i] = 0.0
                else:
                    f[i] = min(0.5 * ul**2, 0.5 * ur**2)
                    
        # Periodic boundaries (flux at i = 0 and i = nx)
        ul = u[-1]
        ur = u[0]
        if ul >= ur:
            f[0] = f[nx] = max(0.5 * ul**2, 0.5 * ur**2)
        else:
            if ul <= 0.0 and ur >= 0.0:
                f[0] = f[nx] = 0.0
            else:
                f[0] = f[nx] = min(0.5 * ul**2, 0.5 * ur**2)

        # Update solution using the conservative form
        u = u - (dt / dx) * (f[1:] - f[:-1])
        u_hist[t + 1, :] = u

    return u_hist
