import torch

class NoiseSchedule:
    """Base class for noise schedules."""
    def get_sigma(self, tau: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class BaselineSchedule(NoiseSchedule):
    """
    Standard EDM Schedule (typically uniform log-normal for training, 
    but continuous power law for tau in PDE diffusion).
    For comparative baseline: sigma(tau) = tau 
    """
    def __init__(self, p_mean=-1.2, p_std=1.2):
        self.p_mean = p_mean
        self.p_std = p_std

    def get_sigma(self, tau: torch.Tensor) -> torch.Tensor:
        # Standard EDM continuous time: simple power or linear
        # Often training doesn't use tau directly, but samples ln(sigma) ~ N(P_mean, P_std).
        # We handle this in loss sampling.
        raise NotImplementedError("Use sample_sigma for training.")

    def sample_sigma(self, batch_size: int, device: str) -> torch.Tensor:
        # EDM's default log-normal sampling
        rnd_normal = torch.randn([batch_size], device=device)
        return (rnd_normal * self.p_std + self.p_mean).exp()

class ViscosityMatchedSchedule(NoiseSchedule):
    """
    EntroDiff Schedule (Section 3.1): sigma^2(tau) = 2 * nu * tau.
    sigma(tau) = sqrt(2 * nu * tau).
    """
    def __init__(self, nu: float = 0.01, tau_max: float = 1.0):
        self.nu = nu
        self.tau_max = tau_max

    def get_sigma(self, tau: torch.Tensor) -> torch.Tensor:
        """Returns physical matched sigma for given continuous time tau."""
        return torch.sqrt(2.0 * self.nu * tau)

    def sample_sigma(self, batch_size: int, device: str) -> torch.Tensor:
        """Uniform sampling in physical time tau, providing mapped sigma."""
        # For viscosity-matched, we might just sample tau ~ U[0, T_d]
        tau = torch.rand([batch_size], device=device) * self.tau_max
        return self.get_sigma(tau)
