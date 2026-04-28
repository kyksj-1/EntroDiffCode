import os
from pathlib import Path
import yaml
import warnings

# Define the root path (one level up from src/utils/env_manager.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# The expected location of the environment config file
ENV_CONFIG_PATH = PROJECT_ROOT / "configs" / "env_config.yaml"
ENV_TEMPLATE_PATH = PROJECT_ROOT / "configs" / "env_config.yaml.example"

class EnvManager:
    """
    Singleton-like access to environment configuration.
    Parses configs/env_config.yaml and makes the paths/settings accessible.
    If the file is not found, issues a warning and defaults to local project paths.
    """
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EnvManager, cls).__new__(cls)
            cls._load_config()
        return cls._instance

    @classmethod
    def _load_config(cls):
        """Loads and validates the environment configuration."""
        if not ENV_CONFIG_PATH.exists():
            warnings.warn(
                f"\n[EnvManager] Expected configuration file not found: {ENV_CONFIG_PATH}"
                f"\nFalling back to default paths relative to PROJECT_ROOT: {PROJECT_ROOT}."
                f"\nTip: Copy {ENV_TEMPLATE_PATH} to {ENV_CONFIG_PATH} and adjust the paths."
            )
            cls._config = {
                "paths": {
                    "data_dir": str(PROJECT_ROOT / "data"),
                    "output_dir": str(PROJECT_ROOT / "experiments"),
                },
                "hardware": {
                    "device": "cuda",
                    "max_batch_size": 64,
                    "num_workers": 0
                }
            }
            return

        with open(ENV_CONFIG_PATH, "r", encoding="utf-8") as f:
            cls._config = yaml.safe_load(f) or {}

        # Ensure essential keys exist
        if "paths" not in cls._config:
            cls._config["paths"] = {}
        if "hardware" not in cls._config:
            cls._config["hardware"] = {}
            
    # ------ Data & Path Helpers
    
    @property
    def data_dir(self) -> Path:
        """Returns the absolute Path to the data directory."""
        path_str = self._config["paths"].get("data_dir", str(PROJECT_ROOT / "data"))
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def output_dir(self) -> Path:
        """Returns the absolute Path to the experiment outputs directory."""
        path_str = self._config["paths"].get("output_dir", str(PROJECT_ROOT / "experiments"))
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------ Hardware Helpers

    @property
    def num_workers(self) -> int:
        """Safe number of data loading workers (useful on Windows)."""
        # Default to 0 for Windows safety unless explicitly configured
        return self._config["hardware"].get("num_workers", 0 if os.name == 'nt' else 4)

    @property
    def default_device(self) -> str:
        """The default device, e.g., 'cuda' or 'cpu'."""
        import torch
        dev = self._config["hardware"].get("device", "cuda")
        if dev == "cuda" and not torch.cuda.is_available():
            warnings.warn("[EnvManager] 'cuda' requested but not available. Falling back to 'cpu'.")
            return "cpu"
        return dev

# Global singleton instance for easy import across modules
env = EnvManager()
