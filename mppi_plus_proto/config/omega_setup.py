import numpy as np

from pathlib import Path
from typing import Any
from omegaconf import OmegaConf, DictConfig


def _include_section(filepath: str, key_path: str | None = None):
    """Loads a YAML file and optionally extracts a specific section."""
    cfg = OmegaConf.load(filepath)
    if key_path:
        # Extracts just the requested section, returns None if not found
        return OmegaConf.select(cfg, key_path)
    return cfg


def omegaconf_setup():
    OmegaConf.register_new_resolver("deg2rad", lambda x: float(np.deg2rad(x)))
    OmegaConf.register_new_resolver("include", _include_section)


def load_combined(config_path: str | Path, 
                  cli: bool = True, 
                  kwargs: dict[str, Any] | None = None) -> DictConfig:
    config = OmegaConf.load(config_path)
    if cli:
        cli_config = OmegaConf.from_cli()
        config = OmegaConf.merge(config, cli_config)
    if kwargs is not None:
        config = OmegaConf.merge(config, kwargs)
    OmegaConf.set_struct(config, True)
    return config
