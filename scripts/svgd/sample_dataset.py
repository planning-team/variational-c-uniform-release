import autoroot
import autorootcwd
import shutil
import torch
import fire

from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from mppi_plus_proto.config.omega_setup import omegaconf_setup, load_combined
from mppi_plus_proto.dynamics_models.models import AbstractDynamicsModel, UnicycleSteeringModel, UnicycleModel, BicycleSteeringModel, BicycleModel
from mppi_plus_proto.dynamics_models.rollout import RolloutCollector
from mppi_plus_proto.samplers.gd import GDSampler, MultiSelectGDSampler
from mppi_plus_proto.plotting.plots import plot_trajectories


def _resolve_dynamics(config: DictConfig, device: str) -> tuple[AbstractDynamicsModel, RolloutCollector]:
    dynamics_cfg = config.dynamics
    backend = "torch"
    # state_dim_cut is 2 for all "flat" models
    if dynamics_cfg.type == "unicycle_steer":
        dynamics_cls = UnicycleSteeringModel
        state_dim_cut = 2
    elif dynamics_cfg.type == "unicycle":
        dynamics_cls = UnicycleModel
        state_dim_cut = 2
    elif dynamics_cfg.type == "bicycle_steer":
        dynamics_cls = BicycleSteeringModel
        state_dim_cut = 2
    elif dynamics_cfg.type == "bicycle":
        dynamics_cls = BicycleModel
        state_dim_cut = 2
    else:
        raise ValueError(f"Invalid dynamics type: {dynamics_cfg.type}")
    dynamics = dynamics_cls(backend=backend, device=device, **dynamics_cfg.params)
    rollout_collector = RolloutCollector(dynamics=dynamics, 
                                         horizon=config.horizon,
                                         action_lb=dynamics_cfg.action_lb,
                                         action_ub=dynamics_cfg.action_ub,
                                         state_dim_cut=state_dim_cut)
    return dynamics, rollout_collector


def _resolve_sampler(sampler_cfg: DictConfig, 
                     rollout_collector: RolloutCollector, 
                     base_sampler: GDSampler | None = None) -> GDSampler:
    sampler_type = sampler_cfg.type
    sampler_args = {k: v for k, v in sampler_cfg.items() if k != "type"}
    if sampler_type == "gd":
        return GDSampler(rollout_collector=rollout_collector, **sampler_args)
    elif sampler_type == "multi_select_gd":
        return MultiSelectGDSampler(base_sampler=base_sampler, **sampler_args)
    else:
        raise ValueError(f"Invalid sampler type: {sampler_type}")
        

def main(cfg: str, device: str = "cuda", overwrite: bool = False, **kwargs):
    omegaconf_setup()
    config = load_combined(cfg, cli=True, kwargs=kwargs)

    dynamics, rollout_collector = _resolve_dynamics(config, device)
    sampler = None
    for sampler_cfg in config.samplers:
        sampler = _resolve_sampler(sampler_cfg, rollout_collector, sampler)
    if sampler is None:
        raise ValueError("No sampler found")

    output_root = Path(config.output_root)
    output_name = config.output_name
    if output_name is None:
        output_name = f"gd__{config.dynamics.type}__{config.horizon}__{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    output_path = output_root / output_name
    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"Output path already exists: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=output_path / "config.yaml", resolve=True)

    result_sampels = sampler.sample()
    torch.save(result_sampels.cpu(), output_path / "samples.pt")

    n_vis_samples = config.n_vis_samples
    if n_vis_samples is not None:
        shuffled_indices = torch.randperm(result_sampels.shape[0])
        vis_samples = result_sampels[shuffled_indices[:n_vis_samples]]   
        vis_trajectories = rollout_collector.rollout(vis_samples).cpu().numpy()
        plot_trajectories(vis_trajectories, 
                          title=f"SVGD Samples", 
                          output_file=output_path / "samples.png", 
                          headless=True, dpi=150)


if __name__ == "__main__":
    fire.Fire(main)
