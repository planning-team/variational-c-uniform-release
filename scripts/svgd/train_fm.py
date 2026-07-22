import autoroot
import autorootcwd
import shutil
import time
import torch
import fire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
from mppi_plus_proto.config.omega_setup import omegaconf_setup, load_combined
from mppi_plus_proto.dynamics_models.models import AbstractDynamicsModel, UnicycleSteeringModel, UnicycleModel, BicycleSteeringModel, BicycleModel
from mppi_plus_proto.dynamics_models.rollout import RolloutCollector
from mppi_plus_proto.samplers.gd import GDSampler, MultiSelectGDSampler
from mppi_plus_proto.plotting.plots import plot_trajectories
from mppi_plus_proto.trainers.datasets import ActionsDataset
from mppi_plus_proto.nn_models.layers import VelocityMLP


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t, **extras)


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


def _save_loss_plot(loss_history: list[float], output_path: Path):
    plt.figure(figsize=(8, 3))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Flow Matching Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "loss.png")
    plt.close()


def _sample_from_vf(vf: VelocityMLP,
                    num_samples: int, 
                    device: str,
                    dataset: ActionsDataset) -> torch.Tensor:
    wrapped_vf = WrappedModel(vf)
    step_size = 0.05
    eps_time = 1e-2
    T = torch.linspace(0,1,10)  # sample times
    T = T.to(device=device)

    x_init = torch.randn((num_samples, vf.flat_input_dim), dtype=torch.float32, device=device)
    solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
    sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=False)  # sample from the model

    samples_gen = sol.reshape(num_samples, vf.horizon, vf.action_dim)
    samples_gen = dataset.unnormalize(samples_gen)

    return samples_gen


def _sample_from_dataset(dataset: ActionsDataset,
                         num_samples: int,
                         device: str) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    samples = next(iter(loader))
    samples = samples.reshape(-1, dataset.horizon, dataset.action_dim).to(device)
    samples = dataset.unnormalize(samples)
    return samples


def main(cfg: str, device: str = "cuda", overwrite: bool = False, **kwargs):
    omegaconf_setup()
    config = load_combined(cfg, cli=True, kwargs=kwargs)

    dataset_path = Path(config.dataset)
    dataset_config = OmegaConf.load(dataset_path / "config.yaml")

    dynamics, rollout_collector = _resolve_dynamics(dataset_config, device)

    output_root = Path(config.output_root)
    output_name = config.output_name
    if output_name is None:
        output_name = f"fm__{dataset_config.dynamics.type}__{dataset_config.horizon}__{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    output_path = output_root / output_name
    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"Output path already exists: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=output_path / "config.yaml", resolve=True)

    dataset = ActionsDataset(data_file=dataset_path / "samples.pt", 
                             normalization=config.preprocess.normalization)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)

    vf = VelocityMLP(action_dim=dataset.action_dim, 
                     horizon=dataset.horizon, 
                     hidden_dim=config.velocity_network.hidden_dim, 
                     n_layers=config.velocity_network.n_layers, 
                     activation=config.velocity_network.activation, 
                     time_dim=1, 
                     init_method=config.velocity_network.init_method)
    vf.to(device)
    vf.train()

    prob_path = AffineProbPath(scheduler=CondOTScheduler())
    optim = torch.optim.Adam(vf.parameters(), lr=config.training.lr)

    loss_history = []
    start_time = time.time()
    i = 0
    for epoch in range(config.training.n_epochs):
        for x_1 in loader:
            optim.zero_grad() 
            x_1 = x_1.reshape(-1, dataset.action_dim * dataset.horizon).to(device)
            # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
            x_0 = torch.randn_like(x_1).to(device)
            # sample time (user's responsibility)
            t = torch.rand(x_1.shape[0]).to(device) 
            # sample probability path
            path_sample = prob_path.sample(t=t, x_0=x_0, x_1=x_1)
            # flow matching l2 loss
            loss = torch.pow( vf(path_sample.x_t,path_sample.t) - path_sample.dx_t, 2).mean() 
            # optimizer step
            loss.backward() # backward
            optim.step() # update
            loss_history.append(loss.item())
            # log loss
            if (i+1) % config.training.print_every_step == 0:
                elapsed = time.time() - start_time
                print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ' 
                    .format(i+1, elapsed*1000/config.training.print_every_step, loss.item())) 
                start_time = time.time()
            i += 1

    torch.save(vf.state_dict(), output_path / "model.pth")

    _save_loss_plot(loss_history, output_path)

    samples_gen = _sample_from_vf(vf, config.n_vis_samples, device, dataset)
    samples_gt = _sample_from_dataset(dataset, config.n_vis_samples, device)

    traj_gen = rollout_collector.rollout(samples_gen, clip=True).cpu().numpy()
    traj_gt = rollout_collector.rollout(samples_gt, clip=True).cpu().numpy()

    plot_trajectories(traj_gen,
                      title=f"Flow Matching Samples",
                      output_file=output_path / "samples.png",
                      headless=True, dpi=150)
    plot_trajectories(traj_gt,
                      title=f"Ground Truth Samples",
                      output_file=output_path / "samples_gt.png",
                      headless=True, dpi=150)


if __name__ == "__main__":
    fire.Fire(main)
