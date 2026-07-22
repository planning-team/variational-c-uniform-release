import json
import numpy as np
import torch
import torch.nn as nn
import oo_ctrl as octrl

from pathlib import Path
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
from mppi_plus_proto.nn_models.generative import generic_realnvp
from mppi_plus_proto.nn_models.layers import VelocityMLP


def _filter_presamples(samples: np.ndarray) -> np.ndarray:
    # Samples shape: (n_samples, horizon, action_dim)
    mask = np.isfinite(samples).all(axis=(1, 2))
    return samples[mask]


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t, **extras)


class NFPresampler(octrl.np.AbstractPresampler):

    def __init__(self,
                 nf_model: nn.Module,
                 weights_path: str | Path,
                 n_samples: int,
                 horizon: int,
                 steer: bool,
                 u_lb: tuple[float, float],
                 u_ub: tuple[float, float],
                 steer_speed: float = 1.,
                 device: str = "cuda") -> None:
        super(NFPresampler, self).__init__()
        nf_model = nf_model.to(device)
        nf_model.load_state_dict(torch.load(str(weights_path), 
                                 map_location=device))
        nf_model.eval()
        self._nf_model = nf_model
        self._n_samples = n_samples
        if not steer:
            self._action_dim = 2
        else:
            self._action_dim = 1
            self._steer_speed = steer_speed
        self._steer = steer
        self._u_lb = np.array(u_lb)
        self._u_ub = np.array(u_ub)
        self._horizon = horizon
        self._device = device

    def sample(self, state, observation) -> np.ndarray:
        with torch.inference_mode():
            samples, _ = self._nf_model.sample(self._n_samples)
            samples = samples.reshape((self._n_samples, self._horizon, self._action_dim))
            samples = samples.cpu().numpy()
        if self._steer:
            speed = np.ones((self._n_samples, self._horizon, 1)) * self._steer_speed
            samples = np.concatenate((speed, samples), axis=-1)
        samples = _filter_presamples(samples)
        samples = np.clip(samples, self._u_lb, self._u_ub)
        return samples


class FMPresampler(octrl.np.AbstractPresampler):

    def __init__(self,
                 weights_root: str | Path,
                 n_samples: int,
                 horizon: int,
                 steer: bool,
                 u_lb: tuple[float, float],
                 u_ub: tuple[float, float],
                 steer_speed: float = 1.,
                 device: str = "cuda") -> None:
        super(FMPresampler, self).__init__()
        weights_root = Path(weights_root)

        vf = VelocityMLP(action_dim=2 if not steer else 1, 
                         horizon=horizon, 
                         hidden_dim=512, 
                         n_layers=3, 
                         activation="silu", 
                         time_dim=1)
        vf = vf.to(device)
        vf.load_state_dict(torch.load(str(weights_root / "model.pth"), 
                           map_location=device))
        vf = vf.eval()
        self._vf = vf
        self._wrapped_vf = WrappedModel(vf)
        self._u_lb = np.array(u_lb)
        self._u_ub = np.array(u_ub)

        with open(weights_root / "stats.json", "r") as f:
            stats = json.load(f)
        self._stds = np.array(stats["stds"])

        self._step_size = 0.05
        self._eps_time = 1e-2
        self._T = torch.linspace(0,1,10)  # sample times
        self._T = self._T.to(device=device)

        self._n_samples = n_samples
        self._device = device
        self._steer = steer
        self._steer_speed = steer_speed

    def sample(self, state, observation) -> np.ndarray:
        with torch.inference_mode():
            x_init = torch.randn((self._n_samples, self._vf.flat_input_dim), dtype=torch.float32, device=self._device)
            solver = ODESolver(velocity_model=self._wrapped_vf)  # create an ODESolver class
            sol = solver.sample(time_grid=self._T, x_init=x_init, method='midpoint', step_size=self._step_size, return_intermediates=False)  # sample from the model

            samples = sol.reshape(self._n_samples, self._vf.horizon, self._vf.action_dim)
            samples = samples.clone().cpu().numpy()
        
        samples = self._stds * samples

        if self._steer:
            speed = np.ones((self._n_samples, self._vf.horizon, 1)) * self._steer_speed
            samples = np.concatenate((speed, samples), axis=-1)

        samples = _filter_presamples(samples)
        samples = np.clip(samples, self._u_lb, self._u_ub)

        return samples


class NLNActionPreSampler(octrl.np.AbstractPresampler):

    def __init__(self,
                 stds: tuple[float, ...],
                 means: tuple[float, ...],
                 horizon: int,
                 n_samples: int,
                 u_lb: tuple[float, float],
                 u_ub: tuple[float, float],
                 steer_speed: float | None = None):
        super(NLNActionPreSampler, self).__init__()
        # Variance of the normal distribution
        var_n = np.array(stds) ** 2
        mu_n = np.array(means)
        # Mean and variance of the log-normal distribution
        mu_ln = np.exp(0.5 * var_n)
        var_ln = np.exp(var_n) * (np.exp(var_n) - 1.)
        
        var_nln = var_n * np.exp(2 * mu_ln + 2 * var_ln)

        self._dim = len(stds)
        self._mu_n = mu_n
        self._std_n = np.sqrt(var_n)
        self._mu_ln = mu_ln
        self._std_ln = np.sqrt(var_ln)
        self._covariance_matrix = np.diag(var_nln)

        self._n_samples = n_samples
        self._horizon = horizon
        self._steer_speed = steer_speed

        self._u_lb = np.array(u_lb)
        self._u_ub = np.array(u_ub)

    def sample(self, state, observation) -> np.ndarray:
        samples_n = np.random.normal(self._mu_n, self._std_n, (self._n_samples, self._horizon, self._dim))
        samples_ln = np.random.lognormal(self._mu_ln, self._std_ln, (self._n_samples, self._horizon, self._dim))
        samples = samples_n * samples_ln
        if self._steer_speed is not None:
            samples = np.concatenate((np.ones((self._n_samples, self._horizon, 1)) * self._steer_speed, samples), axis=-1)
        samples = np.clip(samples, self._u_lb, self._u_ub)
        return samples


class GaussianPresampler(octrl.np.AbstractPresampler):

    def __init__(self,
                 stds: tuple[float, ...],
                 means: tuple[float, ...],
                 horizon: int,
                 n_samples: int,
                 u_lb: tuple[float, float],
                 u_ub: tuple[float, float],
                 steer_speed: float | None = None):
        super(GaussianPresampler, self).__init__()
        self._stds = np.array(stds)
        self._means = np.array(means)
        self._horizon = horizon
        self._n_samples = n_samples
        self._u_lb = np.array(u_lb)
        self._u_ub = np.array(u_ub)
        self._dim = len(stds)
        self._steer_speed = steer_speed

    def sample(self, state, observation) -> np.ndarray:
        samples = np.random.normal(self._means, self._stds, (self._n_samples, self._horizon, self._dim))
        if self._steer_speed is not None:
            samples = np.concatenate((np.ones((self._n_samples, self._horizon, 1)) * self._steer_speed, samples), axis=-1)
        samples = np.clip(samples, self._u_lb, self._u_ub)
        return samples


class UniformPresampler(octrl.np.AbstractPresampler):
    def __init__(self,
                 horizon: int,
                 n_samples: int,
                 u_lb: tuple[float, ...],
                 u_ub: tuple[float, ...]):
        super(UniformPresampler, self).__init__()
        self._horizon = horizon
        self._n_samples = n_samples
        self._u_lb = np.tile(np.array(u_lb), (self._n_samples, self._horizon, 1))
        self._u_ub = np.tile(np.array(u_ub), (self._n_samples, self._horizon, 1))

    def sample(self, state, observation) -> np.ndarray:
        samples = np.random.uniform(self._u_lb, self._u_ub)
        return samples


def build_nf_presampler(
    dynamics_type: str,
    steer: bool,
    n_samples: int,
    horizon: int
) -> NFPresampler:
    if dynamics_type == "bicycle" and not steer:
        num_layers = 64
    else:
        num_layers = 32

    if dynamics_type == "unicycle":
        u_lb = (0.1, -float(np.deg2rad(45.)))
        u_ub = (1.0, float(np.deg2rad(45.)))
    else:
        u_lb = (0.1, -float(np.deg2rad(30.)))
        u_ub = (1.0, float(np.deg2rad(30.)))

    model = generic_realnvp(action_dim=2 if not steer else 1,
                            num_layers=num_layers,
                            horizon=horizon)
    dynamics_name = dynamics_type + ("_steer" if steer else "")
    weights_path = f"deploy_checkpoints/nf/horizon_{horizon}/nf__{dynamics_name}__{horizon}/model.pth"
    return NFPresampler(
        nf_model=model,
        weights_path=weights_path,
        n_samples=n_samples,
        horizon=horizon,
        steer=steer,
        u_lb=u_lb,
        u_ub=u_ub,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


def build_fm_presampler(
    dynamics_type: str,
    steer: bool,
    n_samples: int,
    horizon: int
) -> FMPresampler:
    dynamics_name = dynamics_type + ("_steer" if steer else "")
    weights_root = Path(f"deploy_checkpoints/fm/horizon_{horizon}/fm__{dynamics_name}__{horizon}/")
    if dynamics_type == "unicycle":
        u_lb = (0.1, -float(np.deg2rad(45.)))
        u_ub = (1.0, float(np.deg2rad(45.)))
    else:
        u_lb = (0.1, -float(np.deg2rad(30.)))
        u_ub = (1.0, float(np.deg2rad(30.)))
    return FMPresampler(
        weights_root=weights_root,
        n_samples=n_samples,
        horizon=horizon,
        steer=steer,
        u_lb=u_lb,
        u_ub=u_ub,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


def build_nln_action_sampler(
    dynamics_type: str,
    horizon: int,
    n_samples: int,
    steer: bool,
) -> NLNActionPreSampler:
    # if dynamics_type == "unicycle":
    #     u_lb = (0.1, -float(np.deg2rad(45.)))
    #     u_ub = (1.0, float(np.deg2rad(45.)))
    # else:
    #     u_lb = (0.1, -float(np.deg2rad(30.)))
    #     u_ub = (1.0, float(np.deg2rad(30.)))
    if dynamics_type == "unicycle":
        angle_ampl = float(np.deg2rad(45.))
        stds = (float(np.deg2rad(34.)),)
    else:
        angle_ampl = float(np.deg2rad(30.))
        stds = (float(np.deg2rad(20.)),)
    if not steer:
        speed_lb = 0.1
        speed_ub = 1.
        steer_speed = None
        stds = (0.4,) + stds
        means = (0.55, 0.)
    else:
        speed_lb = 1.
        speed_ub = 1.
        steer_speed = 1.
        means = (0.,)
    u_lb = (speed_lb, -angle_ampl)
    u_ub = (speed_ub, angle_ampl)

    return NLNActionPreSampler(
        stds=stds,
        means=means,
        horizon=horizon,
        n_samples=n_samples,
        steer_speed=steer_speed,
        u_lb=u_lb,
        u_ub=u_ub
    )


def build_gaussian_sampler(
    dynamics_type: str,
    horizon: int,
    n_samples: int,
    steer: bool,
) -> GaussianPresampler:
    if dynamics_type == "unicycle":
        angle_ampl = float(np.deg2rad(45.))
        stds = (float(np.deg2rad(34.)),)
    else:
        angle_ampl = float(np.deg2rad(30.))
        stds = (float(np.deg2rad(20.)),)
    if not steer:
        speed_lb = 0.1
        speed_ub = 1.
        steer_speed = None
        stds = (0.4,) + stds
        means = (0.55, 0.)
    else:
        speed_lb = 1.
        speed_ub = 1.
        steer_speed = 1.
        means = (0.,)
    u_lb = (speed_lb, -angle_ampl)
    u_ub = (speed_ub, angle_ampl)
    
    return GaussianPresampler(
        stds=stds,
        means=means,
        horizon=horizon,
        n_samples=n_samples,
        u_lb=u_lb,
        u_ub=u_ub,
        steer_speed=steer_speed
    )


def build_uniform_sampler(
    dynamics_type: str,
    steer: bool,
    horizon: int,
    n_samples: int
) -> UniformPresampler:

    if dynamics_type == "unicycle":
        angle_ampl = float(np.deg2rad(45.))
    else:
        angle_ampl = float(np.deg2rad(30.))
    if not steer:
        speed_lb = 0.1
        speed_ub = 1.
    else:
        speed_lb = 1.
        speed_ub = 1.
    return UniformPresampler(
        horizon=horizon,
        n_samples=n_samples,
        u_lb=(speed_lb, -angle_ampl),
        u_ub=(speed_ub, angle_ampl)
    )
