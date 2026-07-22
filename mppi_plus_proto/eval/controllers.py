import numpy as np
import oo_ctrl as octrl

from pathlib import Path
from mppi_plus_proto.mppi.mppi_dynamics_models import BicycleSteerMPPIModel, UnicycleSteerMPPIModel
from mppi_plus_proto.mppi.mppi_presamplers import (
    build_nf_presampler,
    build_fm_presampler,
    build_nln_action_sampler
)
from mppi_plus_proto.mppi.controllers import SamplingController


# Hardcoded from other experiments
_POLICY_DT = 0.2
_MAX_SPEED = 1.
_UNICYCLE_ANGULAR = float(np.deg2rad(45.))
_BICYCLE_ANGLE = float(np.deg2rad(30.))
_L = 0.324
_STDS = {
    "unicycle": {
        "low": (0.15, float(np.deg2rad(14.))),
        "high": (0.4, float(np.deg2rad(34.))),
    },
    "bicycle": {
        "low": (0.15, float(np.deg2rad(8.))),
        "high": (0.4, float(np.deg2rad(20.))),
    },
}
_LMBDA = 0.154


def build_controller(
    dynamics: str,
    controller_type: str,
    horizon: int,
    n_samples: int | None,
    stds_set: str | None,
    cost: octrl.np.AbstractNumPyCost,
    presampler_type: str | None = None,
    n_presamples: int | None = None,
    debug: bool = False
) -> tuple[octrl.np.MPPI | SamplingController, str]:
    assert dynamics in ("unicycle", "bicycle"), \
        f"Unknown dynamics {dynamics}. Must be 'unicycle' or 'bicycle'."
    assert controller_type in ("mppi", "logmppi", "sampling"), \
        f"Unknown controller type {controller_type}. Must be 'mppi' or 'logmppi' or 'sampling'."
    assert cost is not None, "Cost cannot be None."
    if controller_type != "sampling":
        assert stds_set in ("low", "high"), \
            f"Unknown stds set {stds_set}. Must be 'low' or 'high'."
        assert presampler_type is None or presampler_type in ("nf", "fm", "nf_steer", "fm_steer", "nln"), \
            f"Unknown presampler {presampler_type}. Must be 'nf' or 'fm' or 'nn_cuniform_steer' or 'nln'."
        if presampler_type is not None:
            assert n_presamples is not None, "n_presamples must be provided if presampler_type is not None."
    else:
        assert presampler_type in ("nf", "fm", "nf_steer", "fm_steer", "nn_cuniform_steer", "nln"), \
            f"Unknown presampler {presampler_type}. Must be 'nf' or 'fm' or 'nn_cuniform_steer' or 'nln'."
        assert n_presamples is not None, "n_presamples must be provided if controller_type is 'sampling'."

    if dynamics == "unicycle":
        model = octrl.np.UnicycleModel(
            dt=_POLICY_DT,
            linear_bounds=(0., _MAX_SPEED),
            angular_bounds=(-_UNICYCLE_ANGULAR, _UNICYCLE_ANGULAR)
        )
        state_transform = None
    elif dynamics == "bicycle":
        model = octrl.np.BicycleModel(
            dt=_POLICY_DT,
            wheel_base=_L,
            linear_bounds=(0., _MAX_SPEED),
            angular_bounds=(-_BICYCLE_ANGLE, _BICYCLE_ANGLE)
        )
        state_transform = octrl.np.RearToCenterTransform(_L)
    else:
        raise ValueError(f"Unknown dynamics {dynamics}")

    if controller_type != "sampling":
        stds = _STDS[dynamics][stds_set]
    else:
        stds = None

    if controller_type == "mppi":
        sampler = octrl.np.GaussianActionSampler(
            stds=stds
        )
    elif controller_type == "logmppi":
        sampler = octrl.np.NLNActionSampler(
            stds=stds
        )
    elif controller_type == "sampling":
        sampler = None
    else:
        raise ValueError(f"Unknown controller type {controller_type}")

    if presampler_type is None:
        presampler = None
    elif presampler_type.startswith("nf"):
        presampler = build_nf_presampler(
            dynamics_type=dynamics,
            steer = presampler_type.endswith("steer"),
            n_samples=n_presamples,
            horizon=horizon
        )
    elif presampler_type.startswith("fm"):
        presampler = build_fm_presampler(
            dynamics_type=dynamics,
            steer = presampler_type.endswith("steer"),
            n_samples=n_presamples,
            horizon=horizon
        )
    elif presampler_type == "nln":
        presampler = build_nln_action_sampler(
            dynamics_type=dynamics,
            steer = presampler_type.endswith("steer"),
            horizon=horizon,
            n_samples=n_presamples
        )
    else:
        raise ValueError(f"Unknown presampler type {presampler_type}")
    
    if controller_type == "mppi" or controller_type == "logmppi":
        controller = octrl.np.MPPI(
            horizon=horizon,
            n_samples=n_samples,
            lmbda=_LMBDA,
            model=model,
            biased=False,
            sampler=sampler,
            cost=cost,
            state_transform=state_transform,
            return_pre_samples=debug,
            return_samples=debug,
            return_state_seq=debug,
            presampler=presampler,
            u_init="uniform"
        )
    elif controller_type == "sampling":
        controller = SamplingController(
            model=model,
            cost=cost,
            presampler=presampler,
            state_transform=state_transform,
            cost_monitor=debug,
            return_state_seq=debug,
            return_pre_samples=debug)
    else:
        raise ValueError(f"Unknown controller type {controller_type}")

    controller_name = f"{controller_type}__{dynamics}__h_{horizon}__stds_{stds_set}__samples_{n_samples}"
    if presampler_type is not None:
        controller_name += f"__{presampler_type}__presamples_{n_presamples}"

    return controller, controller_name


def build_social_cost() -> list[octrl.np.AbstractNumPyCost]:
    cost=[
        octrl.np.EuclideanRatioGoalCost(Q=12.,
                                        squared=False,
                                        state_dims=2,
                                        name="goal"),
        octrl.np.CollisionIndicatorCost(Q=10000.,
                                        safe_distance=0.85,
                                        name="CA")
    ]
    return cost



# def get_available_controllers(config_path: str | Path) -> list[str]:
#     with open(config_path, "r") as f:
#         return list(yaml.safe_load(f)["controllers"].keys())


# def load_controller(config_path: str | Path,
#                     controller_name: str,
#                     cost,
#                     debug: bool = False):
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)["controllers"][controller_name]

#     state_transform = None
#     presampler = None

#     if config["type"] == "mppi":
#         sampler = octrl.np.GaussianActionSampler(
#             stds=(np.sqrt(config["variance"]),)
#         )
#     elif config["type"] == "logmppi":
#         sampler = octrl.np.NLNActionSampler(
#             stds=(np.sqrt(config["variance"]),)
#         )
#     else:
#         raise ValueError(f"Unknown controller type {config['type']}")


#     if config["model"] == "bicycle_steer":
#         angle_limit = np.deg2rad(config["angle_limit"])
#         model = BicycleSteerMPPIModel(dt=config["dt"],
#                                       wheel_base=config["wheel_base"],
#                                       speed=config["speed"],
#                                       angular_bounds=(-angle_limit, angle_limit))
#         state_transform = octrl.np.RearToCenterTransform(config["wheel_base"])
#         if "presampler" in config:
#             if config["presampler"] == "nf":
#                 presampler = presampler_bicycle_steer(n_samples=config["n_presamples"])
#                 assert presampler._horizon == config["horizon"]
#             else:
#                 raise ValueError(f"Unknown config {config['presampler']}")
    
#     elif config["model"] == "unicycle_steer":
#         angular_vel_limit = np.deg2rad(config["angular_vel_limit"])
#         model = UnicycleSteerMPPIModel(dt=config["dt"],
#                                        speed=config["speed"],
#                                        angular_bounds=(-angular_vel_limit, angular_vel_limit))
#         if "presampler" in config:
#             if config["presampler"] == "nf":
#                 presampler = presampler_unicycle_steer(n_samples=config["n_presamples"])
#                 assert presampler._horizon == config["horizon"]
#             else:
#                 raise ValueError(f"Unknown config {config['presampler']}")
#     else:
#         raise ValueError(f"Unknown model {config['modle']}") 

#     return octrl.np.MPPI(
#         horizon=config["horizon"],
#         n_samples=config["n_samples"],
#         lmbda=config["lmbda"],
#         model=model,
#         biased=False,
#         sampler=sampler,
#         cost=cost,
#         state_transform=state_transform,
#         return_pre_samples=debug,
#         return_samples=debug,
#         return_state_seq=debug,
#         presampler=presampler
#     )
