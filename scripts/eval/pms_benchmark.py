import autoroot
import autorootcwd
import json
import csv
import shutil
import time
import fire
import numpy as np

from dataclasses import dataclass, asdict
from pathlib import Path
from functools import partial
from pyminisim.visual import CircleDrawing
from mppi_plus_proto.eval.env import KinematicsConfig, SocialNavEnv, PedPredictionConfig, SocialNavOutcome
from mppi_plus_proto.eval.controllers import build_controller, build_social_cost
from mppi_plus_proto.eval.traj_helpers import to_global_frame
from mppi_plus_proto.util.parallel_util import do_parallel
from mppi_plus_proto.util.random_util import seed_all
from mppi_plus_proto.util.str_util import seconds_to_human_readable_time


@dataclass
class EpisodeResult:
    idx: int
    scenario: str
    outcome: str
    num_steps: int


@dataclass
class CompositeSR:
    total: float
    random: float
    parallel: float
    circular: float


@dataclass
class CompositeAST:
    total: float
    random: float
    parallel: float
    circular: float


@dataclass
class CompositeATR:
    total: float
    random: float
    parallel: float
    circular: float


@dataclass
class BenchmarkResult:
    controller_name: str
    success_rate: CompositeSR
    average_success_time: CompositeAST
    average_timeout_rate: CompositeATR


@dataclass
class ControllerConfig:
    dynamics: str
    controller_type: str
    horizon: int
    n_samples: int | None
    stds_set: str | None
    presampler_type: str | None = None
    n_presamples: int | None = None


def calculate_metrics(controller_name: str, episode_results: list[EpisodeResult]) -> BenchmarkResult:
    n_total = len(episode_results)
    n_random = sum(1 for result in episode_results if result.scenario == "random")
    n_parallel = sum(1 for result in episode_results if result.scenario == "parallel")
    n_circular = sum(1 for result in episode_results if result.scenario == "circular")
    success_rate = CompositeSR(
        total=sum(1 for result in episode_results if result.outcome == "success") / n_total,
        random=sum(1 for result in episode_results if result.scenario == "random" and result.outcome == "success") / n_random,
        parallel=sum(1 for result in episode_results if result.scenario == "parallel" and result.outcome == "success") / n_parallel,
        circular=sum(1 for result in episode_results if result.scenario == "circular" and result.outcome == "success") / n_circular
    )
    average_success_time = CompositeAST(
        total=sum(result.num_steps for result in episode_results if result.outcome == "success") / n_total,
        random=sum(result.num_steps for result in episode_results if result.outcome == "success" and result.scenario == "random") / n_random,
        parallel=sum(result.num_steps for result in episode_results if result.outcome == "success" and result.scenario == "parallel") / n_parallel,
        circular=sum(result.num_steps for result in episode_results if result.outcome == "success" and result.scenario == "circular") / n_circular
    )
    average_timeout_rate = CompositeATR(
        total=sum(1 for result in episode_results if result.outcome == "timeout") / n_total,
        random=sum(1 for result in episode_results if result.outcome == "timeout" and result.scenario == "random") / n_random,
        parallel=sum(1 for result in episode_results if result.outcome == "timeout" and result.scenario == "parallel") / n_parallel,
        circular=sum(1 for result in episode_results if result.outcome == "timeout" and result.scenario == "circular") / n_circular
    )
    return BenchmarkResult(controller_name=controller_name, success_rate=success_rate, average_success_time=average_success_time, average_timeout_rate=average_timeout_rate)


def _seeded_benchmark(config_and_seed: tuple[ControllerConfig, int],
                      output_root: Path,
                      benchmark_name: str,
                      visualize: bool = False) -> BenchmarkResult | None:
    controller_config, task_seed = config_and_seed
    seed_all(task_seed)
    return benchmark_controller(controller_config, output_root, benchmark_name, visualize)


def benchmark_controller(controller_config: ControllerConfig,
                         output_root: Path,
                         benchmark_name: str,
                         visualize: bool = False) -> BenchmarkResult | None:
    controller, controller_name = build_controller(
        dynamics=controller_config.dynamics,
        controller_type=controller_config.controller_type,
        horizon=controller_config.horizon,
        n_samples=controller_config.n_samples,
        stds_set=controller_config.stds_set,
        presampler_type=controller_config.presampler_type,
        n_presamples=controller_config.n_presamples,
        cost=build_social_cost(),
        debug=visualize
    )

    output_dir = output_root / controller_name
    if output_dir.exists():
        return None

    env = SocialNavEnv(
        cases=f"benchmarks/social/{benchmark_name}",
        kinematics=KinematicsConfig.unicycle() if controller_config.dynamics == "unicycle" else KinematicsConfig.bicycle(),
        prediction_config=PedPredictionConfig(horizon=controller_config.horizon),
        visualize=visualize
    )

    episode_results = []

    for i in range(env.num_cases):
        controller.reset()
        obs = env.reset()
        step_count = 0
        outcome = SocialNavOutcome.STEP

        while outcome == SocialNavOutcome.STEP:
            action, info = controller.step(
                current_state=np.array([0., 0., 0.]),
                observation={
                    "goal": obs.goal,
                    "obstacles": obs.predictions
                }
            )

            if env.renderer is not None:
                ref_pose = env._sim.current_state.world.robot.pose
                x_seq = info["x_seq"][:, :2]
                x_seq = to_global_frame(x_seq, ref_pose)
                if "x_seq_samples" in info:
                    x_seq_samples = info["x_seq_samples"]
                    x_seq_samples = x_seq_samples.reshape((-1, x_seq_samples.shape[-1]))[..., :2]
                    x_seq_samples = to_global_frame(x_seq_samples, ref_pose)
                else:
                    x_seq_samples = None
                if "x_seq_pre_samples" in info:
                    x_seq_pre_samples = info["x_seq_pre_samples"]
                    x_seq_pre_samples = x_seq_pre_samples.reshape((-1, x_seq_pre_samples.shape[-1]))[..., :2]
                    x_seq_pre_samples = to_global_frame(x_seq_pre_samples, ref_pose)
                    x_seq_pre_samples_selected = info["x_seq_pre_samples_min"]
                    x_seq_pre_samples_selected = x_seq_pre_samples_selected.reshape((-1, x_seq_pre_samples_selected.shape[-1]))[..., :2]
                    x_seq_pre_samples_selected = to_global_frame(x_seq_pre_samples_selected, ref_pose)
                else:
                    x_seq_pre_samples = None
                    x_seq_pre_samples_selected = None
                renderer = env.renderer
                if x_seq_pre_samples is not None:
                    renderer.draw("pre_samples", CircleDrawing(x_seq_pre_samples, 0.03, (247, 200, 245), 0))
                    renderer.draw("pre_samples_selected", CircleDrawing(x_seq_pre_samples_selected, 0.03, (16, 150, 24), 0))
                elif x_seq_samples is not None:
                    renderer.draw("samples", CircleDrawing(x_seq_samples, 0.03, (171, 226, 245), 0))
                renderer.draw("robot_traj", CircleDrawing(x_seq, 0.05, (252, 196, 98), 0))
            
            obs, outcome = env.step(action)
            step_count += 1

        if outcome == SocialNavOutcome.ERROR:
            continue
        episode_results.append(EpisodeResult(
            idx=env.case_idx,
            scenario=env.case_scenario,
            outcome=str(outcome).lower(),
            num_steps=step_count
        ))

    result = calculate_metrics(controller_name, episode_results)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"metrics.json", "w") as f:
        json.dump(asdict(result), f, indent=4)
    with open(output_dir / f"config.json", "w") as f:
        json.dump(asdict(controller_config), f, indent=4)
    with open(output_dir / f"episode_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "scenario", "outcome", "num_steps"])
        for episode_result in episode_results:
            writer.writerow([episode_result.idx, episode_result.scenario, episode_result.outcome, episode_result.num_steps])


DYNAMICS_TYPES = ["unicycle", "bicycle"]
CONTROLLER_TYPES = ["mppi", "logmppi", "sampling"]
STDS_SETS = ["low", "high"]
PRESAMPLER_TYPES = [None, "nf", "fm", "nf_steer", "fm_steer"]
HORIZONS = [16, 26]
N_SAMPLES = [2500, 5000]
N_PRESAMPLES = [500, 1000, 2500]


def generate_configs() -> list[ControllerConfig]:
    result = []

    for controller_type in CONTROLLER_TYPES:
        if controller_type == "sampling":
            n_samples_range = [None]
            stds_range = [None]
            presamplers_range = [e for e in PRESAMPLER_TYPES if e is not None]
        else:
            n_samples_range = N_SAMPLES
            stds_range = STDS_SETS
            presamplers_range = PRESAMPLER_TYPES

        for presampler_type in presamplers_range:
            if presampler_type is None:
                n_presamples_range = [None]
            else:
                n_presamples_range = N_PRESAMPLES

            for n_presamples in n_presamples_range:
                for n_samples in n_samples_range:
                    for stds_set in stds_range:
                        for horizon in HORIZONS:
                            for dynamics in DYNAMICS_TYPES:
                                result.append(ControllerConfig(
                                    dynamics=dynamics,
                                    controller_type=controller_type,
                                    horizon=horizon,
                                    n_samples=n_samples,
                                    stds_set=stds_set,
                                    presampler_type=presampler_type,
                                    n_presamples=n_presamples,
                                ))
    return result


def main(benchmark_name: str = "medium",
         visualize: bool = False, 
         n_workers: int = 0,
         overwrite: bool = False,
         seed: int = 42):

    seed_all(seed)

    if n_workers > 1:
        visualize = False

    # config = ControllerConfig(
    #     dynamics="unicycle",
    #     controller_type="mppi",
    #     horizon=16,
    #     n_samples=2500,
    #     stds_set="low",
    #     presampler_type="nf",
    #     n_presamples=2500,
    # )
    # configs = [config]
    configs = generate_configs()
    # print(len(configs))

    output_root = Path(f"artifacts/results/social/seed_{seed}/{benchmark_name}")
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)

    task_args = [(config, seed) for config in configs]
    task_fn = partial(
        _seeded_benchmark,
        output_root=output_root, benchmark_name=benchmark_name,
        visualize=visualize)
    start_time = time.time()
    do_parallel(task_fn, task_args, n_workers, use_tqdm=True, mode="process")
    end_time = time.time()
    print(f"\nTime elapsed: {seconds_to_human_readable_time(end_time - start_time)}")
    print(f"Total number of configs: {len(configs)}")


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    fire.Fire(main)
