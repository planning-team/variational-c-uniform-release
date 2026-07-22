from configparser import NoSectionError
import autoroot
import autorootcwd
import numpy as np
import matplotlib.pyplot as plt
import fire
import time
import multiprocessing
from typing import Callable
from pathlib import Path
from dataclasses import dataclass
from functools import partial
from scipy.stats import gaussian_kde
from matplotlib.colors import to_rgba
from oo_ctrl.np.core import AbstractPresampler
from mppi_plus_proto.mppi.mppi_presamplers import (
    build_nf_presampler,
    build_fm_presampler,
    build_nln_action_sampler,
    build_gaussian_sampler,
    build_uniform_sampler,
)
from mppi_plus_proto.util.parallel_util import do_parallel
from mppi_plus_proto.dynamics_models.rollout import RolloutCollector
from mppi_plus_proto.dynamics_models.models import (
    AbstractDynamicsModel, 
    UnicycleSteeringModel, UnicycleModel, 
    BicycleSteeringModel, BicycleModel)


DT = 0.2
L = 0.324
N_SAMPLES = 10000


@dataclass
class SamplerPlotTask:
    dynamics_type: str
    horizon: int
    sampler_types: list[str]


def _setup_matplotlib(headless: bool) -> None:
    if headless:
        plt.switch_backend("Agg")


def plot_trajectories(trajectories: np.ndarray,
                      dashed_trajectories: np.ndarray | None = None,
                      ax: plt.Axes | None = None,
                      figsize: tuple[int, int] = (6, 6),
                      title: str | None = None,
                      output_file: str | Path | None = None,
                      headless: bool = False,
                      dpi: int = 300) -> None:
    """
    Plot multiple trajectories on a 2D plane.
    
    Args:
        trajectories: Array of shape (n_trajectories, horizon, state_dim)
        ax: Matplotlib axes to plot on. If None, creates new figure
        figsize: Figure size for the plot (only used if ax is None)
        title: Title for the plot
        output_file: Optional output image path for saving the figure
        headless: If True, forces non-interactive Agg backend
        dpi: Image resolution used when saving
    """
    _setup_matplotlib(headless)

    created_figure = False
    # Create figure if no axes provided
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        created_figure = True
    
    # Plot each trajectory
    for i in range(trajectories.shape[0]):
        traj = trajectories[i]
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.3)

    if dashed_trajectories is not None:
        for i in range(dashed_trajectories.shape[0]):
            traj = dashed_trajectories[i]
            ax.plot(traj[:, 0], traj[:, 1], alpha=1, linestyle="--")

    # Set labels and grid
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.set_aspect('equal')
    
    # Set consistent axis limits based on data
    margin = 0.5
    x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
    y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    
    if title is not None:
        ax.set_title(title)

    if output_file is not None:
        ax.figure.savefig(str(output_file), dpi=dpi, bbox_inches="tight")

    # In headless mode avoid retaining figures in memory.
    if created_figure and (headless or output_file is not None):
        plt.close(ax.figure)


def plot_coverage_contours(ax, points_2d, color, label, levels_pct=(0.95, 0.75, 0.5)):
    """Draw KDE contour lines enclosing given percentages of the density mass."""
    xy = points_2d.T  # shape (2, N)
    kde = gaussian_kde(xy)

    # Build evaluation grid
    x_min, x_max = xy[0].min() - 0.5, xy[0].max() + 0.5
    y_min, y_max = xy[1].min() - 0.5, xy[1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # Convert percentages to density thresholds via sorted density values
    zz_sorted = np.sort(zz.ravel())
    cumsum = np.cumsum(zz_sorted)
    cumsum /= cumsum[-1]
    thresholds = []
    for pct in sorted(levels_pct):
        thresholds.append(zz_sorted[np.searchsorted(cumsum, 1.0 - pct)])

    # Deduplicate and ensure strictly increasing
    thresholds = sorted(set(thresholds))
    thresholds = [t for i, t in enumerate(thresholds)
                  if i == 0 or t - thresholds[i - 1] > 1e-12]

    if not thresholds:
        ax.plot([], [], color=color, linewidth=2, label=label)
        return

    # Draw filled contour for the outermost level
    ax.contourf(xx, yy, zz, levels=[thresholds[0], zz.max()],
                colors=[to_rgba(color, 0.15)])
    if len(thresholds) > 1:
        ax.contourf(xx, yy, zz, levels=[thresholds[-1], zz.max()],
                    colors=[to_rgba(color, 0.10)])

    linewidths = [2.0, 1.2, 0.8][:len(thresholds)]
    ax.contour(xx, yy, zz, levels=thresholds, colors=[color],
               linewidths=linewidths)
    ax.plot([], [], color=color, linewidth=2, label=label)


def plot_contours(sampler_results: list[tuple[np.ndarray, str, str, str]],
                  title: str | None,
                  output_file: str | Path):
    _setup_matplotlib(True)

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    for x_seq, label, color in sampler_results:
        plot_coverage_contours(ax, x_seq, f"tab:{color}", label)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    if title is not None:
        ax.set_title(title)

    ax.figure.savefig(str(output_file), dpi=300, bbox_inches="tight")
    plt.close(ax.figure)


PLOT_TRAJECTORIES = "trajectories"
PLOT_COVERAGE = "coverage"
ALL_PLOT_TYPES = [PLOT_TRAJECTORIES, PLOT_COVERAGE]


def make_plots(task: SamplerPlotTask,
               output_root: Path,
               plot_types: list[str] = ALL_PLOT_TYPES,
               overwrite: bool = False):
    model_type = task.dynamics_type
    if model_type.startswith("unicycle"):
        model = UnicycleModel(
            dt=DT,
            backend="numpy",
            angle_repr="none",
            wrap_angle=True
        )
    elif model_type.startswith("bicycle"):
        model = BicycleModel(
            dt=DT,
            l=L,
            backend="numpy",
            angle_repr="none",
            wrap_angle=True
        )
    else:
        raise ValueError(f"Unknown dynamics type: {model_type}")
    
    horizon = task.horizon
    steer = model_type.endswith("steer")
    model_type_general = model_type.split("_")[0]

    model_subpath = model_type / Path(f"horizon_{horizon}")

    do_trajectories = PLOT_TRAJECTORIES in plot_types
    do_coverage = PLOT_COVERAGE in plot_types

    output_dir_trajectories = output_root / "trajectories" / model_subpath
    output_file_coverage = output_root / "coverage" / model_subpath / "contours.png"

    if not overwrite:
        if do_trajectories and output_dir_trajectories.exists():
            do_trajectories = False
        if do_coverage and output_file_coverage.exists():
            do_coverage = False

    if not do_trajectories and not do_coverage:
        return

    samplers = []
    for sampler_type in task.sampler_types:
        if sampler_type == "nf":
            samplers.append((build_nf_presampler(
                dynamics_type=model_type_general,
                steer=steer,
                horizon=horizon,
                n_samples=N_SAMPLES
            ), sampler_type, "Normalizing Flow", "blue"))
        elif sampler_type == "fm":
            samplers.append((build_fm_presampler(
                dynamics_type=model_type_general,
                steer=steer,
                horizon=horizon,
                n_samples=N_SAMPLES
            ), sampler_type, "Flow Matching", "purple"))
        elif sampler_type == "nln":
            samplers.append((build_nln_action_sampler(
                dynamics_type=model_type_general,
                steer=steer,
                horizon=horizon,
                n_samples=N_SAMPLES
            ), sampler_type, "Normal Log-Normal", "orange"))
        elif sampler_type == "gaussian":
            samplers.append((build_gaussian_sampler(
                dynamics_type=model_type_general,
                horizon=horizon,
                n_samples=N_SAMPLES,
                steer=steer
            ), sampler_type, "Gaussian", "green"))
        elif sampler_type == "uniform":
            samplers.append((build_uniform_sampler(
                dynamics_type=model_type_general,
                horizon=horizon,
                n_samples=N_SAMPLES,
                steer=steer
            ), sampler_type, "Uniform", "red"))
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

    sampler_results = []
    if do_trajectories:
        output_dir_trajectories.mkdir(parents=True, exist_ok=True)
    for sampler, sampler_type, name, color in samplers:
        trajectories = model.rollout(sampler.sample(None, None))
        sampler_results.append((trajectories.reshape((-1, 2)), name, color))
        if do_trajectories:
            plot_trajectories(trajectories=trajectories,
                              title=None,
                              output_file=output_dir_trajectories / f"{sampler_type}.png",
                              headless=True)

    if do_coverage:
        output_file_coverage.parent.mkdir(parents=True, exist_ok=True)
        plot_contours(sampler_results,
                      title=None,
                      output_file=output_file_coverage)


def build_tasks() -> list[SamplerPlotTask]:
    dynamics_types = ["unicycle", "bicycle", "unicycle_steer", "bicycle_steer"]
    horizons = [16, 26]
    sampler_types = ["nf", "fm", "nln", "gaussian"]

    tasks = []
    
    for dynamics_type in dynamics_types:
        for horizon in horizons:
            for sampler_type in sampler_types:
                tasks.append(SamplerPlotTask(
                    dynamics_type=dynamics_type,
                    horizon=horizon,
                    sampler_types=sampler_types.copy()
                ))
    return tasks


def main(output_root: str = "artifacts/figures/sampling_analysis/",
         n_workers: int = 0,
         overwrite: bool = False,
         plot_types: list[str] | None = None):
    if plot_types is None:
        plot_types = list(ALL_PLOT_TYPES)
    unknown = set(plot_types) - set(ALL_PLOT_TYPES)
    if unknown:
        raise ValueError(
            f"Unknown plot types: {unknown}. "
            f"Choose from {ALL_PLOT_TYPES}")
    output_root = Path(output_root)

    tasks = build_tasks()
    task_fn = partial(make_plots, output_root=output_root,
                      plot_types=plot_types, overwrite=overwrite)

    start_ts = time.time()
    do_parallel(task_fn, tasks, n_workers, use_tqdm=True, mode="process")
    end_ts = time.time()
    print(f"Time elapsed: {end_ts - start_ts} seconds")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    fire.Fire(main)
