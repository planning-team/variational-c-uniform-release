import matplotlib.pyplot as plt
import numpy as np
import torch

from mppi_plus_proto.dynamics_models.rollout import RolloutCollector
from mppi_plus_proto.trainers.data_samplers import sample_uniform


def _setup_matplotlib(headless: bool) -> None:
    if headless:
        plt.switch_backend("Agg")


def plot_trajectories(trajectories: np.ndarray,
                      dashed_trajectories: np.ndarray | None = None,
                      ax: plt.Axes | None = None,
                      figsize: tuple[int, int] = (6, 6),
                      title: str | None = None,
                      output_file: str = None,
                      headless: bool = False,
                      dpi: int = 150) -> None:
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
        ax.figure.savefig(output_file, dpi=dpi, bbox_inches="tight")

    # In headless mode avoid retaining figures in memory.
    if created_figure and (headless or output_file is not None):
        plt.close(ax.figure)


# def plot_model(ax, 
#                model, 
#                n_samples: int,
#                title: str,
#                rollout_collector: RolloutCollector,
#                reference_trajectories: np.ndarray):
#     with torch.no_grad():
#         u_seq, _ = model.sample(n_samples)
#         u_seq = u_seq.reshape(n_samples, rollout_collector.horizon, -1)
#         x_seq = rollout_collector.rollout(u_seq, clip=True)
#         x_seq = x_seq.clone().detach().cpu().numpy()
    
#     plot_trajectories(np.concat([x_seq, reference_trajectories], axis=0),
#                       title=title,
#                       ax=ax)


def plot_uniform(ax, 
                 n_samples: int,
                 title: str,
                 rollout_collector: RolloutCollector,
                 reference_trajectories: np.ndarray):
    with torch.no_grad():
        u_seq = sample_uniform(n_samples * rollout_collector.horizon, 
                               lb=rollout_collector.action_lb,
                               ub=rollout_collector.action_ub,
                               device=rollout_collector.action_lb.device)
        u_seq = u_seq.reshape(n_samples, rollout_collector.horizon, -1)
        x_seq = rollout_collector.rollout(u_seq)
        x_seq = x_seq.clone().detach().cpu().numpy()
    
    plot_trajectories(np.concat([x_seq, reference_trajectories], axis=0),
                      title=title,
                      ax=ax)


def make_plots(rollout_collector: RolloutCollector,
               u_seq: torch.Tensor,
               output_file: str = None,
               headless: bool = False,
               dpi: int = 150):
    _setup_matplotlib(headless)

    with torch.no_grad():
        max_action = rollout_collector.action_ub
        mid_action = rollout_collector.action_mid

        u_seq_max = torch.tile(max_action, (1, rollout_collector.horizon, 1))
        x_seq_max = rollout_collector.rollout(u_seq_max)
        x_seq_max = x_seq_max.clone().detach().cpu().numpy()[0]

        u_seq_mid = torch.tile(mid_action, (1, rollout_collector.horizon, 1))
        x_seq_mid = rollout_collector.rollout(u_seq_mid)
        x_seq_mid = x_seq_mid.clone().detach().cpu().numpy()[0]

        reference = np.stack([x_seq_max, x_seq_mid], axis=0)
    
    u_seq = torch.cat((u_seq, u_seq_max, u_seq_mid), dim=0)
    x_samples = rollout_collector.rollout(u_seq).cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    plot_trajectories(x_samples, ax=ax1, title="SVGD Samples")

    plot_uniform(ax2, u_seq.shape[0] - 2, "Action uniform", rollout_collector, reference)

    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file, dpi=dpi, bbox_inches="tight")

    if headless or output_file is not None:
        plt.close(fig)


def plot_trajectories_unicycle(*args, **kwargs) -> None:
    # Backward-compatible alias.
    plot_trajectories(*args, **kwargs)
