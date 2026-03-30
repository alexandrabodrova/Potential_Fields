"""Visualization tools for the potential field deployment simulation.

Provides:
- Static snapshots of node positions in the environment
- Coverage grid overlays
- Time-series plots of coverage and separation (matching Fig. 3 from paper)
- Animation of the deployment process
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

from .environment import Environment
from .config import SimConfig


def plot_environment(ax, environment: Environment, color='black', linewidth=1.5):
    """Draw environment walls on a matplotlib axis."""
    segments = []
    for wall in environment.walls:
        segments.append([wall.p1, wall.p2])
    if segments:
        lc = LineCollection(segments, colors=color, linewidths=linewidth)
        ax.add_collection(lc)
    ax.set_xlim(environment.bounds[0] - 0.5, environment.bounds[2] + 0.5)
    ax.set_ylim(environment.bounds[1] - 0.5, environment.bounds[3] + 0.5)
    ax.set_aspect('equal')


def plot_snapshot(positions: np.ndarray, environment: Environment,
                  title: str = "", sensor_range: float = None,
                  node_radius: float = 0.0,
                  ax=None, show_range: bool = False, figsize=(10, 8)):
    """Plot a snapshot of node positions in the environment.

    Recreates the style of Fig. 2(a,b) from the paper.
    If node_radius > 0, draws robots as filled circles with that radius.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    plot_environment(ax, environment)

    if node_radius > 0:
        for pos in positions:
            body = plt.Circle(pos, node_radius, facecolor='dodgerblue',
                              edgecolor='navy', linewidth=0.6, alpha=0.85,
                              zorder=5)
            ax.add_patch(body)
    else:
        ax.scatter(positions[:, 0], positions[:, 1], s=20, c='blue',
                   edgecolors='darkblue', linewidths=0.5, zorder=5)

    if show_range and sensor_range is not None:
        for pos in positions:
            circle = plt.Circle(pos, sensor_range, fill=False,
                                color='lightblue', alpha=0.2, linewidth=0.3)
            ax.add_patch(circle)

    ax.set_title(title)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    return fig, ax


def plot_coverage_grid(coverage_result: dict, environment: Environment,
                       ax=None, figsize=(10, 8)):
    """Plot the occupancy/coverage grid, matching Fig. 2(c) from the paper.

    Covered space = white, uncovered = gray, walls = black.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    grid = coverage_result['grid']
    x_edges = coverage_result['x_edges']
    y_edges = coverage_result['y_edges']

    # Gray = uncovered, white = covered
    display = np.where(grid, 1.0, 0.6)
    ax.pcolormesh(x_edges, y_edges, display, cmap='gray', vmin=0, vmax=1)
    plot_environment(ax, environment, color='black', linewidth=2)

    ax.set_title(f"Coverage Grid (area = {coverage_result['coverage_area']:.1f} m²)")
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    return fig, ax


def plot_metrics_over_time(metrics: dict, figsize=(10, 5)):
    """Plot coverage area and node separation vs time.

    Recreates Fig. 3 from the paper with dual y-axes.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    times = metrics['times']

    # Coverage area on left axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Elapsed time (sec)')
    ax1.set_ylabel('Coverage (sq. m)', color=color1)
    ax1.plot(times, metrics['coverage_areas'], color=color1, label='Network coverage area')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Separation on right axis
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Separation (m)', color=color2)
    ax2.plot(times, metrics['separation_means'], color=color2, linestyle='--',
             label='Node separation')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.set_title('Network coverage area and average node separation vs time')
    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_deployment_comparison(positions_initial: np.ndarray,
                               positions_final: np.ndarray,
                               environment: Environment,
                               coverage_result: dict = None,
                               sensor_range: float = None,
                               node_radius: float = 0.0,
                               figsize=(18, 5)):
    """Side-by-side comparison matching Fig. 2 from the paper.

    Shows (a) initial config, (b) final config, (c) coverage grid.
    """
    n_panels = 3 if coverage_result is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    plot_snapshot(positions_initial, environment,
                  title='(a) Initial configuration', ax=axes[0],
                  node_radius=node_radius)
    plot_snapshot(positions_final, environment,
                  title='(b) Final configuration', ax=axes[1],
                  node_radius=node_radius)

    if coverage_result is not None:
        plot_coverage_grid(coverage_result, environment, ax=axes[2])
        axes[2].set_title('(c) Coverage grid')

    fig.tight_layout()
    return fig, axes


def create_animation(simulator, environment: Environment, config: SimConfig,
                     interval: int = 50, save_path: str = None):
    """Create a matplotlib animation of the deployment process.

    Args:
        simulator: Simulator instance (already run, with history).
        environment: The environment.
        config: Simulation config.
        interval: Milliseconds between frames.
        save_path: If provided, save animation to this file (e.g., 'deploy.mp4').

    Returns:
        matplotlib FuncAnimation object.
    """
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_environment(ax, environment)

    scatter = ax.scatter([], [], s=20, c='blue', edgecolors='darkblue',
                         linewidths=0.5, zorder=5)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scatter, time_text

    def update(frame):
        positions = simulator.history_positions[frame]
        scatter.set_offsets(positions)
        t = simulator.history_times[frame]
        time_text.set_text(f't = {t:.1f}s')
        return scatter, time_text

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(simulator.history_positions),
                         interval=interval, blit=True)

    if save_path:
        anim.save(save_path, writer='ffmpeg', dpi=100)

    return anim
