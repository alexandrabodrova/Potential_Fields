#!/usr/bin/env python3
"""Real-time interactive simulation viewer.

Runs the potential field deployment simulation with a live matplotlib
animation that updates as the simulation progresses. Shows the nodes
moving in real-time along with live coverage and energy statistics.

Usage:
    python run_realtime.py [--env ENV] [--nodes N] [--time T] [--speed SPEED]
                           [--save FILE]

Examples:
    # Watch 100 nodes deploy in the paper's hospital (default)
    python run_realtime.py

    # Faster playback with fewer nodes
    python run_realtime.py --env simple_room --nodes 30 --time 120 --speed 5

    # Save as MP4 video (no live window)
    python run_realtime.py --save deployment.mp4

    # Save as GIF
    python run_realtime.py --env simple_room --nodes 20 --time 60 --save demo.gif

Controls (interactive mode):
    - Close the window to stop the simulation
    - The simulation runs at `speed` x real-time (default 3x)
"""

import argparse
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from potential_fields.config import SimConfig
from potential_fields.simulator import Simulator
from potential_fields.metrics import (
    compute_coverage_grid,
    compute_nearest_neighbor_separation,
)
from potential_fields.environments.presets import ENVIRONMENTS


def build_simulation(env_name, config, seed=42):
    """Set up the environment and simulator."""
    np.random.seed(seed)
    env = ENVIRONMENTS[env_name]()

    # Initial cluster position
    if env_name == 'paper_hospital':
        cx, cy = 21.0, 16.0
    elif env_name == 'hospital':
        cx, cy = 20.0, 15.0
    elif env_name == 'two_rooms':
        cx, cy = 5.0, 5.0
    else:
        cx = (env.bounds[0] + env.bounds[2]) / 2
        cy = (env.bounds[1] + env.bounds[3]) / 2

    spread = min(env.width, env.height) * 0.03
    init_pos = np.column_stack([
        np.random.normal(cx, spread, config.num_nodes),
        np.random.normal(cy, spread, config.num_nodes),
    ])

    sim = Simulator(config, env, init_pos)
    return sim, env


def create_realtime_animation(sim, env, config, speed=3, save_path=None):
    """Create and run a real-time animation of the deployment.

    Args:
        sim: Simulator instance.
        env: Environment instance.
        config: SimConfig.
        speed: Simulation speed multiplier (e.g., 3 = 3x real-time).
        save_path: If set, save animation to file instead of displaying live.
    """
    # --- Figure layout ---
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('white')

    # Main deployment view (left, large)
    ax_main = fig.add_axes([0.05, 0.08, 0.55, 0.84])
    # Coverage over time (top-right)
    ax_cov = fig.add_axes([0.67, 0.55, 0.30, 0.37])
    # Separation over time (bottom-right)
    ax_sep = fig.add_axes([0.67, 0.08, 0.30, 0.37])

    # --- Draw environment ---
    segments = [[w.p1, w.p2] for w in env.walls]
    if segments:
        lc = LineCollection(segments, colors='black', linewidths=1.5)
        ax_main.add_collection(lc)
    ax_main.set_xlim(env.bounds[0] - 0.5, env.bounds[2] + 0.5)
    ax_main.set_ylim(env.bounds[1] - 0.5, env.bounds[3] + 0.5)
    ax_main.set_aspect('equal')
    ax_main.set_xlabel('x (m)')
    ax_main.set_ylabel('y (m)')

    # Trajectory traces — one Line2D per node, very light
    n_nodes = config.num_nodes
    trajectory_lines = []
    # Store position history per node
    trajectory_history = [[] for _ in range(n_nodes)]
    # Record initial positions
    for j in range(n_nodes):
        trajectory_history[j].append(sim.positions[j].copy())
    # Create a light line for each node with slightly different hues
    cmap = matplotlib.colormaps['tab20']
    for j in range(n_nodes):
        color = cmap(j % 20)
        line, = ax_main.plot([], [], color=color, alpha=0.25, linewidth=0.5, zorder=2)
        trajectory_lines.append(line)

    # Node circles — draw as PatchCollection for accurate radius
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    node_circles = [Circle(sim.positions[j], config.node_radius)
                    for j in range(n_nodes)]
    node_collection = PatchCollection(node_circles, facecolors='dodgerblue',
                                      edgecolors='navy', linewidths=0.6,
                                      alpha=0.85, zorder=5)
    ax_main.add_collection(node_collection)

    # HUD text
    hud = ax_main.text(0.02, 0.98, '', transform=ax_main.transAxes,
                       verticalalignment='top', fontsize=10, family='monospace',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                                 edgecolor='gray', alpha=0.9))

    # --- Metric axes setup ---
    ax_cov.set_xlabel('Time (s)')
    ax_cov.set_ylabel('Coverage (m²)')
    ax_cov.set_title('Coverage Area', fontsize=10)
    ax_cov.grid(True, alpha=0.3)
    cov_line, = ax_cov.plot([], [], 'b-', linewidth=1.5)

    ax_sep.set_xlabel('Time (s)')
    ax_sep.set_ylabel('Separation (m)')
    ax_sep.set_title('Avg NN Separation', fontsize=10)
    ax_sep.grid(True, alpha=0.3)
    sep_line, = ax_sep.plot([], [], 'r-', linewidth=1.5)
    sep_fill = None

    # Metric history
    metric_times = []
    metric_coverage = []
    metric_sep_mean = []
    metric_sep_std = []

    # Steps per animation frame — target ~30 FPS visual update
    # Each frame advances steps_per_frame * dt of simulation time
    target_fps = 30
    sim_time_per_frame = speed / target_fps  # sim seconds per visual frame
    steps_per_frame = max(1, int(sim_time_per_frame / config.dt))
    # Compute metrics every N frames (~every 3s of sim time)
    metric_interval_sim = 3.0  # seconds of sim time between metric updates
    metric_every = max(1, int(metric_interval_sim / (config.dt * steps_per_frame)))

    fig.suptitle(f'Potential Field Deployment — {config.num_nodes} nodes, '
                 f'{env.__class__.__name__}', fontsize=12)

    frame_count = [0]
    printed_final = [False]

    def update(frame):
        nonlocal sep_fill

        # Advance simulation
        for _ in range(steps_per_frame):
            if sim.time >= config.total_time:
                break
            sim.step()

        # Record trajectory positions
        for j in range(n_nodes):
            trajectory_history[j].append(sim.positions[j].copy())

        # Update trajectory lines
        for j in range(n_nodes):
            pts = trajectory_history[j]
            if len(pts) >= 2:
                arr = np.array(pts)
                trajectory_lines[j].set_data(arr[:, 0], arr[:, 1])

        # Update node circle positions
        new_circles = [Circle(sim.positions[j], config.node_radius)
                       for j in range(n_nodes)]
        node_collection.set_paths(new_circles)

        # Compute KE
        ke = sim.get_kinetic_energy()

        # Periodically compute coverage metrics
        is_final = sim.time >= config.total_time
        if frame_count[0] % metric_every == 0 or is_final:
            cov = compute_coverage_grid(sim.positions, env, config)
            sep = compute_nearest_neighbor_separation(sim.positions)

            metric_times.append(sim.time)
            metric_coverage.append(cov['coverage_area'])
            metric_sep_mean.append(sep['mean'])
            metric_sep_std.append(sep['std'])

            # Update coverage plot
            cov_line.set_data(metric_times, metric_coverage)
            ax_cov.relim()
            ax_cov.autoscale_view()

            # Update separation plot
            sep_line.set_data(metric_times, metric_sep_mean)
            if sep_fill is not None:
                sep_fill.remove()
            means = np.array(metric_sep_mean)
            stds = np.array(metric_sep_std)
            sep_fill = ax_sep.fill_between(
                metric_times, means - stds, means + stds,
                alpha=0.15, color='red'
            )
            ax_sep.relim()
            ax_sep.autoscale_view()

            cov_text = f"{cov['coverage_area']:.0f}"
            sep_text = f"{sep['mean']:.2f}"
        else:
            cov_text = f"{metric_coverage[-1]:.0f}" if metric_coverage else "..."
            sep_text = f"{metric_sep_mean[-1]:.2f}" if metric_sep_mean else "..."

        # Update HUD
        hud.set_text(
            f"t = {sim.time:6.1f}s / {config.total_time:.0f}s\n"
            f"KE = {ke:.6f}\n"
            f"Coverage ≈ {cov_text} m²\n"
            f"Separation ≈ {sep_text} m"
        )

        frame_count[0] += 1

        if is_final and not printed_final[0]:
            printed_final[0] = True
            print(f"\nSimulation complete at t={sim.time:.1f}s")
            if metric_coverage:
                print(f"  Final coverage: {metric_coverage[-1]:.1f} m²")
                print(f"  Final separation: {metric_sep_mean[-1]:.2f} "
                      f"± {metric_sep_std[-1]:.2f} m")

        return node_collection, hud, cov_line, sep_line

    total_frames = int(config.total_time / (config.dt * steps_per_frame)) + 10

    if save_path:
        # Non-interactive: render and save
        print(f"Rendering animation ({total_frames} frames)...")
        interval = max(16, int(1000 * config.dt * steps_per_frame / speed))
        anim = FuncAnimation(fig, update, frames=total_frames,
                             interval=interval, blit=False, repeat=False)
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=20, dpi=100)
        else:
            anim.save(save_path, writer='ffmpeg', fps=30, dpi=100)
        print(f"Saved to {save_path}")
        plt.close(fig)
    else:
        # Interactive: show live window
        # Interval in ms — aim for ~30 FPS visual update
        interval = max(16, int(1000 * config.dt * steps_per_frame / speed))
        anim = FuncAnimation(fig, update, frames=total_frames,
                             interval=interval, blit=False, repeat=False)
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Real-time Potential Field Deployment Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--env', type=str, default='paper_hospital',
                        choices=list(ENVIRONMENTS.keys()),
                        help='Environment (default: paper_hospital)')
    parser.add_argument('--nodes', type=int, default=100,
                        help='Number of nodes (default: 100)')
    parser.add_argument('--time', type=float, default=300.0,
                        help='Simulation time in seconds (default: 300)')
    parser.add_argument('--speed', type=float, default=3.0,
                        help='Playback speed multiplier (default: 3x)')
    parser.add_argument('--k-obstacle', type=float, default=1.0)
    parser.add_argument('--k-node', type=float, default=1.0)
    parser.add_argument('--viscosity', type=float, default=5.0)
    parser.add_argument('--mass', type=float, default=1.0)
    parser.add_argument('--save', type=str, default=None,
                        help='Save animation to file (e.g., deploy.mp4 or deploy.gif)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = SimConfig(
        num_nodes=args.nodes,
        total_time=args.time,
        k_obstacle=args.k_obstacle,
        k_node=args.k_node,
        viscosity=args.viscosity,
        mass=args.mass,
        plot_every_n_steps=1,  # record every step for smooth animation
    )

    # Use interactive backend unless saving
    if args.save:
        matplotlib.use('Agg')
    else:
        try:
            matplotlib.use('TkAgg')
        except ImportError:
            try:
                matplotlib.use('Qt5Agg')
            except ImportError:
                print("No interactive matplotlib backend found (TkAgg/Qt5Agg).")
                print("Use --save deploy.gif to save an animation file instead.")
                sys.exit(1)

    sim, env = build_simulation(args.env, config, args.seed)

    print(f"Starting real-time simulation: {args.env}, {args.nodes} nodes, "
          f"{args.time}s @ {args.speed}x speed")
    if args.save:
        print(f"Saving to: {args.save}")

    create_realtime_animation(sim, env, config, speed=args.speed,
                              save_path=args.save)


if __name__ == '__main__':
    main()
