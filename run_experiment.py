#!/usr/bin/env python3
"""Main experiment runner — recreates the paper's deployment experiment.

Usage:
    python run_experiment.py [--env ENV_NAME] [--nodes N] [--time T]
                             [--no-animate] [--save-dir DIR]

Examples:
    # Basic paper recreation (100 nodes, hospital, 300s)
    python run_experiment.py

    # Quick test with fewer nodes
    python run_experiment.py --env simple_room --nodes 20 --time 60

    # All environments comparison
    python run_experiment.py --env all --nodes 50 --time 120
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

from potential_fields.config import SimConfig
from potential_fields.simulator import Simulator
from potential_fields.environment import Environment
from potential_fields.metrics import (
    compute_coverage_grid,
    compute_nearest_neighbor_separation,
    compute_metrics_over_time,
)
from potential_fields.visualizer import (
    plot_snapshot,
    plot_coverage_grid,
    plot_metrics_over_time,
    plot_deployment_comparison,
)
from potential_fields.environments.presets import ENVIRONMENTS


def run_single_experiment(env_name: str, config: SimConfig,
                          save_dir: str, seed: int = 42):
    """Run a single deployment experiment and save results."""
    np.random.seed(seed)

    # Create environment
    env_factory = ENVIRONMENTS[env_name]
    env = env_factory()

    # Place nodes in a tight cluster (top-center for hospital, center otherwise)
    if env_name == 'hospital':
        cx, cy = 20.0, 15.0  # upper area of hospital
    elif env_name == 'two_rooms':
        cx, cy = 5.0, 5.0    # left room
    else:
        cx = (env.bounds[0] + env.bounds[2]) / 2
        cy = (env.bounds[1] + env.bounds[3]) / 2

    spread = min(env.width, env.height) * 0.03
    initial_positions = np.column_stack([
        np.random.normal(cx, spread, config.num_nodes),
        np.random.normal(cy, spread, config.num_nodes),
    ])

    # Create and run simulator
    print(f"\n{'='*60}")
    print(f"Running: {env_name} | {config.num_nodes} nodes | {config.total_time}s")
    print(f"  k_o={config.k_obstacle}, k_n={config.k_node}, "
          f"nu={config.viscosity}, m={config.mass}")
    print(f"{'='*60}")

    sim = Simulator(config, env, initial_positions)
    initial_pos = sim.positions.copy()

    t_start = time.time()
    total_steps = config.num_steps

    def progress_callback(sim, step_i):
        if (step_i + 1) % (total_steps // 10) == 0:
            pct = 100 * (step_i + 1) / total_steps
            ke = sim.get_kinetic_energy()
            print(f"  [{pct:5.1f}%] t={sim.time:.1f}s  KE={ke:.6f}")

    sim.run(callback=progress_callback)
    elapsed = time.time() - t_start
    print(f"  Simulation completed in {elapsed:.1f}s wall-clock time")

    # Compute final metrics
    final_cov = compute_coverage_grid(sim.positions, env, config)
    final_sep = compute_nearest_neighbor_separation(sim.positions)
    print(f"  Final coverage: {final_cov['coverage_area']:.1f} m² "
          f"({final_cov['coverage_fraction']*100:.1f}%)")
    print(f"  Final avg separation: {final_sep['mean']:.2f} ± {final_sep['std']:.2f} m")

    # Compute metrics over time
    print("  Computing metrics over time...")
    metrics = compute_metrics_over_time(sim, env, config)

    # Save plots
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"{save_dir}/{env_name}"

    # Fig 2-style comparison
    fig, _ = plot_deployment_comparison(
        initial_pos, sim.positions, env, final_cov, config.sensor_range
    )
    fig.savefig(f"{prefix}_deployment.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Fig 3-style metrics plot
    fig, _ = plot_metrics_over_time(metrics)
    fig.savefig(f"{prefix}_metrics.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Standalone coverage grid
    fig, _ = plot_coverage_grid(final_cov, env)
    fig.savefig(f"{prefix}_coverage.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Plots saved to {save_dir}/")

    return {
        'env_name': env_name,
        'config': config,
        'final_coverage_area': final_cov['coverage_area'],
        'final_coverage_fraction': final_cov['coverage_fraction'],
        'final_separation_mean': final_sep['mean'],
        'final_separation_std': final_sep['std'],
        'metrics': metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='Potential Field Deployment Experiment')
    parser.add_argument('--env', type=str, default='hospital',
                        choices=list(ENVIRONMENTS.keys()) + ['all'],
                        help='Environment name (default: hospital)')
    parser.add_argument('--nodes', type=int, default=100,
                        help='Number of nodes (default: 100)')
    parser.add_argument('--time', type=float, default=300.0,
                        help='Simulation time in seconds (default: 300)')
    parser.add_argument('--k-obstacle', type=float, default=1.0,
                        help='Obstacle repulsion weight k_o (default: 1.0)')
    parser.add_argument('--k-node', type=float, default=1.0,
                        help='Node repulsion weight k_n (default: 1.0)')
    parser.add_argument('--viscosity', type=float, default=5.0,
                        help='Viscosity coefficient nu (default: 5.0)')
    parser.add_argument('--mass', type=float, default=1.0,
                        help='Node mass m (default: 1.0)')
    parser.add_argument('--save-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    config = SimConfig(
        num_nodes=args.nodes,
        total_time=args.time,
        k_obstacle=args.k_obstacle,
        k_node=args.k_node,
        viscosity=args.viscosity,
        mass=args.mass,
    )

    if args.env == 'all':
        results = []
        for env_name in ENVIRONMENTS:
            result = run_single_experiment(env_name, config, args.save_dir, args.seed)
            results.append(result)

        # Summary table
        print(f"\n{'='*70}")
        print(f"{'Environment':<15} {'Coverage (m²)':>14} {'Coverage %':>11} "
              f"{'Separation':>12}")
        print(f"{'-'*70}")
        for r in results:
            print(f"{r['env_name']:<15} {r['final_coverage_area']:>14.1f} "
                  f"{r['final_coverage_fraction']*100:>10.1f}% "
                  f"{r['final_separation_mean']:>8.2f} ± {r['final_separation_std']:.2f}")
        print(f"{'='*70}")
    else:
        run_single_experiment(args.env, config, args.save_dir, args.seed)


if __name__ == '__main__':
    main()
