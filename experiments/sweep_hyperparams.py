#!/usr/bin/env python3
"""Hyperparameter sweep experiment.

Sweeps over the key parameters identified in Section 5 of the paper:
- k_o (obstacle repulsion weight)
- k_n (inter-node repulsion weight)
- m (node mass)
- nu (viscosity coefficient)
- Network size (number of nodes)

Produces comparison plots and a summary CSV.

Usage:
    python experiments/sweep_hyperparams.py [--env ENV] [--time T] [--save-dir DIR]
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from potential_fields.config import SimConfig
from potential_fields.simulator import Simulator
from potential_fields.metrics import (
    compute_coverage_grid,
    compute_nearest_neighbor_separation,
    compute_metrics_over_time,
)
from potential_fields.visualizer import plot_metrics_over_time
from potential_fields.environments.presets import ENVIRONMENTS


def run_sweep_trial(env_name: str, config: SimConfig, seed: int = 42) -> dict:
    """Run a single trial and return summary metrics."""
    np.random.seed(seed)
    env = ENVIRONMENTS[env_name]()

    cx = (env.bounds[0] + env.bounds[2]) / 2
    cy = (env.bounds[1] + env.bounds[3]) / 2
    spread = min(env.width, env.height) * 0.03
    initial_positions = np.column_stack([
        np.random.normal(cx, spread, config.num_nodes),
        np.random.normal(cy, spread, config.num_nodes),
    ])

    sim = Simulator(config, env, initial_positions)
    sim.run()

    final_cov = compute_coverage_grid(sim.positions, env, config)
    final_sep = compute_nearest_neighbor_separation(sim.positions)

    return {
        'coverage_area': final_cov['coverage_area'],
        'coverage_fraction': final_cov['coverage_fraction'],
        'separation_mean': final_sep['mean'],
        'separation_std': final_sep['std'],
        'final_ke': sim.get_kinetic_energy(),
    }


def sweep_parameter(env_name: str, param_name: str, param_values: list,
                    base_config: dict, save_dir: str):
    """Sweep one parameter and plot results."""
    results = []
    for val in param_values:
        cfg_dict = base_config.copy()
        cfg_dict[param_name] = val
        config = SimConfig(**cfg_dict)

        print(f"  {param_name}={val:.4g} ...", end=' ', flush=True)
        t0 = time.time()
        result = run_sweep_trial(env_name, config)
        dt = time.time() - t0
        print(f"coverage={result['coverage_fraction']*100:.1f}%, "
              f"sep={result['separation_mean']:.2f}m ({dt:.1f}s)")

        result[param_name] = val
        results.append(result)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    vals = [r[param_name] for r in results]
    ax1.plot(vals, [r['coverage_fraction'] * 100 for r in results], 'bo-')
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Coverage (%)')
    ax1.set_title(f'Coverage vs {param_name}')
    ax1.grid(True, alpha=0.3)

    ax2.plot(vals, [r['separation_mean'] for r in results], 'ro-')
    ax2.fill_between(vals,
                     [r['separation_mean'] - r['separation_std'] for r in results],
                     [r['separation_mean'] + r['separation_std'] for r in results],
                     alpha=0.2, color='red')
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Avg NN Separation (m)')
    ax2.set_title(f'Separation vs {param_name}')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Hyperparameter Sweep: {param_name} ({env_name})')
    fig.tight_layout()
    fig.savefig(f"{save_dir}/sweep_{param_name}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    return results


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Sweep')
    parser.add_argument('--env', type=str, default='simple_room',
                        choices=list(ENVIRONMENTS.keys()))
    parser.add_argument('--time', type=float, default=120.0,
                        help='Simulation time per trial (default: 120)')
    parser.add_argument('--nodes', type=int, default=30,
                        help='Base number of nodes (default: 30)')
    parser.add_argument('--save-dir', type=str, default='results/sweeps')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    base = {
        'num_nodes': args.nodes,
        'total_time': args.time,
        'k_obstacle': 1.0,
        'k_node': 1.0,
        'viscosity': 5.0,
        'mass': 1.0,
    }

    all_results = []

    # 1. Sweep k_obstacle
    print("\n=== Sweeping k_obstacle ===")
    r = sweep_parameter(args.env, 'k_obstacle',
                        [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                        base, args.save_dir)
    all_results.extend(r)

    # 2. Sweep k_node
    print("\n=== Sweeping k_node ===")
    r = sweep_parameter(args.env, 'k_node',
                        [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                        base, args.save_dir)
    all_results.extend(r)

    # 3. Sweep viscosity
    print("\n=== Sweeping viscosity ===")
    r = sweep_parameter(args.env, 'viscosity',
                        [0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
                        base, args.save_dir)
    all_results.extend(r)

    # 4. Sweep mass
    print("\n=== Sweeping mass ===")
    r = sweep_parameter(args.env, 'mass',
                        [0.1, 0.5, 1.0, 2.0, 5.0],
                        base, args.save_dir)
    all_results.extend(r)

    # 5. Sweep network size
    print("\n=== Sweeping num_nodes ===")
    r = sweep_parameter(args.env, 'num_nodes',
                        [10, 20, 30, 50, 75, 100],
                        base, args.save_dir)
    all_results.extend(r)

    # Save CSV summary
    csv_path = f"{args.save_dir}/sweep_results.csv"
    fieldnames = sorted(set().union(*(r.keys() for r in all_results)))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    main()
