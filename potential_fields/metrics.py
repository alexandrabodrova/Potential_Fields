"""Coverage and separation metrics for evaluating deployments.

Implements the two key metrics from the paper:
1. Coverage area — area visible to the sensor network
2. Average nearest-neighbor separation
"""

import numpy as np
from scipy.spatial import KDTree

from .config import SimConfig
from .environment import Environment


def compute_coverage_grid(positions: np.ndarray, environment: Environment,
                          config: SimConfig) -> dict:
    """Compute an occupancy grid showing network coverage.

    A grid cell is "covered" if it is within sensor range of at least
    one node AND has line-of-sight to that node. For performance, we
    skip the full line-of-sight check and use a faster approximation
    (range-only), which is reasonable for the paper's experiments.

    Args:
        positions: (N, 2) node positions.
        environment: The environment.
        config: Simulation config.

    Returns:
        Dictionary with:
        - 'grid': 2D boolean array (True = covered)
        - 'coverage_area': float, total covered area in m^2
        - 'total_free_area': float, total area of bounding box
        - 'coverage_fraction': float, coverage_area / total_free_area
        - 'x_edges', 'y_edges': grid edges for plotting
    """
    res = config.coverage_grid_resolution
    x_min, y_min, x_max, y_max = environment.bounds

    x_edges = np.arange(x_min, x_max + res, res)
    y_edges = np.arange(y_min, y_max + res, res)

    # Cell centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers)  # shape (ny, nx)

    grid_points = np.column_stack([xx.ravel(), yy.ravel()])  # (M, 2)

    # For each grid point, check if any node is within sensor range
    node_tree = KDTree(positions)
    distances, _ = node_tree.query(grid_points, k=1)
    covered_mask = distances <= config.sensor_range  # (M,)

    grid = covered_mask.reshape(xx.shape)  # (ny, nx)

    cell_area = res * res
    coverage_area = np.sum(grid) * cell_area
    total_area = (x_max - x_min) * (y_max - y_min)

    return {
        'grid': grid,
        'coverage_area': coverage_area,
        'total_free_area': total_area,
        'coverage_fraction': coverage_area / total_area if total_area > 0 else 0.0,
        'x_edges': x_edges,
        'y_edges': y_edges,
    }


def compute_nearest_neighbor_separation(positions: np.ndarray) -> dict:
    """Compute nearest-neighbor separation statistics.

    The paper reports average separation of 1.6 ± 0.4m for 100 nodes.

    Args:
        positions: (N, 2) node positions.

    Returns:
        Dictionary with 'mean', 'std', 'min', 'max', 'distances' (per-node).
    """
    if len(positions) < 2:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'distances': np.array([])}

    tree = KDTree(positions)
    distances, _ = tree.query(positions, k=2)  # k=2: self + nearest neighbor
    nn_distances = distances[:, 1]  # skip self (distance = 0)

    return {
        'mean': float(np.mean(nn_distances)),
        'std': float(np.std(nn_distances)),
        'min': float(np.min(nn_distances)),
        'max': float(np.max(nn_distances)),
        'distances': nn_distances,
    }


def compute_metrics_over_time(simulator, environment: Environment,
                              config: SimConfig) -> dict:
    """Compute coverage and separation for all recorded time steps.

    Args:
        simulator: A Simulator instance (after running).
        environment: The environment.
        config: Simulation config.

    Returns:
        Dictionary with arrays: 'times', 'coverage_areas',
        'coverage_fractions', 'separation_means', 'separation_stds'.
    """
    times = []
    coverage_areas = []
    coverage_fractions = []
    sep_means = []
    sep_stds = []

    for t, pos in zip(simulator.history_times, simulator.history_positions):
        cov = compute_coverage_grid(pos, environment, config)
        sep = compute_nearest_neighbor_separation(pos)
        times.append(t)
        coverage_areas.append(cov['coverage_area'])
        coverage_fractions.append(cov['coverage_fraction'])
        sep_means.append(sep['mean'])
        sep_stds.append(sep['std'])

    return {
        'times': np.array(times),
        'coverage_areas': np.array(coverage_areas),
        'coverage_fractions': np.array(coverage_fractions),
        'separation_means': np.array(sep_means),
        'separation_stds': np.array(sep_stds),
    }
