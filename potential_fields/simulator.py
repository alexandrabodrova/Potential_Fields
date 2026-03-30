"""Core simulation engine implementing the potential field deployment algorithm.

Implements the approach from Howard, Mataric & Sukhatme (DARS 2002):
- Electrostatic repulsive potentials from obstacles (Eq. 2,4)
- Electrostatic repulsive potentials from other nodes (Eq. 5,6)
- Equation of motion with viscous friction (Eq. 7)
- Discrete control law with velocity/acceleration clipping (Eq. 8,9)
"""

import numpy as np
from scipy.spatial import KDTree

from .config import SimConfig
from .environment import Environment


class Simulator:
    """Potential-field-based sensor network deployment simulator."""

    def __init__(self, config: SimConfig, environment: Environment,
                 initial_positions: np.ndarray | None = None):
        """
        Args:
            config: Simulation hyperparameters.
            environment: The 2D environment with walls.
            initial_positions: (num_nodes, 2) array. If None, nodes are
                placed in a tight cluster at the environment center.
        """
        self.config = config
        self.env = environment

        # Pre-sample obstacle points for force computation
        self.obstacle_points = environment.sample_obstacle_points(
            config.obstacle_sample_spacing
        )
        # Build KD-tree for fast obstacle neighbor queries
        if len(self.obstacle_points) > 0:
            self.obstacle_tree = KDTree(self.obstacle_points)
        else:
            self.obstacle_tree = None

        # Initialize node positions
        if initial_positions is not None:
            self.positions = np.array(initial_positions, dtype=float)
        else:
            # Default: tight Gaussian cluster at center of bounds
            cx = (environment.bounds[0] + environment.bounds[2]) / 2
            cy = (environment.bounds[1] + environment.bounds[3]) / 2
            self.positions = np.column_stack([
                np.random.normal(cx, 0.5, config.num_nodes),
                np.random.normal(cy, 0.5, config.num_nodes),
            ])

        assert self.positions.shape == (config.num_nodes, 2)

        # Initialize velocities to zero
        self.velocities = np.zeros((config.num_nodes, 2))

        # Simulation clock
        self.time = 0.0
        self.step_count = 0

        # History for metrics/visualization
        self.history_positions = [self.positions.copy()]
        self.history_times = [0.0]

    def step(self):
        """Advance the simulation by one time step.

        Implements the discrete control law from Eq. 8-9 of the paper.
        """
        cfg = self.config
        n = cfg.num_nodes

        # Compute total force on each node: F = F_obstacle + F_node
        forces = self._compute_forces()

        # Equation of motion (Eq. 8): delta_v = (F - nu * v) / m * dt
        delta_v = (forces - cfg.viscosity * self.velocities) / cfg.mass * cfg.dt

        # Clip acceleration (delta_v) component-wise
        a_max_dt = cfg.a_max * cfg.dt
        delta_v = np.clip(delta_v, -a_max_dt, a_max_dt)

        # Update velocity (Eq. 9): v = v + delta_v
        self.velocities += delta_v

        # Clip velocity component-wise
        self.velocities = np.clip(self.velocities, -cfg.v_max, cfg.v_max)

        # Apply velocity dead-band to prevent oscillation
        speed = np.linalg.norm(self.velocities, axis=1)
        dead_mask = speed < cfg.v_dead_band
        self.velocities[dead_mask] = 0.0

        # Update positions
        self.positions += self.velocities * cfg.dt

        # Clamp to environment bounds
        self.positions[:, 0] = np.clip(
            self.positions[:, 0], self.env.bounds[0] + 0.05, self.env.bounds[2] - 0.05
        )
        self.positions[:, 1] = np.clip(
            self.positions[:, 1], self.env.bounds[1] + 0.05, self.env.bounds[3] - 0.05
        )

        self.time += cfg.dt
        self.step_count += 1

        # Record history periodically
        if self.step_count % cfg.plot_every_n_steps == 0:
            self.history_positions.append(self.positions.copy())
            self.history_times.append(self.time)

    def _compute_forces(self) -> np.ndarray:
        """Compute total potential field force on each node.

        Returns:
            (num_nodes, 2) array of force vectors.
        """
        cfg = self.config
        n = cfg.num_nodes
        forces = np.zeros((n, 2))

        # --- Obstacle repulsion (Eq. 4) ---
        # F_o = k_o * sum_i (1/r_i^2) * (-r_hat_i)
        # where r_i = x_obstacle - x_node, so -r_hat pushes away from obstacle
        if self.obstacle_tree is not None:
            for j in range(n):
                # Find obstacles within sensor range
                nearby_idx = self.obstacle_tree.query_ball_point(
                    self.positions[j], cfg.sensor_range
                )
                if not nearby_idx:
                    continue
                nearby_obs = self.obstacle_points[nearby_idx]
                r_vec = nearby_obs - self.positions[j]  # vectors from node to obstacles
                r_dist = np.linalg.norm(r_vec, axis=1, keepdims=True)
                r_dist = np.maximum(r_dist, 0.05)  # prevent division by zero
                r_hat = r_vec / r_dist
                # Force pushes node AWAY from obstacles (negative direction)
                # Scale by sampling spacing so force is independent of discretization
                f_obs = -cfg.k_obstacle * cfg.obstacle_sample_spacing * np.sum(
                    r_hat / r_dist**2, axis=0
                )
                forces[j] += f_obs

        # --- Inter-node repulsion (Eq. 6) ---
        # F_n = k_n * sum_i (1/r_i^2) * (-r_hat_i)
        # Use KD-tree for efficient neighbor search
        node_tree = KDTree(self.positions)
        for j in range(n):
            nearby_idx = node_tree.query_ball_point(
                self.positions[j], cfg.sensor_range
            )
            for idx in nearby_idx:
                if idx == j:
                    continue
                r_vec = self.positions[idx] - self.positions[j]  # from node j to node idx
                r_dist = np.linalg.norm(r_vec)
                if r_dist < 0.05:
                    # Jitter to break degeneracy when nodes overlap
                    r_vec = np.random.randn(2) * 0.01
                    r_dist = np.linalg.norm(r_vec)
                r_hat = r_vec / r_dist
                # Force pushes node j AWAY from node idx
                forces[j] -= cfg.k_node * r_hat / r_dist**2

        return forces

    def run(self, callback=None):
        """Run the full simulation.

        Args:
            callback: Optional function called each step with (simulator, step).
                Return False to stop early.
        """
        total_steps = self.config.num_steps
        for step_i in range(total_steps):
            self.step()
            if callback is not None:
                if callback(self, step_i) is False:
                    break

    def get_kinetic_energy(self) -> float:
        """Total kinetic energy of the system: sum(0.5 * m * v^2)."""
        speeds_sq = np.sum(self.velocities**2, axis=1)
        return 0.5 * self.config.mass * np.sum(speeds_sq)

    def is_equilibrium(self, threshold: float = 1e-4) -> bool:
        """Check if the system has reached static equilibrium."""
        return self.get_kinetic_energy() < threshold
