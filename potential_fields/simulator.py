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

        # Precompute wall data for collision detection
        environment.precompute_wall_data()

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

        # Update positions with wall collision detection
        candidate = self.positions + self.velocities * cfg.dt

        self._resolve_wall_collisions(candidate)

        # Clamp to environment bounds (safety net)
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

        Uses vectorized numpy operations for performance.

        Returns:
            (num_nodes, 2) array of force vectors.
        """
        cfg = self.config
        n = cfg.num_nodes
        forces = np.zeros((n, 2))

        # --- Obstacle repulsion (Eq. 4) ---
        # F_o = k_o * sum_i (1/r_i^2) * (-r_hat_i)
        if self.obstacle_tree is not None:
            for j in range(n):
                nearby_idx = self.obstacle_tree.query_ball_point(
                    self.positions[j], cfg.sensor_range
                )
                if not nearby_idx:
                    continue
                nearby_obs = self.obstacle_points[nearby_idx]
                r_vec = nearby_obs - self.positions[j]
                r_dist_sq = np.sum(r_vec**2, axis=1, keepdims=True)
                r_dist_sq = np.maximum(r_dist_sq, 0.0025)  # min 0.05^2
                r_dist = np.sqrt(r_dist_sq)
                # F = -k_o * spacing * sum(r_hat / r^2)
                #   = -k_o * spacing * sum(r_vec / (r^3))
                forces[j] -= cfg.k_obstacle * cfg.obstacle_sample_spacing * np.sum(
                    r_vec / (r_dist_sq * r_dist), axis=0
                )

        # --- Inter-node repulsion (Eq. 6) ---
        # Fully vectorized: compute all pairwise distances at once
        # r_vec[i,j] = pos[j] - pos[i], shape (N, N, 2)
        diff = self.positions[None, :, :] - self.positions[:, None, :]  # (N, N, 2)
        dist_sq = np.sum(diff**2, axis=2)  # (N, N)

        # Mask: only consider nodes within sensor range, exclude self
        np.fill_diagonal(dist_sq, np.inf)  # exclude self
        in_range = dist_sq <= cfg.sensor_range**2  # (N, N)

        # Clamp minimum distance to avoid singularity
        dist_sq_safe = np.maximum(dist_sq, 0.0025)  # 0.05^2
        dist = np.sqrt(dist_sq_safe)  # (N, N)

        # Force: F_j = -k_n * sum_i(r_hat_ji / r_ji^2) for i in range
        #       = -k_n * sum_i(diff[j,i] / (dist[j,i]^3))
        # diff[j,i] = pos[i] - pos[j], pointing from j toward i
        # Negative sign means force pushes j AWAY from i
        inv_r3 = np.where(in_range, 1.0 / (dist_sq_safe * dist), 0.0)  # (N, N)
        forces -= cfg.k_node * np.einsum('ij,ijk->ik', inv_r3, diff)

        return forces

    def _resolve_wall_collisions(self, candidate: np.ndarray):
        """Check all nodes against all walls and resolve collisions.

        Uses vectorized numpy operations to test all node-movement segments
        against all wall segments simultaneously.

        For each node whose movement crosses a wall, the node is placed just
        before the first wall hit and its velocity component into the wall
        is removed (so it slides along the wall).
        """
        env = self.env
        if not env.walls:
            self.positions[:] = candidate
            return

        old = self.positions  # (N, 2)
        n = old.shape[0]

        # Wall data: (W, 2)
        wp1 = env._wall_p1   # (W, 2)
        wp2 = env._wall_p2   # (W, 2)
        wnorm = env._wall_normals  # (W, 2)
        w_count = wp1.shape[0]

        # Movement vectors: d1 = candidate - old, shape (N, 2)
        d1 = candidate - old  # (N, 2)

        # Wall direction vectors: d2 = wp2 - wp1, shape (W, 2)
        d2 = wp2 - wp1  # (W, 2)

        # We need cross products for all (node, wall) pairs.
        # cross(d1, d2) = d1x*d2y - d1y*d2x, shape (N, W)
        cross = d1[:, 0:1] * d2[None, :, 1] - d1[:, 1:2] * d2[None, :, 0]  # (N, W)

        # Vector from node_old to wall_p1: dp = wp1 - old, shape (N, W, 2)
        dpx = wp1[None, :, 0] - old[:, 0:1]  # (N, W)
        dpy = wp1[None, :, 1] - old[:, 1:2]  # (N, W)

        # t = cross(dp, d2) / cross(d1, d2)  — parameter along node movement
        # u = cross(dp, d1) / cross(d1, d2)  — parameter along wall segment
        parallel = np.abs(cross) < 1e-12  # (N, W)

        # Avoid division by zero for parallel segments
        safe_cross = np.where(parallel, 1.0, cross)
        t = (dpx * d2[None, :, 1] - dpy * d2[None, :, 0]) / safe_cross  # (N, W)
        u = (dpx * d1[:, 1:2] - dpy * d1[:, 0:1]) / safe_cross  # (N, W)

        # Valid intersection: 0 < t < 1 and 0 < u < 1, not parallel
        eps = 1e-9
        valid = (~parallel) & (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)

        # For invalid intersections, set t to infinity so they're never chosen
        t_masked = np.where(valid, t, np.inf)  # (N, W)

        # For each node, find earliest collision (smallest t)
        best_wall_idx = np.argmin(t_masked, axis=1)  # (N,)
        best_t = t_masked[np.arange(n), best_wall_idx]  # (N,)
        has_collision = best_t < 1.0  # (N,)

        # Update positions
        # Nodes without collision: move to candidate
        self.positions[~has_collision] = candidate[~has_collision]

        # Nodes with collision: place just before the wall and slide
        if np.any(has_collision):
            col_idx = np.where(has_collision)[0]
            safe_t = np.maximum(best_t[col_idx] - 0.02, 0.0)
            hit_pos = old[col_idx] + safe_t[:, None] * d1[col_idx]

            # Compute wall tangent direction for each collision
            col_wall_idx = best_wall_idx[col_idx]
            wall_dir = wp2[col_wall_idx] - wp1[col_wall_idx]  # (K, 2)
            wall_len = np.linalg.norm(wall_dir, axis=1, keepdims=True)
            wall_len = np.maximum(wall_len, 1e-12)
            wall_tangent = wall_dir / wall_len  # (K, 2) unit tangent

            # Project remaining movement onto wall tangent (slide)
            remaining = candidate[col_idx] - hit_pos  # (K, 2)
            slide_amount = np.sum(remaining * wall_tangent, axis=1, keepdims=True)
            slide_pos = hit_pos + slide_amount * wall_tangent

            # Check that the slide doesn't cross another wall
            # If it does, just stay at the hit position
            for ki, ci in enumerate(col_idx):
                result2 = env.find_wall_collision(hit_pos[ki], slide_pos[ki])
                if result2 is not None:
                    slide_pos[ki] = hit_pos[ki]

            self.positions[col_idx] = slide_pos

            # Project velocity onto tangent (remove normal component)
            v_along_tangent = np.sum(
                self.velocities[col_idx] * wall_tangent, axis=1, keepdims=True
            )
            self.velocities[col_idx] = v_along_tangent * wall_tangent

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
