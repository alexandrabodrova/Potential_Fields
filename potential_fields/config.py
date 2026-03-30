"""Configuration dataclass for simulation parameters."""

from dataclasses import dataclass, field


@dataclass
class SimConfig:
    """All simulation hyperparameters.

    Default values are chosen to match the paper's experimental setup
    (Howard, Mataric & Sukhatme, DARS 2002).
    """

    # --- Network ---
    num_nodes: int = 100

    # --- Sensor ---
    sensor_range: float = 4.0       # meters, 360-degree laser range finder
    sensor_fov: float = 360.0       # degrees (full circle)

    # --- Potential field weights ---
    k_obstacle: float = 1.0         # obstacle repulsion strength (k_o)
    k_node: float = 1.0             # inter-node repulsion strength (k_n)

    # --- Dynamics ---
    mass: float = 1.0               # virtual mass (m)
    viscosity: float = 5.0          # viscous friction coefficient (nu)
    v_max: float = 0.5              # max velocity (m/s)
    a_max: float = 0.5              # max acceleration (m/s^2)
    v_dead_band: float = 0.001      # velocity dead-band to prevent oscillation

    # --- Simulation ---
    dt: float = 0.1                 # time step (seconds)
    total_time: float = 300.0       # total simulation time (seconds)

    # --- Obstacle sampling ---
    obstacle_sample_spacing: float = 0.1  # spacing between obstacle sample points

    # --- Metrics ---
    coverage_grid_resolution: float = 0.2  # meters per grid cell

    # --- Visualization ---
    animate: bool = True
    animation_interval: int = 50    # ms between frames
    plot_every_n_steps: int = 10    # record metrics every N steps

    @property
    def num_steps(self) -> int:
        return int(self.total_time / self.dt)
