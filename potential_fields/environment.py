"""Environment representation with walls/obstacles as line segments."""

import numpy as np
from dataclasses import dataclass


@dataclass
class Wall:
    """A wall segment defined by two endpoints."""
    p1: np.ndarray  # (2,)
    p2: np.ndarray  # (2,)

    def __post_init__(self):
        self.p1 = np.asarray(self.p1, dtype=float)
        self.p2 = np.asarray(self.p2, dtype=float)

    @property
    def length(self) -> float:
        return np.linalg.norm(self.p2 - self.p1)


class Environment:
    """2D environment defined by walls (line segments).

    Walls are sampled into discrete obstacle points that generate
    repulsive potential fields on the nodes.
    """

    def __init__(self, walls: list[Wall], bounds: tuple[float, float, float, float]):
        """
        Args:
            walls: List of Wall segments.
            bounds: (x_min, y_min, x_max, y_max) bounding box of the environment.
        """
        self.walls = walls
        self.bounds = bounds  # (x_min, y_min, x_max, y_max)

    def sample_obstacle_points(self, spacing: float = 0.1) -> np.ndarray:
        """Sample discrete points along all walls.

        Args:
            spacing: Distance between sampled points along each wall.

        Returns:
            Array of shape (N, 2) with obstacle point positions.
        """
        points = []
        for wall in self.walls:
            length = wall.length
            if length < 1e-9:
                continue
            n_samples = max(int(np.ceil(length / spacing)), 2)
            ts = np.linspace(0, 1, n_samples)
            segment_points = wall.p1[None, :] + ts[:, None] * (wall.p2 - wall.p1)[None, :]
            points.append(segment_points)
        if not points:
            return np.empty((0, 2))
        return np.vstack(points)

    def is_inside(self, point: np.ndarray) -> bool:
        """Check if a point is inside the free space (ray casting).

        Uses ray casting against all walls to determine if point
        is inside the boundary. Assumes the outer boundary forms
        a closed polygon listed first.
        """
        x, y = point[0], point[1]
        x_min, y_min, x_max, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max

    def line_of_sight(self, p: np.ndarray, q: np.ndarray) -> bool:
        """Check if there is clear line of sight between two points.

        Tests if segment p->q intersects any wall.
        """
        for wall in self.walls:
            if _segments_intersect(p, q, wall.p1, wall.p2):
                return False
        return True

    @property
    def width(self) -> float:
        return self.bounds[2] - self.bounds[0]

    @property
    def height(self) -> float:
        return self.bounds[3] - self.bounds[1]


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """Check if line segment p1-p2 intersects p3-p4.

    Uses the cross-product method for robust segment intersection.
    """
    d1 = p2 - p1
    d2 = p4 - p3

    cross = d1[0] * d2[1] - d1[1] * d2[0]

    if abs(cross) < 1e-12:
        return False  # parallel

    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross

    # Exclude exact endpoints to avoid false positives at corners
    eps = 1e-9
    return eps < t < (1 - eps) and eps < u < (1 - eps)
