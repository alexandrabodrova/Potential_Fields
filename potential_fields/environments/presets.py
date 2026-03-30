"""Predefined environment layouts for experiments.

Environments range from simple (open field, single room) to complex
(hospital floor plan, maze) to support the paper recreation and the
extended experiments with more complicated environments.
"""

import numpy as np
from ..environment import Environment, Wall


def _rect_walls(x, y, w, h) -> list[Wall]:
    """Create walls for a rectangle: bottom-left at (x, y), size (w, h)."""
    return [
        Wall(np.array([x, y]), np.array([x + w, y])),          # bottom
        Wall(np.array([x + w, y]), np.array([x + w, y + h])),  # right
        Wall(np.array([x + w, y + h]), np.array([x, y + h])),  # top
        Wall(np.array([x, y + h]), np.array([x, y])),          # left
    ]


def _wall(x1, y1, x2, y2) -> Wall:
    return Wall(np.array([x1, y1]), np.array([x2, y2]))


# ---------------------------------------------------------------------------
# 1. Open field (no internal obstacles, just bounding walls)
# ---------------------------------------------------------------------------
def make_open_field(width: float = 20.0, height: float = 20.0) -> Environment:
    """Simple open rectangular environment with no internal obstacles."""
    walls = _rect_walls(0, 0, width, height)
    return Environment(walls, bounds=(0, 0, width, height))


# ---------------------------------------------------------------------------
# 2. Simple room with one obstacle (matches Fig. 1 from the paper)
# ---------------------------------------------------------------------------
def make_simple_room(width: float = 16.0, height: float = 12.0) -> Environment:
    """Room with a central L-shaped obstacle, similar to Fig. 1 in the paper."""
    walls = _rect_walls(0, 0, width, height)
    # L-shaped internal obstacle
    walls += [
        _wall(5, 2, 5, 7),
        _wall(5, 7, 9, 7),
        _wall(9, 7, 9, 5),
        _wall(9, 5, 7, 5),
        _wall(7, 5, 7, 2),
        _wall(7, 2, 5, 2),
    ]
    # Small rectangular obstacle in upper right
    walls += _rect_walls(11, 8, 2, 2)
    return Environment(walls, bounds=(0, 0, width, height))


# ---------------------------------------------------------------------------
# 3. Two connected rooms (doorway)
# ---------------------------------------------------------------------------
def make_two_rooms(room_w: float = 10.0, room_h: float = 10.0,
                   door_width: float = 2.0) -> Environment:
    """Two rooms connected by a doorway."""
    total_w = 2 * room_w
    walls = _rect_walls(0, 0, total_w, room_h)
    # Dividing wall with gap (door) in the middle
    door_y = (room_h - door_width) / 2
    walls += [
        _wall(room_w, 0, room_w, door_y),
        _wall(room_w, door_y + door_width, room_w, room_h),
    ]
    return Environment(walls, bounds=(0, 0, total_w, room_h))


# ---------------------------------------------------------------------------
# 4. L-shaped environment
# ---------------------------------------------------------------------------
def make_l_shaped() -> Environment:
    """L-shaped corridor environment."""
    walls = [
        _wall(0, 0, 20, 0),
        _wall(20, 0, 20, 8),
        _wall(20, 8, 8, 8),
        _wall(8, 8, 8, 16),
        _wall(8, 16, 0, 16),
        _wall(0, 16, 0, 0),
    ]
    return Environment(walls, bounds=(0, 0, 20, 16))


# ---------------------------------------------------------------------------
# 5. Hospital-like floor plan (simplified version of Fig. 2 from paper)
# ---------------------------------------------------------------------------
def make_hospital() -> Environment:
    """Simplified hospital floor plan inspired by the paper's experiments.

    A large building with multiple rooms, corridors, and doorways.
    Approximately 40m x 20m to match the scale of the paper.
    Each room has a 2m-wide doorway opening into the central corridor.
    """
    W, H = 40, 20
    walls = _rect_walls(0, 0, W, H)

    # --- Main horizontal corridor (y=8 to y=12, 4m wide) ---
    # Bottom corridor wall: gaps at each room's doorway position
    # Room door positions (center of gap): x=4, 12, 20, 28, 36
    # Each gap is 2m wide
    bottom_wall_y = 8
    top_wall_y = 12
    door_centers = [4, 12, 20, 28, 36]
    door_half = 1.0  # 2m wide doors

    # Build corridor walls with gaps for doors
    # Bottom wall of corridor
    prev_x = 0
    for dx in door_centers:
        if dx - door_half > prev_x:
            walls.append(_wall(prev_x, bottom_wall_y, dx - door_half, bottom_wall_y))
        prev_x = dx + door_half
    if prev_x < W:
        walls.append(_wall(prev_x, bottom_wall_y, W, bottom_wall_y))

    # Top wall of corridor
    prev_x = 0
    for dx in door_centers:
        if dx - door_half > prev_x:
            walls.append(_wall(prev_x, top_wall_y, dx - door_half, top_wall_y))
        prev_x = dx + door_half
    if prev_x < W:
        walls.append(_wall(prev_x, top_wall_y, W, top_wall_y))

    # --- Room dividers (vertical walls between rooms) ---
    # Bottom rooms: dividers at x=8, 16, 24, 32 from y=0 to y=8
    for x in [8, 16, 24, 32]:
        walls.append(_wall(x, 0, x, bottom_wall_y))

    # Top rooms: dividers at x=8, 16, 24, 32 from y=12 to y=20
    for x in [8, 16, 24, 32]:
        walls.append(_wall(x, top_wall_y, x, H))

    return Environment(walls, bounds=(0, 0, W, H))


# ---------------------------------------------------------------------------
# 6. Maze environment
# ---------------------------------------------------------------------------
def make_maze() -> Environment:
    """A maze-like environment with narrow corridors.

    Tests how well potential fields navigate tight spaces.
    """
    S = 20  # size
    walls = _rect_walls(0, 0, S, S)

    # Horizontal internal walls with gaps
    for y in [4, 8, 12, 16]:
        gap_x = np.random.RandomState(42 + y).randint(2, S - 4)
        walls += [
            _wall(0, y, gap_x, y),
            _wall(gap_x + 3, y, S, y),
        ]

    # A few vertical walls
    walls += [
        _wall(6, 0, 6, 3),
        _wall(14, 5, 14, 8),
        _wall(10, 9, 10, 12),
        _wall(6, 13, 6, 16),
        _wall(14, 13, 14, 16),
    ]

    return Environment(walls, bounds=(0, 0, S, S))


# ---------------------------------------------------------------------------
# Registry for easy access by name
# ---------------------------------------------------------------------------
ENVIRONMENTS = {
    'open_field': make_open_field,
    'simple_room': make_simple_room,
    'two_rooms': make_two_rooms,
    'l_shaped': make_l_shaped,
    'hospital': make_hospital,
    'maze': make_maze,
}
