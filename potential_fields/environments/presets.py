"""Predefined environment layouts for experiments.

Environments range from simple (open field, single room) to complex
(hospital floor plan, maze) to support the paper recreation and the
extended experiments with more complicated environments.
"""

import numpy as np
from ..environment import Environment, Wall

# Door width used across environments (meters)
DOOR_W = 1.5


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


def _hwall_with_door(x_start, x_end, y, door_center_x, door_w=DOOR_W):
    """Horizontal wall from x_start to x_end at height y, with a door gap."""
    walls = []
    d_left = door_center_x - door_w / 2
    d_right = door_center_x + door_w / 2
    if d_left > x_start + 0.01:
        walls.append(_wall(x_start, y, d_left, y))
    if x_end > d_right + 0.01:
        walls.append(_wall(d_right, y, x_end, y))
    return walls


def _vwall_with_door(x, y_start, y_end, door_center_y, door_w=DOOR_W):
    """Vertical wall from y_start to y_end at x, with a door gap."""
    walls = []
    d_bot = door_center_y - door_w / 2
    d_top = door_center_y + door_w / 2
    if d_bot > y_start + 0.01:
        walls.append(_wall(x, y_start, x, d_bot))
    if y_end > d_top + 0.01:
        walls.append(_wall(x, d_top, x, y_end))
    return walls


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
    """Simplified hospital floor plan with rooms and a central corridor.

    A clean layout (~40x20m) with rooms on both sides of a corridor.
    Suitable for large-scale deployment experiments.
    """
    W, H = 40, 20
    walls = _rect_walls(0, 0, W, H)

    # Central corridor y=8..12
    bottom_wall_y, top_wall_y = 8, 12
    door_centers = [4, 12, 20, 28, 36]
    door_half = 1.0

    prev_x = 0
    for dx in door_centers:
        if dx - door_half > prev_x:
            walls.append(_wall(prev_x, bottom_wall_y, dx - door_half, bottom_wall_y))
        prev_x = dx + door_half
    if prev_x < W:
        walls.append(_wall(prev_x, bottom_wall_y, W, bottom_wall_y))

    prev_x = 0
    for dx in door_centers:
        if dx - door_half > prev_x:
            walls.append(_wall(prev_x, top_wall_y, dx - door_half, top_wall_y))
        prev_x = dx + door_half
    if prev_x < W:
        walls.append(_wall(prev_x, top_wall_y, W, top_wall_y))

    for x in [8, 16, 24, 32]:
        walls.append(_wall(x, 0, x, bottom_wall_y))
        walls.append(_wall(x, top_wall_y, x, H))

    return Environment(walls, bounds=(0, 0, W, H))


def make_paper_hospital() -> Environment:
    """Detailed hospital floor plan closely matching Fig. 2 of the paper.

    Recreates the USC hospital environment used by Howard et al.:
    - Irregular outer boundary (wider left section, narrow right wing)
    - Multiple rooms of various sizes along corridors
    - Several corridor intersections
    - Doorways connecting rooms to corridors
    - Scale: ~50m x 24m (matching the paper's proportions)

    The initial node cluster should be placed near (20, 20) to match
    the paper's Fig. 2(a) top-center starting position.
    """
    walls = []

    # ===================================================================
    # Outer boundary — L-shaped: wide left block + narrower right wing
    # ===================================================================
    # Left block: x=0..30, y=0..24
    # Right wing: x=30..50, y=6..24
    outer = [
        (0, 0), (30, 0), (30, 6), (50, 6), (50, 24),
        (0, 24), (0, 0),
    ]
    for i in range(len(outer) - 1):
        walls.append(_wall(*outer[i], *outer[i + 1]))

    # ===================================================================
    # Main horizontal corridor through the left block (y=10..13)
    # ===================================================================
    corr_bot, corr_top = 10, 13

    # Bottom corridor wall (y=10) with doors for lower rooms
    lower_doors_x = [3.5, 8.5, 13.5, 18.5, 23.5, 28.5]
    prev = 0.0
    for dx in lower_doors_x:
        left = dx - DOOR_W / 2
        right = dx + DOOR_W / 2
        if left > prev + 0.01:
            walls.append(_wall(prev, corr_bot, left, corr_bot))
        prev = right
    if prev < 30:
        walls.append(_wall(prev, corr_bot, 30, corr_bot))

    # Top corridor wall (y=13) with doors for upper rooms
    upper_doors_x = [3.5, 8.5, 13.5, 18.5, 23.5, 28.5]
    prev = 0.0
    for dx in upper_doors_x:
        left = dx - DOOR_W / 2
        right = dx + DOOR_W / 2
        if left > prev + 0.01:
            walls.append(_wall(prev, corr_top, left, corr_top))
        prev = right
    if prev < 30:
        walls.append(_wall(prev, corr_top, 30, corr_top))

    # ===================================================================
    # Room dividers — lower rooms (y=0..10)
    # ===================================================================
    for x in [6, 11, 16, 21, 26]:
        walls.append(_wall(x, 0, x, corr_bot))

    # ===================================================================
    # Room dividers — upper rooms (y=13..24)
    # ===================================================================
    for x in [6, 11, 16, 21, 26]:
        walls.append(_wall(x, corr_top, x, 24))

    # ===================================================================
    # Right wing corridor (y=13..16, x=30..50) connecting to main corridor
    # ===================================================================
    rw_corr_bot, rw_corr_top = 13, 16

    # Bottom wall of right-wing corridor (y=13, x=30..50)
    # Opening at x=30 connects to main corridor's top-right
    rw_lower_doors = [35, 40, 45]
    prev = 30.0
    for dx in rw_lower_doors:
        left = dx - DOOR_W / 2
        right = dx + DOOR_W / 2
        if left > prev + 0.01:
            walls.append(_wall(prev, rw_corr_bot, left, rw_corr_bot))
        prev = right
    if prev < 50:
        walls.append(_wall(prev, rw_corr_bot, 50, rw_corr_bot))

    # Top wall of right-wing corridor (y=16, x=30..50)
    rw_upper_doors = [35, 40, 45]
    prev = 30.0
    for dx in rw_upper_doors:
        left = dx - DOOR_W / 2
        right = dx + DOOR_W / 2
        if left > prev + 0.01:
            walls.append(_wall(prev, rw_corr_top, left, rw_corr_top))
        prev = right
    if prev < 50:
        walls.append(_wall(prev, rw_corr_top, 50, rw_corr_top))

    # Room dividers in right wing — lower rooms (y=6..13)
    for x in [35, 40, 45]:
        walls.append(_wall(x, 6, x, rw_corr_bot))

    # Room dividers in right wing — upper rooms (y=16..24)
    for x in [35, 40, 45]:
        walls.append(_wall(x, rw_corr_top, x, 24))

    # ===================================================================
    # Vertical corridor connecting left block's corridor to right wing
    # (x=28..30, y=10..13 is already open; extend passage y=13..16)
    # ===================================================================
    # The junction at x=30 needs an opening: remove part of left-block
    # outer wall at x=30 between y=10..16 (already handled since the
    # outer boundary goes (30,0)->(30,6) and the corridor walls stop at 30)

    # Small connecting gap wall on x=30 between y=6..10
    walls.append(_wall(30, 6, 30, corr_bot))

    # ===================================================================
    # Interior details — small obstacles in some rooms (furniture-like)
    # ===================================================================
    # A couple of small rectangular obstacles in the larger left rooms
    walls += _rect_walls(2, 4, 1.5, 1.0)   # table in bottom-left room
    walls += _rect_walls(13, 5, 1.0, 1.5)  # desk in a lower room
    walls += _rect_walls(8, 18, 1.0, 1.0)  # cabinet in upper room
    walls += _rect_walls(18, 16, 1.5, 1.0) # table in upper room

    return Environment(walls, bounds=(0, 0, 50, 24))


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
    'paper_hospital': make_paper_hospital,
    'maze': make_maze,
}
