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

    Recreates the USC hospital environment used by Howard et al. (DARS 2002).
    The figure shows a real hospital floor (SDH / USC campus) with:
    - Irregular outer boundary
    - Dense cluster of small offices on the left
    - Central lobby / open areas
    - Main east-west corridor with T-junctions
    - Right wing extending as a narrower corridor section with rooms
    - Various room sizes, internal furniture, pillars

    Scale: approx 55m x 18m.  Initial cluster at top-center (~25, 16).
    """
    w = _wall  # shorthand
    walls = []

    # =======================================================================
    # OUTER BOUNDARY  (irregular, traced from Fig. 2)
    # =======================================================================
    #
    #  The left portion is roughly 28m wide × 18m tall.
    #  A narrower right wing (x=28..55) is about 10m tall, attached at
    #  mid-height.  The bottom-left has a protruding section.
    #
    outer = [
        # Bottom edge, left to right
        (0, 0), (7, 0), (7, -3), (14, -3), (14, 0), (28, 0),
        # Right wing bottom
        (28, 4), (55, 4),
        # Right wing right & top
        (55, 14), (28, 14),
        # Back to left block top
        (28, 18), (0, 18),
        # Close
        (0, 0),
    ]
    for i in range(len(outer) - 1):
        walls.append(w(*outer[i], *outer[i + 1]))

    # =======================================================================
    # LEFT BLOCK — offices (x=0..14, y=0..18)
    # =======================================================================

    # -- Row of small rooms along the bottom (y=0..4) --
    walls.append(w(0, 4, 14, 4))           # corridor south wall at y=4
    # Room dividers
    walls += _vwall_with_door(3.5, 0, 4, 2)
    walls += _vwall_with_door(7, 0, 4, 2)
    walls += _vwall_with_door(10.5, 0, 4, 2)

    # -- Bottom protrusion rooms (x=7..14, y=-3..0) --
    walls += _vwall_with_door(10.5, -3, 0, -1.5)

    # -- Row of small rooms along the top (y=14..18) --
    walls.append(w(0, 14, 14, 14))          # corridor north wall at y=14
    walls += _vwall_with_door(3.5, 14, 18, 16)
    walls += _vwall_with_door(7, 14, 18, 16)
    walls += _vwall_with_door(10.5, 14, 18, 16)

    # -- Vertical corridor in left block (x=14, y=4..14 — east wall of
    #    left offices).  Doors let people into the central area. --
    walls += _vwall_with_door(14, 4, 9, 6.5)
    walls += _vwall_with_door(14, 9, 14, 11.5)

    # -- Interior rooms within left offices (y=4..9 and y=9..14) --
    # Horizontal divider splitting left offices into upper/lower halves
    walls += _hwall_with_door(0, 14, 9, 3.5)
    walls += _hwall_with_door(5, 14, 9, 10)

    # Small vertical partitions inside upper-left offices
    walls += _vwall_with_door(3.5, 9, 14, 11.5)
    walls += _vwall_with_door(7, 9, 14, 11.5)
    walls += _vwall_with_door(10.5, 9, 14, 11.5)

    # Small vertical partitions inside lower-left offices
    walls += _vwall_with_door(3.5, 4, 9, 6.5)
    walls += _vwall_with_door(7, 4, 9, 6.5)
    walls += _vwall_with_door(10.5, 4, 9, 6.5)

    # =======================================================================
    # CENTRAL AREA  (x=14..28, y=0..18) — larger rooms & lobby
    # =======================================================================

    # -- Main east-west corridor (y=7..11), the building's spine --
    walls += _hwall_with_door(14, 21, 7, 17.5)   # south wall of corridor
    walls += _hwall_with_door(21, 28, 7, 24.5)
    walls += _hwall_with_door(14, 21, 11, 17.5)  # north wall of corridor
    walls += _hwall_with_door(21, 28, 11, 24.5)

    # -- Rooms south of corridor (y=0..7) --
    # Two medium rooms + one large room
    walls += _vwall_with_door(18, 0, 7, 3.5)
    walls += _vwall_with_door(23, 0, 7, 3.5)

    # Internal furniture in south rooms
    walls += _rect_walls(15.5, 2, 1.0, 0.6)    # desk
    walls += _rect_walls(24.5, 1.5, 0.6, 1.2)  # cabinet

    # -- Rooms north of corridor (y=11..18) --
    # A large room (operating / conference) and a medium room
    walls += _vwall_with_door(19, 11, 18, 14.5)
    walls += _vwall_with_door(24, 11, 18, 14.5)

    # Partition inside the large north room
    walls.append(w(19, 15, 22, 15))  # partial wall, no door (counter)

    # Internal furniture in north rooms
    walls += _rect_walls(15, 15.5, 1.5, 0.8)   # table
    walls += _rect_walls(20.5, 12, 0.8, 0.8)   # pillar
    walls += _rect_walls(25.5, 15, 1.0, 1.0)   # equipment

    # =======================================================================
    # RIGHT WING  (x=28..55, y=4..14)  — corridor with rooms on both sides
    # =======================================================================

    # -- East-west corridor through the right wing (y=7..11) --
    #    This connects to the central corridor at x=28.
    #    Build corridor walls with doors every ~5m.
    rw_south_doors = [32, 37, 42, 47, 52]
    prev = 28.0
    for dx in rw_south_doors:
        left = dx - DOOR_W / 2
        right = dx + DOOR_W / 2
        if left > prev + 0.01:
            walls.append(w(prev, 7, left, 7))
        prev = right
    if prev < 55:
        walls.append(w(prev, 7, 55, 7))

    rw_north_doors = [32, 37, 42, 47, 52]
    prev = 28.0
    for dx in rw_north_doors:
        left = dx - DOOR_W / 2
        right = dx + DOOR_W / 2
        if left > prev + 0.01:
            walls.append(w(prev, 11, left, 11))
        prev = right
    if prev < 55:
        walls.append(w(prev, 11, 55, 11))

    # -- Room dividers south of corridor (y=4..7) --
    walls += _vwall_with_door(33, 4, 7, 5.5)
    walls += _vwall_with_door(38, 4, 7, 5.5)
    walls += _vwall_with_door(44, 4, 7, 5.5)
    walls += _vwall_with_door(50, 4, 7, 5.5)

    # -- Room dividers north of corridor (y=11..14) --
    walls += _vwall_with_door(33, 11, 14, 12.5)
    walls += _vwall_with_door(38, 11, 14, 12.5)
    walls += _vwall_with_door(44, 11, 14, 12.5)
    walls += _vwall_with_door(50, 11, 14, 12.5)

    # -- Internal details in right-wing rooms --
    walls += _rect_walls(29.5, 5, 0.6, 0.6)   # small obstacle
    walls += _rect_walls(40, 12, 1.0, 0.5)     # desk
    walls += _rect_walls(46, 5, 0.5, 0.8)      # cabinet
    walls += _rect_walls(52, 12, 0.8, 0.6)     # equipment

    # =======================================================================
    # JUNCTION between central area corridor and right wing
    # =======================================================================
    # At x=28 the outer boundary goes from (28,0) to (28,4) and (28,14) to
    # (28,18).  The corridor (y=7..11) passes through.  Fill the gaps:
    walls.append(w(28, 0, 28, 7))    # below corridor
    walls.append(w(28, 11, 28, 14))  # above corridor on right-wing side
    walls.append(w(28, 14, 28, 18))  # above right-wing top

    return Environment(walls, bounds=(-1, -4, 56, 19))


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
