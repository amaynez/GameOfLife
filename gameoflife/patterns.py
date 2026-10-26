"""
Pattern library for Conway's Game of Life.
Classic patterns: oscillators, spaceships, guns, puffers, methuselahs, still lifes.
"""

from typing import Set

# ============================================================================
# STILL LIFES (period 1)
# ============================================================================

BLOCK: Set[tuple[int, int]] = {(0, 0), (1, 0), (0, 1), (1, 1)}

BEEHIVE: Set[tuple[int, int]] = {(1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (2, 2)}

LOAF: Set[tuple[int, int]] = {(1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (3, 2), (2, 3)}

BOAT: Set[tuple[int, int]] = {(0, 0), (1, 0), (0, 1), (2, 1), (1, 2)}

SHIP: Set[tuple[int, int]] = {(0, 0), (1, 0), (0, 1), (2, 1), (1, 2), (2, 2)}

TUB: Set[tuple[int, int]] = {(1, 0), (0, 1), (2, 1), (1, 2)}

# ============================================================================
# OSCILLATORS (period 2+)
# ============================================================================

BLINKER: Set[tuple[int, int]] = {(0, 0), (1, 0), (2, 0)}

TOAD: Set[tuple[int, int]] = {(1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1)}

BEACON: Set[tuple[int, int]] = {(0, 0), (1, 0), (0, 1), (3, 2), (2, 3), (3, 3)}

PENTADECATHLON: Set[tuple[int, int]] = {
    (1, 0), (2, 0), (3, 0),
    (0, 1), (4, 1),
    (1, 2), (2, 2), (3, 2),
    (0, 3), (4, 3),
    (1, 4), (2, 4), (3, 4),
    (1, 5), (2, 5), (3, 5),
    (0, 6), (4, 6),
    (1, 7), (2, 7), (3, 7),
    (0, 8), (4, 8),
    (1, 9), (2, 9), (3, 9),
}

PULSAR: Set[tuple[int, int]] = {
    # Top-left quadrant (mirrored 4 ways)
    (2, -6), (3, -6), (4, -6), (6, -6), (7, -6),
    (8, -6), 
    (0, -5), (5, -5), (7, -5), (12, -5),
    (0, -4), (5, -4), (7, -4), (12, -4),
    (2, -3), (3, -3), (4, -3), (6, -3), (7, -3), (8, -3),
    (2, -2), (3, -2), (4, -2), (6, -2), (7, -2), (8, -2),
    (2, -1), (3, -1), (4, -1), (6, -1), (7, -1), (8, -1),
}

# Full pulsar (period 3)
PULSAR_FULL: Set[tuple[int, int]] = {
    # Top-left quadrant
    (2, -6), (3, -6), (4, -6), (6, -6), (7, -6), (8, -6),
    (0, -5), (5, -5), (7, -5), (12, -5),
    (0, -4), (5, -4), (7, -4), (12, -4),
    (2, -3), (3, -3), (4, -3), (6, -3), (7, -3), (8, -3),
    (2, -2), (3, -2), (4, -2), (6, -2), (7, -2), (8, -2),
    (2, -1), (3, -1), (4, -1), (6, -1), (7, -1), (8, -1),
    (4, 0), (5, 0), (6, 0),
}

# ============================================================================
# SPACESHIPS (period 4)
# ============================================================================

GLIDER: Set[tuple[int, int]] = {(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)}

# Light-weight spaceship (LWSS)
LWSS: Set[tuple[int, int]] = {
    (1, 0), (4, 0),
    (0, 1), (4, 1),
    (0, 2), (3, 2), (4, 2),
    (1, 3), (2, 3), (3, 3),
}

# Medium-weight spaceship (MWSS)
MWSS: Set[tuple[int, int]] = {
    (2, 0), (5, 0),
    (1, 1), (5, 1),
    (0, 2), (1, 2), (2, 2), (4, 2), (5, 2),
    (0, 3), (2, 3), (3, 3), (4, 3),
    (1, 4), (2, 4), (3, 4),
}

# Heavy-weight spaceship (HWSS)
HWSS: Set[tuple[int, int]] = {
    (3, 0), (6, 0),
    (2, 1), (6, 1),
    (1, 2), (2, 2), (3, 2), (4, 2), (6, 2),
    (0, 3), (1, 3), (3, 3), (4, 3), (5, 3),
    (1, 4), (2, 4), (3, 4), (4, 4),
}

# ============================================================================
# GUNS (period N)
# ============================================================================

GLIDER_GUN: Set[tuple[int, int]] = {
    (0, 4), (0, 5),
    (10, 4), (10, 5), (10, 6),
    (11, 3), (12, 2), (13, 2),
    (11, 7), (12, 8), (13, 8),
    (14, 5),
    (15, 3), (15, 7),
    (16, 4), (16, 5), (16, 6),
    (17, 5),
    (20, 2), (20, 3), (20, 4),
    (21, 2), (21, 3), (21, 4),
    (22, 1), (22, 5),
    (24, 0), (24, 1), (24, 5), (24, 6),
    (34, 2), (34, 3), (35, 2), (35, 3),
}

# Simkin's Gun (period 90)
SIMKIN_GUN: Set[tuple[int, int]] = {
    (0, 0), (1, 0),
    (0, 1), (1, 1),
    (5, 0), (6, 0), (7, 0),
    (5, 2), (9, 2),
    (6, 1), (10, 1), (11, 1), (12, 1),
    (7, 2), (11, 2), (12, 2), (13, 2),
    (8, 3), (12, 3),
    (9, 4), (10, 4),
    (9, 5), (10, 5),
    (9, 6), (11, 6),
    (20, 2), (21, 2), (22, 2),
    (20, 3), (21, 3), (22, 3),
    (20, 4), (22, 4),
    (24, 1), (25, 1),
    (24, 2), (25, 2),
    (24, 3), (25, 3),
    (34, 2), (34, 3), (35, 2), (35, 3),
    (44, 4), (45, 4),
    (44, 5), (46, 5),
    (44, 6), (45, 6),
}

# ============================================================================
# PUFFERS & RAKES
# ============================================================================

# Rake (spaceship generator)
RAKE: Set[tuple[int, int]] = {
    (0, 0), (1, 0), (2, 0), (3, 0),
    (4, 1),
    (0, 2), (4, 2),
    (1, 3), (2, 3), (3, 3),
}

# Simple puffer
PUFFER: Set[tuple[int, int]] = {
    (0, 0), (3, 0),
    (0, 1), (4, 1),
    (1, 2), (2, 2), (3, 2),
    (2, 3),
}

# ============================================================================
# METHUSELAHS (long-lived seeds)
# ============================================================================

# R-pentomino (period 110, 248 generations to stabilize)
R_PENTOMINO: Set[tuple[int, int]] = {(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)}

# Diehard (period 7, dies after 130 generations)
DIEHARD: Set[tuple[int, int]] = {(6, 0), (0, 1), (1, 1), (1, 2), (5, 2), (6, 2), (7, 2)}

# Acorn (period 5206, dies after 6330 generations)
ACORN: Set[tuple[int, int]] = {(1, 0), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2), (3, 3)}

# Gosper's glider gun (exists as GLIDER_GUN above)

# ============================================================================
# PATTERN COLLECTIONS
# ============================================================================

ALL_PATTERNS: dict[str, Set[tuple[int, int]]] = {
    # Still lifes
    "block": BLOCK,
    "beehive": BEEHIVE,
    "loaf": LOAF,
    "boat": BOAT,
    "ship": SHIP,
    "tub": TUB,
    # Oscillators
    "blinker": BLINKER,
    "toad": TOAD,
    "beacon": BEACON,
    "pulsar": PULSAR_FULL,
    "pentadecathlon": PENTADECATHLON,
    # Spaceships
    "glider": GLIDER,
    "lwss": LWSS,
    "mwss": MWSS,
    "hwss": HWSS,
    # Guns
    "glider-gun": GLIDER_GUN,
    "simkin-gun": SIMKIN_GUN,
    # Methuselahs
    "rpentomino": R_PENTOMINO,
    "diehard": DIEHARD,
    "acorn": ACORN,
    # Puffer
    "puffer": PUFFER,
}


def get_pattern(name: str) -> Set[tuple[int, int]] | None:
    """Get a pattern by name (case-insensitive)."""
    return ALL_PATTERNS.get(name.lower())


def list_patterns() -> list[str]:
    """List all available pattern names."""
    return sorted(ALL_PATTERNS.keys())