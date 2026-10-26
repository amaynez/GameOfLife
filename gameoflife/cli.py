from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from gameoflife.patterns import ALL_PATTERNS, get_pattern
from gameoflife.video import export_video, capture_simulation

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency at runtime
    njit = None
    NUMBA_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None
    F = None
    TORCH_AVAILABLE = False


GOSPER_GLIDER_GUN = {
    (0, 4), (0, 5), (1, 4), (1, 5),
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

LARGERLIFE_PRESETS: dict[str, str] = {
    # Bosco-tuned preset for this implementation (robust activity with center-excluded neighbor sums).
    "bosco": "R5,B30-44,S28-52",
    # Tighter growth/survival windows for more filamentary structures.
    "coral": "R3,B9-14,S6-9",
    # Slower moving fronts with wider radius.
    "nova": "R6,B40-55,S38-52",
    # Noisy aggregate behavior.
    "storm": "R4,B18-28,S16-24",
}


class LifeEngine:
    name = "base"
    display_mode = "scatter"

    def __init__(self, width: int, height: int, wrap: bool = False) -> None:
        self.width = width
        self.height = height
        self.wrap = wrap

    def clear(self) -> None:
        raise NotImplementedError

    def randomize(self, density: float, seed: int) -> None:
        raise NotImplementedError

    def seed_pattern(self, pattern: Iterable[tuple[int, int]], anchor_x: int = 0, anchor_y: int = 0) -> None:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def advance(self, generations: int) -> int:
        generations = max(0, int(generations))
        advanced = 0
        while advanced < generations:
            delta = self.step()
            if not isinstance(delta, int) or delta <= 0:
                delta = 1
            advanced += delta
        return advanced

    def alive_count(self) -> int:
        raise NotImplementedError

    def alive_points(self) -> np.ndarray:
        raise NotImplementedError

    def board_view(self) -> np.ndarray:
        raise NotImplementedError


def sample_live_coordinates(width: int, height: int, density: float, seed: int) -> np.ndarray:
    density = max(0.0, min(1.0, density))
    total = width * height
    count = min(total, max(0, math.floor(total * density)))
    rng = np.random.default_rng(seed)
    flat_indices = rng.choice(total, size=count, replace=False)
    ys = flat_indices // width
    xs = flat_indices % width
    return np.column_stack((xs, ys))


def advance_engine(engine: LifeEngine) -> int:
    delta = engine.step()
    if isinstance(delta, int) and delta > 0:
        return delta
    return 1


def next_power_of_two(n: int) -> int:
    n = max(1, n)
    return 1 << (n - 1).bit_length()


def parse_generations_rule(rule: str | None) -> tuple[set[int], set[int], int]:
    # Default to Brian's Brain.
    if not rule:
        return {2}, set(), 3
    m = re.fullmatch(r"B([0-8]*)/S([0-8]*)/C([2-9][0-9]*)", rule.strip(), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid generations rule: {rule}. Expected format like B2/S/C3.")
    birth = {int(ch) for ch in m.group(1)} if m.group(1) else set()
    survive = {int(ch) for ch in m.group(2)} if m.group(2) else set()
    states = int(m.group(3))
    return birth, survive, states


def parse_largerlife_rule(rule: str | None) -> tuple[int, tuple[int, int], tuple[int, int]]:
    # Default to Bosco-like behavior.
    if not rule:
        rule = LARGERLIFE_PRESETS["bosco"]
    m = re.fullmatch(r"R(\d+),B(\d+)-(\d+),S(\d+)-(\d+)", rule.strip(), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid largerlife rule: {rule}. Expected format like R2,B34-45,S34-58.")
    radius = int(m.group(1))
    bmin, bmax = int(m.group(2)), int(m.group(3))
    smin, smax = int(m.group(4)), int(m.group(5))
    max_neighbors = (2 * radius + 1) ** 2 - 1
    if (
        radius < 1
        or bmin > bmax
        or smin > smax
        or bmin < 0
        or smin < 0
        or bmax > max_neighbors
        or smax > max_neighbors
    ):
        raise ValueError(f"Invalid largerlife rule values: {rule}.")
    return radius, (bmin, bmax), (smin, smax)


def parse_bs_rule(rule: str | None, max_neighbors: int, default: tuple[set[int], set[int]]) -> tuple[set[int], set[int]]:
    if not rule:
        return default
    m = re.fullmatch(r"B([0-9]*)/S([0-9]*)", rule.strip(), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid B/S rule: {rule}. Expected format like B3/S23.")
    birth = {int(ch) for ch in m.group(1)} if m.group(1) else set()
    survive = {int(ch) for ch in m.group(2)} if m.group(2) else set()
    if any(v < 0 or v > max_neighbors for v in birth | survive):
        raise ValueError(f"Rule values out of range 0..{max_neighbors}: {rule}.")
    return birth, survive


def parse_jvn_rule(rule: str | None) -> tuple[int, set[int], set[int]]:
    # Accept B../S.. (R defaults to 1) or Rn,B../S..
    if not rule:
        return 1, {2}, {1, 2}
    rrule = rule.strip()
    m = re.fullmatch(r"R(\d+),\s*B([0-9]*)/S([0-9]*)", rrule, flags=re.IGNORECASE)
    if m:
        radius = int(m.group(1))
        birth = {int(ch) for ch in m.group(2)} if m.group(2) else set()
        survive = {int(ch) for ch in m.group(3)} if m.group(3) else set()
        max_neighbors = 2 * radius * (radius + 1)
        if any(v < 0 or v > max_neighbors for v in birth | survive):
            raise ValueError(f"jvn rule values out of range 0..{max_neighbors}: {rule}.")
        return radius, birth, survive
    birth, survive = parse_bs_rule(rrule, max_neighbors=4, default=({2}, {1, 2}))
    return 1, birth, survive


def _neighbor_sum_radius(board: np.ndarray, radius: int, wrap: bool) -> np.ndarray:
    if radius == 1:
        if wrap:
            return (
                np.roll(np.roll(board, 1, axis=0), 1, axis=1)
                + np.roll(board, 1, axis=0)
                + np.roll(np.roll(board, 1, axis=0), -1, axis=1)
                + np.roll(board, 1, axis=1)
                + np.roll(board, -1, axis=1)
                + np.roll(np.roll(board, -1, axis=0), 1, axis=1)
                + np.roll(board, -1, axis=0)
                + np.roll(np.roll(board, -1, axis=0), -1, axis=1)
            )
        p = np.pad(board, 1, mode="constant")
        return (
            p[:-2, :-2]
            + p[:-2, 1:-1]
            + p[:-2, 2:]
            + p[1:-1, :-2]
            + p[1:-1, 2:]
            + p[2:, :-2]
            + p[2:, 1:-1]
            + p[2:, 2:]
        )

    if wrap:
        total = np.zeros_like(board, dtype=np.int32)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                total += np.roll(np.roll(board, dy, axis=0), dx, axis=1)
        return total

    k = 2 * radius + 1
    p = np.pad(board, radius, mode="constant").astype(np.int32)
    integral = np.pad(p, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    window = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    return window - board.astype(np.int32)


def _neighbor_sum_von_neumann(board: np.ndarray, wrap: bool) -> np.ndarray:
    if wrap:
        return (
            np.roll(board, 1, axis=0)
            + np.roll(board, -1, axis=0)
            + np.roll(board, 1, axis=1)
            + np.roll(board, -1, axis=1)
        )
    p = np.pad(board, 1, mode="constant")
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]


def _neighbor_sum_von_neumann_radius(board: np.ndarray, radius: int, wrap: bool) -> np.ndarray:
    radius = max(1, int(radius))
    if radius == 1:
        return _neighbor_sum_von_neumann(board, wrap=wrap)
    total = np.zeros_like(board, dtype=np.int32)
    for dy in range(-radius, radius + 1):
        max_dx = radius - abs(dy)
        for dx in range(-max_dx, max_dx + 1):
            if dx == 0 and dy == 0:
                continue
            if wrap:
                total += np.roll(np.roll(board, dy, axis=0), dx, axis=1)
            else:
                y0_src = max(0, -dy)
                y1_src = min(board.shape[0], board.shape[0] - dy)
                x0_src = max(0, -dx)
                x1_src = min(board.shape[1], board.shape[1] - dx)
                y0_dst = y0_src + dy
                y1_dst = y1_src + dy
                x0_dst = x0_src + dx
                x1_dst = x1_src + dx
                total[y0_dst:y1_dst, x0_dst:x1_dst] += board[y0_src:y1_src, x0_src:x1_src]
    return total


def load_rule_spec(rule_file: str) -> dict:
    with open(rule_file, "r", encoding="utf-8") as f:
        raw = f.read()
    stripped = raw.lstrip()
    if stripped.startswith("{"):
        return json.loads(raw)

    # Minimal .rule compatibility parser.
    spec: dict[str, object] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("@"):
            continue
        if "=" in line:
            key, value = [part.strip() for part in line.split("=", 1)]
        elif ":" in line:
            key, value = [part.strip() for part in line.split(":", 1)]
        else:
            continue
        kl = key.lower().replace(" ", "_")
        vl = value.strip()
        if kl in ("rule", "rulestring"):
            if "/C" in vl.upper():
                b, s, c = parse_generations_rule(vl)
                spec["birth"] = sorted(b)
                spec["survive"] = sorted(s)
                spec["states"] = c
            elif vl.upper().startswith("R"):
                r, br, sr = parse_largerlife_rule(vl)
                spec["radius"] = r
                spec["birth"] = list(range(br[0], br[1] + 1))
                spec["survive"] = list(range(sr[0], sr[1] + 1))
            else:
                b, s = parse_bs_rule(vl, 8, ({3}, {2, 3}))
                spec["birth"] = sorted(b)
                spec["survive"] = sorted(s)
        elif kl in ("num_states", "states"):
            spec["states"] = int(vl)
        elif kl in ("neighborhood",):
            n = vl.lower().replace("-", "")
            if n in ("vonneumann", "vn"):
                n = "vonneumann"
            spec["neighborhood"] = n
        elif kl in ("radius",):
            spec["radius"] = int(vl)
        elif kl in ("birth", "survive"):
            nums = [int(v) for v in re.findall(r"\d+", vl)]
            spec[kl] = nums
    return spec


@dataclass
class SparseLifeEngine(LifeEngine):
    width: int
    height: int
    wrap: bool = False
    alive_cells: set[tuple[int, int]] = field(default_factory=set)

    name = "sparse"
    display_mode = "scatter"

    def clear(self) -> None:
        self.alive_cells.clear()

    def randomize(self, density: float, seed: int) -> None:
        self.clear()
        points = sample_live_coordinates(self.width, self.height, density, seed)
        self.alive_cells = {(int(x), int(y)) for x, y in points}

    def seed_pattern(self, pattern: Iterable[tuple[int, int]], anchor_x: int = 0, anchor_y: int = 0) -> None:
        self.clear()
        for dx, dy in pattern:
            x = anchor_x + dx
            y = anchor_y + dy
            if self.wrap:
                self.alive_cells.add((x % self.width, y % self.height))
            elif 0 <= x < self.width and 0 <= y < self.height:
                self.alive_cells.add((x, y))

    def _neighbors(self, x: int, y: int) -> Iterable[tuple[int, int]]:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if self.wrap:
                    yield nx % self.width, ny % self.height
                elif 0 <= nx < self.width and 0 <= ny < self.height:
                    yield nx, ny

    def step(self) -> None:
        neighbor_counts: Counter[tuple[int, int]] = Counter()
        for x, y in self.alive_cells:
            for neighbor in self._neighbors(x, y):
                neighbor_counts[neighbor] += 1

        self.alive_cells = {
            cell
            for cell, count in neighbor_counts.items()
            if count == 3 or (count == 2 and cell in self.alive_cells)
        }

    def alive_count(self) -> int:
        return len(self.alive_cells)

    def alive_points(self) -> np.ndarray:
        if not self.alive_cells:
            return np.empty((0, 2), dtype=float)
        return np.array(list(self.alive_cells), dtype=float)

    def board_view(self) -> np.ndarray:
        board = np.zeros((self.height, self.width), dtype=np.uint8)
        if self.alive_cells:
            points = np.array(list(self.alive_cells), dtype=int)
            board[points[:, 1], points[:, 0]] = 1
        return board


@dataclass
class DenseVectorizedEngine(LifeEngine):
    width: int
    height: int
    wrap: bool = False
    board: np.ndarray = field(init=False)

    name = "dense"
    display_mode = "image"

    def __post_init__(self) -> None:
        self.board = np.zeros((self.height, self.width), dtype=np.uint8)

    def clear(self) -> None:
        self.board.fill(0)

    def randomize(self, density: float, seed: int) -> None:
        self.clear()
        points = sample_live_coordinates(self.width, self.height, density, seed)
        if points.size:
            self.board[points[:, 1], points[:, 0]] = 1

    def seed_pattern(self, pattern: Iterable[tuple[int, int]], anchor_x: int = 0, anchor_y: int = 0) -> None:
        self.clear()
        for dx, dy in pattern:
            x = anchor_x + dx
            y = anchor_y + dy
            if self.wrap:
                self.board[y % self.height, x % self.width] = 1
            elif 0 <= x < self.width and 0 <= y < self.height:
                self.board[y, x] = 1

    def step(self) -> None:
        b = self.board
        if self.wrap:
            neighbors = (
                np.roll(np.roll(b, 1, axis=0), 1, axis=1)
                + np.roll(b, 1, axis=0)
                + np.roll(np.roll(b, 1, axis=0), -1, axis=1)
                + np.roll(b, 1, axis=1)
                + np.roll(b, -1, axis=1)
                + np.roll(np.roll(b, -1, axis=0), 1, axis=1)
                + np.roll(b, -1, axis=0)
                + np.roll(np.roll(b, -1, axis=0), -1, axis=1)
            )
        else:
            p = np.pad(b, 1, mode="constant")
            neighbors = (
                p[:-2, :-2]
                + p[:-2, 1:-1]
                + p[:-2, 2:]
                + p[1:-1, :-2]
                + p[1:-1, 2:]
                + p[2:, :-2]
                + p[2:, 1:-1]
                + p[2:, 2:]
            )

        self.board = np.where((neighbors == 3) | ((b == 1) & (neighbors == 2)), 1, 0).astype(np.uint8)

    def alive_count(self) -> int:
        return int(self.board.sum())

    def alive_points(self) -> np.ndarray:
        points = np.argwhere(self.board == 1)
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        return np.column_stack((points[:, 1], points[:, 0])).astype(float)

    def board_view(self) -> np.ndarray:
        return self.board


@dataclass
class GenerationsEngine(DenseVectorizedEngine):
    """Golly Generations-style multistate engine for B/S/C rules."""

    birth: set[int] = field(default_factory=lambda: {2})
    survive: set[int] = field(default_factory=set)
    states: int = 3
    name = "generations"

    def step(self) -> None:
        alive = (self.board == 1).astype(np.uint8)
        neighbors = _neighbor_sum_radius(alive, radius=1, wrap=self.wrap)

        next_board = np.zeros_like(self.board, dtype=np.uint8)
        dead = self.board == 0
        live = self.board == 1

        if self.birth:
            birth_mask = dead & np.isin(neighbors, list(self.birth))
            next_board[birth_mask] = 1
        if self.survive:
            survive_mask = live & np.isin(neighbors, list(self.survive))
            next_board[survive_mask] = 1
        else:
            survive_mask = np.zeros_like(live, dtype=bool)

        if self.states > 2:
            die_from_live = live & (~survive_mask)
            next_board[die_from_live] = 2

            decaying = self.board > 1
            if np.any(decaying):
                progressed = self.board[decaying].astype(np.int32) + 1
                progressed[progressed >= self.states] = 0
                next_board[decaying] = progressed.astype(np.uint8)

        self.board = next_board

    def alive_count(self) -> int:
        return int(np.count_nonzero(self.board == 1))

    def alive_points(self) -> np.ndarray:
        points = np.argwhere(self.board > 0)
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        return np.column_stack((points[:, 1], points[:, 0])).astype(float)

    def board_view(self) -> np.ndarray:
        return (self.board > 0).astype(np.uint8)


@dataclass
class LargerThanLifeEngine(DenseVectorizedEngine):
    """Golly Larger-than-Life style totalistic range-R rule engine."""

    radius: int = 2
    birth_range: tuple[int, int] = (34, 45)
    survive_range: tuple[int, int] = (34, 58)
    name = "largerlife"

    def step(self) -> None:
        b = self.board
        neighbors = _neighbor_sum_radius(b, radius=self.radius, wrap=self.wrap)
        bmin, bmax = self.birth_range
        smin, smax = self.survive_range
        born = (b == 0) & (neighbors >= bmin) & (neighbors <= bmax)
        survive = (b == 1) & (neighbors >= smin) & (neighbors <= smax)
        self.board = np.where(born | survive, 1, 0).astype(np.uint8)


@dataclass
class VonNeumannEngine(DenseVectorizedEngine):
    """jvN-style 2-state totalistic engine using von Neumann neighborhood."""

    radius: int = 1
    birth: set[int] = field(default_factory=lambda: {2})
    survive: set[int] = field(default_factory=lambda: {1, 2})
    name = "jvn"

    def step(self) -> None:
        b = self.board
        neighbors = _neighbor_sum_von_neumann_radius(b, radius=self.radius, wrap=self.wrap)
        born = (b == 0) & np.isin(neighbors, list(self.birth))
        survive = (b == 1) & np.isin(neighbors, list(self.survive))
        self.board = np.where(born | survive, 1, 0).astype(np.uint8)


@dataclass
class RuleLoaderEngine(DenseVectorizedEngine):
    """RuleLoader-inspired configurable totalistic engine loaded from a JSON rule file."""

    neighborhood: str = "moore"  # moore | vonneumann
    radius: int = 1
    birth: set[int] = field(default_factory=lambda: {3})
    survive: set[int] = field(default_factory=lambda: {2, 3})
    states: int = 2
    name = "ruleloader"

    def step(self) -> None:
        if self.neighborhood == "moore":
            neighbors = _neighbor_sum_radius((self.board == 1).astype(np.uint8), radius=self.radius, wrap=self.wrap)
        elif self.neighborhood == "vonneumann":
            if self.radius != 1:
                raise ValueError("ruleloader currently supports vonNeumann only with radius=1.")
            neighbors = _neighbor_sum_von_neumann((self.board == 1).astype(np.uint8), wrap=self.wrap)
        else:
            raise ValueError(f"Unsupported neighborhood: {self.neighborhood}")

        dead = self.board == 0
        live = self.board == 1
        next_board = np.zeros_like(self.board, dtype=np.uint8)

        if self.birth:
            next_board[dead & np.isin(neighbors, list(self.birth))] = 1
        if self.survive:
            survive_mask = live & np.isin(neighbors, list(self.survive))
            next_board[survive_mask] = 1
        else:
            survive_mask = np.zeros_like(live, dtype=bool)

        if self.states > 2:
            next_board[live & (~survive_mask)] = 2
            decaying = self.board > 1
            if np.any(decaying):
                progressed = self.board[decaying].astype(np.int32) + 1
                progressed[progressed >= self.states] = 0
                next_board[decaying] = progressed.astype(np.uint8)

        self.board = next_board

    def alive_count(self) -> int:
        return int(np.count_nonzero(self.board == 1))

    def alive_points(self) -> np.ndarray:
        points = np.argwhere(self.board > 0)
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        return np.column_stack((points[:, 1], points[:, 0])).astype(float)

    def board_view(self) -> np.ndarray:
        return (self.board > 0).astype(np.uint8)

if NUMBA_AVAILABLE:
    from numba import prange

    @njit(cache=False, parallel=True)
    def _step_bounded_numba(board: np.ndarray) -> np.ndarray:
        h, w = board.shape
        out = np.zeros((h, w), dtype=np.uint8)
        for y in prange(h):
            for x in range(w):
                neighbors = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny = y + dy
                        nx = x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            neighbors += board[ny, nx]

                alive = board[y, x] == 1
                if neighbors == 3 or (alive and neighbors == 2):
                    out[y, x] = 1
        return out


    @njit(cache=False)
    def _step_wrap_numba(board: np.ndarray) -> np.ndarray:
        h, w = board.shape
        out = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                neighbors = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny = (y + dy) % h
                        nx = (x + dx) % w
                        neighbors += board[ny, nx]

                alive = board[y, x] == 1
                if neighbors == 3 or (alive and neighbors == 2):
                    out[y, x] = 1
        return out


@dataclass
class DenseNumbaEngine(DenseVectorizedEngine):
    name = "numba"

    def step(self) -> None:
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba backend selected but numba is not installed.")
        if self.wrap:
            self.board = _step_wrap_numba(self.board)
        else:
            self.board = _step_bounded_numba(self.board)


@dataclass
class DenseTorchEngine(LifeEngine):
    width: int
    height: int
    wrap: bool = False
    board: torch.Tensor = field(init=False)
    kernel: torch.Tensor = field(init=False)

    name = "torch"
    display_mode = "image"

    def __post_init__(self) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch backend selected but torch is not installed.")
        self.board = torch.zeros((self.height, self.width), dtype=torch.uint8, device="cpu")
        self.kernel = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
            device="cpu",
        ).view(1, 1, 3, 3)

    def clear(self) -> None:
        self.board.zero_()

    def randomize(self, density: float, seed: int) -> None:
        self.clear()
        points = sample_live_coordinates(self.width, self.height, density, seed)
        if points.size:
            coords = torch.from_numpy(points).to(dtype=torch.long)
            self.board[coords[:, 1], coords[:, 0]] = 1

    def seed_pattern(self, pattern: Iterable[tuple[int, int]], anchor_x: int = 0, anchor_y: int = 0) -> None:
        self.clear()
        for dx, dy in pattern:
            x = anchor_x + dx
            y = anchor_y + dy
            if self.wrap:
                self.board[y % self.height, x % self.width] = 1
            elif 0 <= x < self.width and 0 <= y < self.height:
                self.board[y, x] = 1

    def step(self) -> None:
        board_f = self.board.to(torch.float32)
        x = board_f.unsqueeze(0).unsqueeze(0)

        if self.wrap:
            x = F.pad(x, (1, 1, 1, 1), mode="circular")
            neighbors = F.conv2d(x, self.kernel).squeeze(0).squeeze(0)
        else:
            neighbors = F.conv2d(x, self.kernel, padding=1).squeeze(0).squeeze(0)

        alive = self.board == 1
        self.board = ((neighbors == 3.0) | (alive & (neighbors == 2.0))).to(torch.uint8)

    def alive_count(self) -> int:
        return int(self.board.sum().item())

    def alive_points(self) -> np.ndarray:
        points = torch.nonzero(self.board, as_tuple=False)
        if points.numel() == 0:
            return np.empty((0, 2), dtype=float)
        points_np = points.cpu().numpy()
        return np.column_stack((points_np[:, 1], points_np[:, 0])).astype(float)

    def board_view(self) -> np.ndarray:
        return self.board.cpu().numpy()


@dataclass
class DenseCUDAEngine(LifeEngine):
    """
    CUDA GPU-accelerated engine using PyTorch with automatic GPU detection.
    
    Uses torch.cuda for GPU acceleration when available, Falls back to CPU
    when CUDA is not available. Supports custom rules via rule string (B.../S...).
    """
    width: int
    height: int
    wrap: bool = False
    rule: str = "B3/S23"
    board: torch.Tensor = field(init=False)
    kernel: torch.Tensor = field(init=False)
    _device: str = field(init=False)
    _rule_parsed: tuple[set[int], set[int]] = field(init=False)

    name = "cuda"
    display_mode = "image"

    def __post_init__(self) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("CUDA backend selected but torch is not installed.")
        
        # Auto-detect CUDA GPU
        if torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
        
        # Parse rule string
        self._rule_parsed = self._parse_rule(self.rule)
        birth, survive = self._rule_parsed
        max_neighbors = 9
        
        self.board = torch.zeros((self.height, self.width), dtype=torch.uint8, device=self._device)
        
        # Create pre-computed lookup table for faster transitions
        # Maps (neighbor_count, current_state) -> next_state
        self._transition = torch.zeros((max_neighbors + 1, 2), dtype=torch.uint8, device=self._device)
        for n in range(max_neighbors + 1):
            self._transition[n, 0] = 1 if n in birth else 0
            self._transition[n, 1] = 1 if n in survive else 0
        
        # Pre-create kernel on appropriate device
        self.kernel = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
            device=self._device,
        ).view(1, 1, 3, 3)

    @staticmethod
    def _parse_rule(rule: str) -> tuple[set[int], set[int]]:
        """Parse rule string like 'B3/S23' into birth/survive sets."""
        birth, survive = set(), set()
        if '/' not in rule:
            return birth, survive
        
        parts = rule.split('/')
        for part in parts:
            target = birth if part.startswith('B') else survive if part.startswith('S') else None
            if target is not None:
                nums = ''.join(c for c in part if c.isdigit())
                target.update(int(c) for c in nums)
        return birth, survive

    def clear(self) -> None:
        self.board.zero_()

    def randomize(self, density: float, seed: int) -> None:
        self.clear()
        # Use CPU for random generation then transfer to GPU
        cpu_board = (np.random.default_rng(seed).random((self.height, self.width)) < density).astype(np.uint8)
        self.board = torch.from_numpy(cpu_board).to(device=self._device)

    def seed_pattern(self, pattern: Iterable[tuple[int, int]], anchor_x: int = 0, anchor_y: int = 0) -> None:
        self.clear()
        for dx, dy in pattern:
            x = anchor_x + dx
            y = anchor_y + dy
            if self.wrap:
                self.board[y % self.height, x % self.width] = 1
            elif 0 <= x < self.width and 0 <= y < self.height:
                self.board[y, x] = 1

    def step(self) -> None:
        board_f = self.board.to(torch.float32)
        x = board_f.unsqueeze(0).unsqueeze(0)

        if self.wrap:
            x = F.pad(x, (1, 1, 1, 1), mode="circular")
            neighbors = F.conv2d(x, self.kernel).squeeze(0).squeeze(0)
        else:
            neighbors = F.conv2d(x, self.kernel, padding=1).squeeze(0).squeeze(0)

        # Use pre-computed transition table
        neighbors_int = neighbors.to(torch.int64).clamp(0, 9)
        alive = self.board == 1
        self.board = self._transition[neighbors_int, alive.to(torch.int64)].to(torch.uint8)

    def alive_count(self) -> int:
        return int(self.board.sum().item())

    def alive_points(self) -> np.ndarray:
        points = torch.nonzero(self.board, as_tuple=False)
        if points.numel() == 0:
            return np.empty((0, 2), dtype=float)
        points_np = points.cpu().numpy()
        return np.column_stack((points_np[:, 1], points_np[:, 0])).astype(float)

    def board_view(self) -> np.ndarray:
        return self.board.cpu().numpy()


@dataclass
class QuickLifeEngine(DenseVectorizedEngine):
    """QuickLife-inspired dense engine that only updates a live-cell ROI when unbounded wrapping is off."""

    name = "quicklife"
    _bbox: tuple[int, int, int, int] | None = None  # x0, x1, y0, y1 inclusive

    def _update_bbox(self) -> None:
        points = np.argwhere(self.board == 1)
        if points.size == 0:
            self._bbox = None
            return
        y0 = int(points[:, 0].min())
        y1 = int(points[:, 0].max())
        x0 = int(points[:, 1].min())
        x1 = int(points[:, 1].max())
        self._bbox = (x0, x1, y0, y1)

    def clear(self) -> None:
        super().clear()
        self._bbox = None

    def randomize(self, density: float, seed: int) -> None:
        super().randomize(density, seed)
        self._update_bbox()

    def seed_pattern(self, pattern: Iterable[tuple[int, int]], anchor_x: int = 0, anchor_y: int = 0) -> None:
        super().seed_pattern(pattern, anchor_x, anchor_y)
        self._update_bbox()

    def step(self) -> None:
        if self.wrap:
            super().step()
            self._update_bbox()
            return
        if self._bbox is None:
            return

        x0, x1, y0, y1 = self._bbox
        ex0 = max(0, x0 - 1)
        ex1 = min(self.width - 1, x1 + 1)
        ey0 = max(0, y0 - 1)
        ey1 = min(self.height - 1, y1 + 1)

        region = self.board[ey0 : ey1 + 1, ex0 : ex1 + 1]
        padded = np.pad(region, 1, mode="constant")
        neighbors = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        )
        next_region = np.where((neighbors == 3) | ((region == 1) & (neighbors == 2)), 1, 0).astype(np.uint8)

        self.board.fill(0)
        if np.any(next_region):
            self.board[ey0 : ey1 + 1, ex0 : ex1 + 1] = next_region
        self._update_bbox()


@dataclass
class HashLifeEngine(DenseVectorizedEngine):
    """HashLife-inspired memoized one-step transitions on dense boards."""

    name = "hashlife"
    _cache: OrderedDict[bytes, np.ndarray] = field(default_factory=OrderedDict)
    _max_cache_entries: int = 4096

    def step(self) -> None:
        key = self.board.tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            self.board = cached.copy()
            self._cache.move_to_end(key)
            return

        super().step()
        self._cache[key] = self.board.copy()
        if len(self._cache) > self._max_cache_entries:
            self._cache.popitem(last=False)


@dataclass(frozen=True)
class QuadNode:
    level: int
    pop: int
    nw: "QuadNode | None" = None
    ne: "QuadNode | None" = None
    sw: "QuadNode | None" = None
    se: "QuadNode | None" = None


class QuadHashLifeCore:
    def __init__(self) -> None:
        self._intern: dict[tuple, QuadNode] = {}
        self._step_cache: dict[QuadNode, QuadNode] = {}
        self._array_cache: dict[QuadNode, np.ndarray] = {}

    def leaf(self, alive: int) -> QuadNode:
        key = ("leaf", int(bool(alive)))
        node = self._intern.get(key)
        if node is None:
            node = QuadNode(level=0, pop=int(bool(alive)))
            self._intern[key] = node
        return node

    def join(self, nw: QuadNode, ne: QuadNode, sw: QuadNode, se: QuadNode) -> QuadNode:
        key = ("node", nw, ne, sw, se)
        node = self._intern.get(key)
        if node is None:
            node = QuadNode(
                level=nw.level + 1,
                pop=nw.pop + ne.pop + sw.pop + se.pop,
                nw=nw,
                ne=ne,
                sw=sw,
                se=se,
            )
            self._intern[key] = node
        return node

    def from_array(self, arr: np.ndarray) -> QuadNode:
        size = arr.shape[0]
        if size == 1:
            return self.leaf(int(arr[0, 0]))
        half = size // 2
        nw = self.from_array(arr[:half, :half])
        ne = self.from_array(arr[:half, half:])
        sw = self.from_array(arr[half:, :half])
        se = self.from_array(arr[half:, half:])
        return self.join(nw, ne, sw, se)

    def to_array(self, node: QuadNode) -> np.ndarray:
        cached = self._array_cache.get(node)
        if cached is not None:
            return cached
        if node.level == 0:
            out = np.array([[1 if node.pop else 0]], dtype=np.uint8)
        else:
            nw = self.to_array(node.nw)
            ne = self.to_array(node.ne)
            sw = self.to_array(node.sw)
            se = self.to_array(node.se)
            top = np.concatenate((nw, ne), axis=1)
            bottom = np.concatenate((sw, se), axis=1)
            out = np.concatenate((top, bottom), axis=0)
        self._array_cache[node] = out
        return out

    def _dense_step(self, arr: np.ndarray) -> np.ndarray:
        padded = np.pad(arr, 1, mode="constant")
        neighbors = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        )
        return np.where((neighbors == 3) | ((arr == 1) & (neighbors == 2)), 1, 0).astype(np.uint8)

    def step_once(self, node: QuadNode) -> QuadNode:
        cached = self._step_cache.get(node)
        if cached is not None:
            return cached
        arr = self.to_array(node)
        next_arr = self._dense_step(arr)
        res = self.from_array(next_arr)
        self._step_cache[node] = res
        return res

    def center_step(self, node: QuadNode) -> QuadNode:
        """Return center half after one generation for compatibility with legacy API."""
        arr = self.to_array(node)
        next_arr = self._dense_step(arr)
        half = arr.shape[0] // 2
        off = (arr.shape[0] - half) // 2
        center = next_arr[off : off + half, off : off + half]
        return self.from_array(center)


@dataclass
class HashLifeTreeEngine(DenseVectorizedEngine):
    """Quadtree HashLife-style core with canonical node interning + memoized transitions/jumps."""

    name = "hashlife-tree"
    _core: QuadHashLifeCore = field(default_factory=QuadHashLifeCore)
    _jump_cache: dict[tuple[QuadNode, int], QuadNode] = field(default_factory=dict)

    def step(self) -> int:
        if self.wrap:
            DenseVectorizedEngine.step(self)
            return 1
        return self.advance(1)

    def _embed_root(self) -> tuple[QuadNode, int, int, int]:
        if self.wrap:
            raise RuntimeError("Internal: _embed_root should not be called for wrap mode.")

        side = next_power_of_two(max(self.width, self.height))
        side = max(4, side)
        container = np.zeros((side, side), dtype=np.uint8)
        oy = (side - self.height) // 2
        ox = (side - self.width) // 2
        container[oy : oy + self.height, ox : ox + self.width] = self.board

        return self._core.from_array(container), oy, ox, side

    def _safe_jump_limit(self) -> int:
        points = np.argwhere(self.board == 1)
        if points.size == 0:
            return max(self.width, self.height)
        y0 = int(points[:, 0].min())
        y1 = int(points[:, 0].max())
        x0 = int(points[:, 1].min())
        x1 = int(points[:, 1].max())
        margin = min(x0, y0, self.width - 1 - x1, self.height - 1 - y1)
        return max(1, margin)

    def _pow2_jump(self, node: QuadNode, jump: int) -> QuadNode:
        key = (node, jump)
        cached = self._jump_cache.get(key)
        if cached is not None:
            return cached
        if jump == 1:
            res = self._core.step_once(node)
            self._jump_cache[key] = res
            return res
        half = self._pow2_jump(node, jump // 2)
        res = self._pow2_jump(half, jump // 2)
        self._jump_cache[key] = res
        return res

    def advance(self, generations: int) -> int:
        generations = max(0, int(generations))
        if generations == 0:
            return 0
        if self.wrap:
            for _ in range(generations):
                DenseVectorizedEngine.step(self)
            return generations

        root, oy, ox, _ = self._embed_root()
        remaining = generations
        while remaining > 0:
            safe_limit = min(remaining, self._safe_jump_limit())
            jump = 1 << (safe_limit.bit_length() - 1)
            root = self._pow2_jump(root, jump)
            next_arr = self._core.to_array(root)
            self.board = next_arr[oy : oy + self.height, ox : ox + self.width].astype(np.uint8)
            root, oy, ox, _ = self._embed_root()
            remaining -= jump

        return generations


@dataclass
class AutoAdaptiveEngine(LifeEngine):
    """Adaptive backend selector that profiles available engines and keeps the fastest for the current setup."""

    width: int
    height: int
    wrap: bool = False
    rule: str | None = None
    rule_file: str | None = None
    rule_preset: str | None = None
    _selected: LifeEngine | None = None
    _init_state: tuple = ("empty",)
    _candidates: tuple[str, ...] = ("quicklife", "hashlife", "numba", "torch")
    _profile_steps: int = 16
    display_mode: str = "image"
    name: str = "auto"

    def _apply_init_state(self, engine: LifeEngine) -> None:
        mode = self._init_state[0]
        if mode == "random":
            _, density, seed = self._init_state
            engine.randomize(density, seed)
        elif mode == "pattern":
            _, pattern, anchor_x, anchor_y = self._init_state
            engine.seed_pattern(pattern, anchor_x, anchor_y)
        else:
            engine.clear()

    def _instantiate_backend(self, backend: str) -> LifeEngine:
        return build_engine(
            backend=backend,
            width=self.width,
            height=self.height,
            wrap=self.wrap,
            rule=self.rule,
            rule_file=self.rule_file,
            rule_preset=self.rule_preset,
        )

    def _select_fastest_backend(self) -> None:
        best_backend: str | None = None
        best_sps = -1.0

        for backend in self._candidates:
            if backend == "numba" and not NUMBA_AVAILABLE:
                continue
            if backend == "torch" and not TORCH_AVAILABLE:
                continue
            try:
                candidate = self._instantiate_backend(backend)
            except Exception:
                continue
            self._apply_init_state(candidate)

            if backend == "numba":
                candidate.step()

            started = time.perf_counter()
            candidate.advance(self._profile_steps)
            elapsed = time.perf_counter() - started
            sps = self._profile_steps / elapsed if elapsed > 0 else float("inf")
            if sps > best_sps:
                best_sps = sps
                best_backend = backend

        if best_backend is None:
            best_backend = "dense"

        self._selected = self._instantiate_backend(best_backend)
        self._apply_init_state(self._selected)
        self._selected.wrap = self.wrap
        self.display_mode = self._selected.display_mode
        self.name = f"auto->{best_backend}"

    def _ensure_selected(self) -> LifeEngine:
        if self._selected is None:
            self._select_fastest_backend()
        self._selected.wrap = self.wrap
        return self._selected

    def select_backend(self) -> str:
        self._ensure_selected()
        return self.name

    def clear(self) -> None:
        self._selected = None
        self._init_state = ("empty",)

    def randomize(self, density: float, seed: int) -> None:
        self._selected = None
        self._init_state = ("random", density, seed)

    def seed_pattern(self, pattern: Iterable[tuple[int, int]], anchor_x: int = 0, anchor_y: int = 0) -> None:
        self._selected = None
        self._init_state = ("pattern", tuple(pattern), anchor_x, anchor_y)

    def step(self) -> int:
        return advance_engine(self._ensure_selected())

    def advance(self, generations: int) -> int:
        return self._ensure_selected().advance(generations)

    def alive_count(self) -> int:
        return self._ensure_selected().alive_count()

    def alive_points(self) -> np.ndarray:
        return self._ensure_selected().alive_points()

    def board_view(self) -> np.ndarray:
        return self._ensure_selected().board_view()


def build_engine(
    backend: str,
    width: int,
    height: int,
    wrap: bool,
    rule: str | None = None,
    rule_file: str | None = None,
    rule_preset: str | None = None,
) -> LifeEngine:
    if backend == "auto":
        return AutoAdaptiveEngine(
            width=width,
            height=height,
            wrap=wrap,
            rule=rule,
            rule_file=rule_file,
            rule_preset=rule_preset,
        )
    if backend == "jvn":
        radius, birth, survive = parse_jvn_rule(rule)
        return VonNeumannEngine(width=width, height=height, wrap=wrap, radius=radius, birth=birth, survive=survive)
    if backend == "generations":
        birth, survive, states = parse_generations_rule(rule)
        return GenerationsEngine(
            width=width,
            height=height,
            wrap=wrap,
            birth=birth,
            survive=survive,
            states=states,
        )
    if backend == "largerlife":
        selected_rule = rule
        if rule_preset:
            if rule_preset not in LARGERLIFE_PRESETS:
                raise ValueError(
                    f"Unknown largerlife preset: {rule_preset}. "
                    f"Available: {', '.join(sorted(LARGERLIFE_PRESETS))}"
                )
            selected_rule = LARGERLIFE_PRESETS[rule_preset]
        radius, birth_range, survive_range = parse_largerlife_rule(selected_rule)
        return LargerThanLifeEngine(
            width=width,
            height=height,
            wrap=wrap,
            radius=radius,
            birth_range=birth_range,
            survive_range=survive_range,
        )
    if backend == "ruleloader":
        if not rule_file:
            raise ValueError("ruleloader backend requires --rule-file.")
        spec = load_rule_spec(rule_file)
        neighborhood = str(spec.get("neighborhood", "moore")).lower().replace("-", "")
        if neighborhood in ("vonneumann", "vn"):
            neighborhood = "vonneumann"
        radius = int(spec.get("radius", 1))
        states = int(spec.get("states", 2))
        birth = {int(v) for v in spec.get("birth", [3])}
        survive = {int(v) for v in spec.get("survive", [2, 3])}
        if radius < 1 or states < 2:
            raise ValueError("Invalid rule-file values: radius must be >=1 and states >=2.")
        return RuleLoaderEngine(
            width=width,
            height=height,
            wrap=wrap,
            neighborhood=neighborhood,
            radius=radius,
            birth=birth,
            survive=survive,
            states=states,
        )
    if backend == "quicklife":
        return QuickLifeEngine(width=width, height=height, wrap=wrap)
    if backend == "hashlife":
        return HashLifeEngine(width=width, height=height, wrap=wrap)
    if backend == "hashlife-tree":
        return HashLifeTreeEngine(width=width, height=height, wrap=wrap)
    if backend == "hashlife-infinite":
        from gameoflife.hashlife_infinite import InfiniteHashLifeEngine
        return InfiniteHashLifeEngine(width=width, height=height, wrap=wrap, rule=rule or "B3/S23")
    if backend == "rl":
        from gameoflife.backends.rl_backend import RLBackendEngine

        return RLBackendEngine(width=width, height=height, wrap=wrap)
    if backend == "numba":
        if NUMBA_AVAILABLE:
            return DenseNumbaEngine(width=width, height=height, wrap=wrap)
        print("[warn] numba backend requested but numba is unavailable; falling back to quicklife.")
        return QuickLifeEngine(width=width, height=height, wrap=wrap)
    if backend == "cuda":
        if TORCH_AVAILABLE:
            return DenseCUDAEngine(width=width, height=height, wrap=wrap, rule=rule or "B3/S23")
        print("[warn] cuda backend requested but torch is unavailable; falling back to quicklife.")
        return QuickLifeEngine(width=width, height=height, wrap=wrap)
    if backend == "torch":
        if TORCH_AVAILABLE:
            return DenseTorchEngine(width=width, height=height, wrap=wrap)
        print("[warn] torch backend requested but torch is unavailable; falling back to quicklife.")
        return QuickLifeEngine(width=width, height=height, wrap=wrap)
    raise ValueError(f"Unsupported backend: {backend}")


@dataclass
class SimulationApp:
    engine: LifeEngine
    density: float
    seed: int
    paused: bool = False
    generation: int = 0
    interval_ms: int = 50

    def __post_init__(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.colors import ListedColormap

        if isinstance(self.engine, AutoAdaptiveEngine):
            self.engine.select_backend()

        self._plt = plt
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self.ax.set_xlim(0, self.engine.width)
        self.ax.set_ylim(0, self.engine.height)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_facecolor("#f7f9fc")
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("Conway's Game of Life")

        self.scatter = None
        self.image = None

        if self.engine.display_mode == "scatter":
            self.scatter = self.ax.scatter([], [], c="#1f2937", marker="s", s=12)
        else:
            self.image = self.ax.imshow(
                self.engine.board_view(),
                cmap=ListedColormap(["#f7f9fc", "#111827"]),
                interpolation="nearest",
                origin="lower",
                extent=(0, self.engine.width, 0, self.engine.height),
                vmin=0,
                vmax=1,
            )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self._draw()

        self.ani = FuncAnimation(self.fig, self.animate, interval=self.interval_ms, blit=False)

    def _draw(self) -> None:
        if self.scatter is not None:
            self.scatter.set_offsets(self.engine.alive_points())
        else:
            self.image.set_data(self.engine.board_view())

        mode = "WRAP" if self.engine.wrap else "BOUNDED"
        state = "PAUSED" if self.paused else "RUNNING"
        self.ax.set_title(
            (
                f"Backend {self.engine.name.upper()} | Gen {self.generation:,} | Alive {self.engine.alive_count():,} "
                f"| {mode} | {state} | {self.interval_ms}ms"
            ),
            fontsize=11,
            pad=10,
        )
        self.fig.supxlabel(
            "Keys: [space]=pause  [n]=step  [r]=random  [g]=glider gun  [c]=clear  [w]=toggle wrap  [up/down]=speed",
            fontsize=9,
            y=0.03,
        )

    def on_key_press(self, event) -> None:
        key = (event.key or "").lower()

        if key == " ":
            self.paused = not self.paused
        elif key == "n" and self.paused:
            self.generation += advance_engine(self.engine)
        elif key == "r":
            self.engine.randomize(self.density, self.seed)
            self.generation = 0
        elif key == "g":
            anchor_x = max(0, self.engine.width // 2 - 18)
            anchor_y = max(0, self.engine.height // 2 - 5)
            self.engine.seed_pattern(GOSPER_GLIDER_GUN, anchor_x, anchor_y)
            self.generation = 0
        elif key == "c":
            self.engine.clear()
            self.generation = 0
        elif key == "w":
            self.engine.wrap = not self.engine.wrap
        elif key == "up":
            self.interval_ms = max(5, self.interval_ms - 5)
            self.ani.event_source.interval = self.interval_ms
        elif key == "down":
            self.interval_ms = min(500, self.interval_ms + 5)
            self.ani.event_source.interval = self.interval_ms

        self._draw()
        self.fig.canvas.draw_idle()

    def animate(self, _) -> None:
        if not self.paused:
            self.generation += advance_engine(self.engine)
        self._draw()

    def run(self) -> None:
        self._plt.tight_layout()
        self._plt.show()


@dataclass
class TerminalSimulationApp:
    engine: LifeEngine
    density: float
    seed: int
    paused: bool = False
    generation: int = 0
    interval_ms: int = 50
    quit_requested: bool = False

    def _reset_random(self) -> None:
        self.engine.randomize(self.density, self.seed)
        self.generation = 0

    def _reset_glider(self) -> None:
        anchor_x = max(0, self.engine.width // 2 - 18)
        anchor_y = max(0, self.engine.height // 2 - 5)
        self.engine.seed_pattern(GOSPER_GLIDER_GUN, anchor_x, anchor_y)
        self.generation = 0

    def _draw(self, stdscr, alive_attr: int) -> None:
        board = self.engine.board_view()
        board_h, board_w = board.shape
        term_h, term_w = stdscr.getmaxyx()

        draw_top = 1
        draw_rows = max(1, term_h - 3)
        draw_cols_cells = max(1, (term_w - 1) // 2)

        y_idx = np.linspace(0, board_h - 1, draw_rows, dtype=int)
        x_idx = np.linspace(0, board_w - 1, draw_cols_cells, dtype=int)
        sampled = board[np.ix_(y_idx, x_idx)]

        mode = "WRAP" if self.engine.wrap else "BOUNDED"
        state = "PAUSED" if self.paused else "RUNNING"
        header = (
            f"GameOfLife TUI | {self.engine.name.upper()} | Gen {self.generation:,} | "
            f"Alive {self.engine.alive_count():,} | {mode} | {state} | {self.interval_ms}ms"
        )
        footer = "Keys: q quit | space pause | n step | r random | g gun | c clear | w wrap | +/- speed"

        stdscr.erase()
        try:
            stdscr.addnstr(0, 0, header, max(0, term_w - 1))
        except Exception:
            return

        for row_i in range(sampled.shape[0]):
            y = draw_top + row_i
            if y >= term_h - 1:
                break
            row = sampled[row_i]
            x = 0
            for value in row:
                if x + 1 >= term_w:
                    break
                if value:
                    try:
                        stdscr.addstr(y, x, "██", alive_attr)
                    except Exception:
                        break
                else:
                    try:
                        stdscr.addstr(y, x, "  ")
                    except Exception:
                        break
                x += 2

        if term_h >= 2:
            try:
                stdscr.addnstr(term_h - 1, 0, footer, max(0, term_w - 1))
            except Exception:
                pass
        stdscr.refresh()

    def _handle_key(self, key: int) -> None:
        if key in (ord("q"), ord("Q")):
            self.quit_requested = True
        elif key == ord(" "):
            self.paused = not self.paused
        elif key in (ord("n"), ord("N")) and self.paused:
            self.generation += advance_engine(self.engine)
        elif key in (ord("r"), ord("R")):
            self._reset_random()
        elif key in (ord("g"), ord("G")):
            self._reset_glider()
        elif key in (ord("c"), ord("C")):
            self.engine.clear()
            self.generation = 0
        elif key in (ord("w"), ord("W")):
            self.engine.wrap = not self.engine.wrap
        elif key in (ord("+"), ord("=")):
            self.interval_ms = max(5, self.interval_ms - 5)
        elif key == ord("-"):
            self.interval_ms = min(500, self.interval_ms + 5)

    def _run_loop(self, stdscr) -> None:
        import curses

        stdscr.nodelay(True)
        stdscr.keypad(True)
        try:
            curses.curs_set(0)
        except Exception:
            pass

        alive_attr = curses.A_REVERSE
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
            alive_attr = curses.color_pair(1) | curses.A_BOLD

        while not self.quit_requested:
            started = time.perf_counter()

            key = stdscr.getch()
            while key != -1:
                self._handle_key(key)
                key = stdscr.getch()

            if not self.paused:
                self.generation += advance_engine(self.engine)

            self._draw(stdscr, alive_attr)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            sleep_ms = max(0.0, self.interval_ms - elapsed_ms)
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

    def run(self) -> None:
        import curses

        curses.wrapper(self._run_loop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conway's Game of Life")
    parser.add_argument("--width", type=int, default=300, help="Board width")
    parser.add_argument("--height", type=int, default=200, help="Board height")
    parser.add_argument("--density", type=float, default=0.15, help="Random initial population density [0,1]")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic random initialization")
    parser.add_argument("--wrap", action="store_true", help="Use toroidal wrapping at edges")
    parser.add_argument(
        "--pattern",
        choices=tuple(["random", "empty"] + list(ALL_PATTERNS.keys())),
        default="glider-gun",
        help="Initial board pattern: random, empty, or pattern name (see gameoflife/patterns.py)",
    )
    parser.add_argument("--interval", type=int, default=30, help="Animation interval in milliseconds")
    parser.add_argument(
        "--ui",
        choices=("gui", "tui"),
        default="tui",
        help="Rendering mode: gui (matplotlib) or tui (retro terminal).",
    )
    parser.add_argument(
        "--backend",
        choices=(
            "auto",
            "jvn",
            "generations",
            "largerlife",
            "ruleloader",
            "quicklife",
            "hashlife",
            "hashlife-tree",
            "hashlife-infinite",
            "rl",
            "numba",
            "torch",
            "cuda",
        ),
        default="auto",
        help=(
            "Simulation backend: auto (torch conv if available, else quicklife), "
            "jvn, generations, largerlife, ruleloader, quicklife, hashlife, hashlife-tree, "
            "hashlife-infinite, rl, numba, torch conv2d, or cuda GPU"
        ),
    )
    parser.add_argument(
        "--rule",
        type=str,
        default=None,
        help=(
            "Optional rule string for certain backends: "
            "generations uses B../S../Ck (e.g. B2/S/C3), "
            "largerlife uses Rr,Bx-y,Su-v (e.g. R5,B34-45,S34-58), "
            "jvn uses B../S.. or Rn,B../S.. (e.g. R2,B3/S23)."
        ),
    )
    parser.add_argument(
        "--rule-file",
        type=str,
        default=None,
        help="Path to a JSON rule spec for --backend ruleloader.",
    )
    parser.add_argument(
        "--rule-preset",
        type=str,
        default=None,
        help=(
            "Named rule preset (currently for largerlife). "
            f"Available: {', '.join(sorted(LARGERLIFE_PRESETS))}."
        ),
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=0,
        help="Run headless benchmark for N generations and exit (no GUI).",
    )
    parser.add_argument(
        "--benchmark-all",
        action="store_true",
        help="Benchmark all available backends headlessly and exit (uses --benchmark-steps).",
    )
    parser.add_argument(
        "--export-video",
        type=str,
        default=None,
        help="Export simulation to video file (.mp4 or .gif)",
    )
    parser.add_argument(
        "--export-fps",
        type=int,
        default=30,
        help="Frames per second for video export",
    )
    parser.add_argument(
        "--export-scale",
        type=int,
        default=2,
        help="Pixel scale for video export (1=original, 2=2x, etc.)",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run environment and backend diagnostics, then exit.",
    )
    parser.add_argument(
        "--rl-stats",
        action="store_true",
        help="Enable periodic RL backend inference stats output (avg/max/last ms).",
    )
    return parser.parse_args()


def _initialize_engine(engine: LifeEngine, pattern: str, density: float, seed: int) -> None:
    if pattern == "random":
        engine.randomize(density, seed)
    elif pattern in ALL_PATTERNS:
        # Use pattern from library
        pattern_cells = ALL_PATTERNS[pattern]
        anchor_x = max(0, engine.width // 2 - 10)
        anchor_y = max(0, engine.height // 2 - 10)
        engine.seed_pattern(pattern_cells, anchor_x, anchor_y)
    else:
        engine.clear()


def run_benchmark(
    engine: LifeEngine, pattern: str, density: float, seed: int, steps: int
) -> tuple[float, float, int]:
    _initialize_engine(engine, pattern, density, seed)

    if isinstance(engine, AutoAdaptiveEngine):
        engine.select_backend()

    if engine.name == "numba" and NUMBA_AVAILABLE:
        # Warm up JIT compilation before measured steps.
        engine.step()

    started = time.perf_counter()
    advanced = engine.advance(steps)
    elapsed = time.perf_counter() - started

    sps = advanced / elapsed if elapsed > 0 else float("inf")
    cells = engine.width * engine.height
    throughput = cells * sps
    alive = engine.alive_count()
    return sps, throughput, alive


def print_benchmark_result(backend: str, steps: int, sps: float, throughput: float, alive: int) -> None:
    print(
        (
            f"backend={backend} steps={steps} "
            f"steps_per_sec={sps:,.1f} cell_updates_per_sec={throughput:,.0f} alive={alive:,}"
        )
    )


def run_benchmark_all(
    width: int,
    height: int,
    wrap: bool,
    pattern: str,
    density: float,
    seed: int,
    steps: int,
    rule: str | None = None,
    rule_file: str | None = None,
    rule_preset: str | None = None,
) -> None:
    print(
        (
            f"benchmark-all width={width} height={height} wrap={wrap} "
            f"pattern={pattern} density={density} seed={seed} steps={steps}"
        )
    )
    for backend in (
        "jvn",
        "generations",
        "largerlife",
        "ruleloader",
        "quicklife",
        "hashlife",
        "hashlife-tree",
        "rl",
        "numba",
        "torch",
    ):
        if backend == "numba" and not NUMBA_AVAILABLE:
            print("backend=numba skipped reason=numba_unavailable")
            continue
        if backend == "torch" and not TORCH_AVAILABLE:
            print("backend=torch skipped reason=torch_unavailable")
            continue
        if backend == "rl" and not TORCH_AVAILABLE:
            print("backend=rl skipped reason=torch_unavailable")
            continue
        if backend == "ruleloader" and not rule_file:
            print("backend=ruleloader skipped reason=rule_file_missing")
            continue
        engine = build_engine(backend, width, height, wrap, rule, rule_file, rule_preset)
        sps, throughput, alive = run_benchmark(engine, pattern, density, seed, steps)
        print_benchmark_result(backend, steps, sps, throughput, alive)


def run_doctor(args: argparse.Namespace) -> int:
    import os
    import platform
    import sys

    print("gameoflife doctor")
    print(f"python_version={platform.python_version()}")
    print(f"python_executable={sys.executable}")
    print(f"platform={platform.platform()}")
    print(f"cwd={os.getcwd()}")
    print(f"default_ui={args.ui}")
    print(f"default_backend={args.backend}")
    print(f"numba_available={NUMBA_AVAILABLE}")
    print(f"torch_available={TORCH_AVAILABLE}")

    try:
        import matplotlib

        print(f"matplotlib_version={matplotlib.__version__}")
        print(f"matplotlib_backend={matplotlib.get_backend()}")
    except Exception as exc:
        print(f"matplotlib_status=error:{exc}")

    try:
        engine = build_engine(
            backend=args.backend,
            width=max(3, args.width),
            height=max(3, args.height),
            wrap=args.wrap,
            rule=args.rule,
            rule_file=args.rule_file,
            rule_preset=args.rule_preset,
        )
        _initialize_engine(engine, args.pattern, args.density, args.seed)
        engine.advance(1)
        print(f"engine_smoke_test=ok:{engine.name}")
    except Exception as exc:
        print(f"engine_smoke_test=error:{exc}")
        return 1
    return 0


def main() -> None:
    args = parse_args()

    if args.doctor:
        raise SystemExit(run_doctor(args))

    width = max(3, args.width)
    height = max(3, args.height)
    interval = max(5, args.interval)
    benchmark_steps = max(1, args.benchmark_steps)

    if args.benchmark_all:
        run_benchmark_all(
            width=width,
            height=height,
            wrap=args.wrap,
            pattern=args.pattern,
            density=args.density,
            seed=args.seed,
            steps=benchmark_steps,
            rule=args.rule,
            rule_file=args.rule_file,
            rule_preset=args.rule_preset,
        )
        return

    engine = build_engine(args.backend, width, height, args.wrap, args.rule, args.rule_file, args.rule_preset)
    if args.rl_stats and hasattr(engine, "set_stats"):
        try:
            engine.set_stats(True)
        except Exception:
            pass

    if args.benchmark_steps > 0:
        sps, throughput, alive = run_benchmark(engine, args.pattern, args.density, args.seed, benchmark_steps)
        print_benchmark_result(engine.name, benchmark_steps, sps, throughput, alive)
        return

    # Video export mode (headless)
    if args.export_video:
        _initialize_engine(engine, args.pattern, args.density, args.seed)
        frames = capture_simulation(engine, steps=benchmark_steps, skip=1)
        export_video(
            args.export_video,
            frames,
            fps=args.export_fps,
            pixel_size=args.export_scale,
        )
        return

    if args.pattern == "random":
        engine.randomize(args.density, args.seed)
    elif args.pattern == "glider-gun":
        anchor_x = max(0, width // 2 - 18)
        anchor_y = max(0, height // 2 - 5)
        engine.seed_pattern(GOSPER_GLIDER_GUN, anchor_x, anchor_y)

    if args.ui == "tui":
        app = TerminalSimulationApp(engine=engine, density=args.density, seed=args.seed, interval_ms=interval)
        app.run()
        return

    app = SimulationApp(engine=engine, density=args.density, seed=args.seed, interval_ms=interval)
    app.ani.event_source.interval = interval
    app.run()


if __name__ == "__main__":
    main()
