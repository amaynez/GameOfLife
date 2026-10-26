"""
Ruletable and Macrocell support for Golly-compatible rule files.

This module parses and executes rules defined in Golly's .rule format,
which supports:
- Multiple states (not just alive/dead)
- Custom neighborhoods (Moore, von Neumann, or custom)
- Complex birth/survival conditions
- Colors and display information
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np


#: Maximum number of states supported
MAX_STATES = 256


@dataclass
class RuleTableSpec:
    """Complete specification for a.rule or .table rule."""
    name: str
    states: int = 2
    neighborhood: str = "moore"  # "moore", "vonneumann", or "unknown"
    radius: int = 1
    birth: set[int] = field(default_factory=set)
    survive: set[int] = field(default_factory=set)
    colors: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    #: For ruletables, maps (neighbors, current_state) -> next_state
    transitions: dict[tuple[int, int], int] = field(default_factory=dict)
    is_ruletable: bool = False
    #: Additional rule file info
    info: dict[str, str] = field(default_factory=dict)


def parse_color(col_str: str) -> tuple[int, int, int]:
    """Parse a color string like '#FF0000' or '255,0,0'."""
    col_str = col_str.strip().strip('"').strip("'")
    if col_str.startswith('#'):
        return (
            int(col_str[1:3], 16),
            int(col_str[3:5], 16),
            int(col_str[5:7], 16),
        )
    parts = [int(x.strip()) for x in col_str.split(',')]
    return tuple(parts[:3]) if len(parts) >= 3 else (0, 0, 0)


def load_ruletable_file(path: str | Path) -> RuleTableSpec:
    """
    Load a Golly .rule or .table file.
    
    Args:
        path: Path to the rule file
        
    Returns:
        RuleTableSpec with all parsed information
    """
    path = Path(path)
    content = path.read_text()
    
    # Check if it's a .rule (XML) or .table (text) format
    if content.strip().startswith('<?xml') or content.strip().startswith('<rule'):
        return _parse_xml_rule(content)
    else:
        return _parse_table_rule(content)


def _parse_xml_rule(content: str) -> RuleTableSpec:
    """Parse XML-format .rule files."""
    # Remove XML declaration if present
    if content.startswith('<?xml'):
        content = content[content.find('<'):]
    
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML in rule file: {e}")
    
    spec = RuleTableSpec(name=root.get('name', 'unnamed'))
    
    # Parse states
    states_elem = root.find('states')
    if states_elem is not None:
        spec.states = int(states_elem.get('count', 2))
    
    # Parse neighborhood
    neighborhood_elem = root.find('neighborhood')
    if neighborhood_elem is not None:
        spec.neighborhood = neighborhood_elem.get('type', 'moore').lower()
        spec.radius = int(neighborhood_elem.get('radius', 1))
    
    # Parse colors
    for color_elem in root.findall('.//color'):
        state = int(color_elem.get('state', 0))
        color_str = color_elem.get('color', '#000000')
        spec.colors[state] = parse_color(color_str)
    
    # Parse ruletable (table-based transition definitions)
    table_elem = root.find('table')
    if table_elem is not None:
        spec.is_ruletable = True
        for rule_elem in table_elem.findall('rule'):
            input_str = rule_elem.get('inputs', '')
            output = int(rule_elem.get('output', 0))
            
            # Parse input format like "1,2,3" (states) or "0-2" (ranges)
            # Each input represents neighbor state -> current state -> output state
            # Simplify: for now handle simple neighbor count based rules
            if ',' in input_str:
                parts = [p.strip() for p in input_str.split(',')]
                # Format: current_state,neighbor_count,neighbor_state,...
                # For 2-state rules: current_state,neighbor_count
                if len(parts) == 2:
                    try:
                        current = int(parts[0])
                        neighbors = int(parts[1])
                        spec.transitions[(current, neighbors)] = output
                    except ValueError:
                        pass
    
    # Parse simple birth/survive from <rule> elements
    # Format: <rule name="B3/S23">...</rule> inside <rules> element
    rules_elem = root.find('rules')
    if rules_elem is not None:
        for rule_child in rules_elem.findall('rule'):
            rule_str = rule_child.get('name', '')
            if '/' in rule_str:
                parts = rule_str.split('/')
                if parts[0].startswith('B'):
                    # Parse B.../S... format
                    birth_str = parts[0][1:]
                    survive_str = parts[1] if len(parts) > 1 else ''
                    spec.birth = set(int(c) for c in birth_str if c.isdigit())
                    spec.survive = set(int(c) for c in survive_str if c.isdigit())
    
    # Parse info elements
    for info_elem in root.findall('.//info'):
        spec.info[info_elem.get('name', '')] = info_elem.text or ''
    
    return spec


def _parse_table_rule(content: str) -> RuleTableSpec:
    """
    Parse legacy .table format (simple text format).
    
    The table format is:
    @TABLE
    n_states = 2
    neighbourhood = Moore
    [states definitions...]
    """
    spec = RuleTableSpec(name='unnamed')
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if line.startswith('n_states'):
            match = re.search(r'n_states\s*=\s*(\d+)', line)
            if match:
                spec.states = int(match.group(1))
        
        elif line.startswith('neighbourhood') or line.startswith('neighborhood'):
            match = re.search(r'neighbourhood\s*=\s*(\w+)', line)
            if match:
                spec.neighborhood = match.group(1).lower()
        
        elif line.startswith('Birth'):
            match = re.search(r'Birth\s*[=:]\s*\{([^}]*)\}', line)
            if match:
                nums = re.findall(r'\d+', match.group(1))
                spec.birth = set(int(n) for n in nums)
        
        elif line.startswith('Survive'):
            match = re.search(r'Survive\s*[=:]\s*\{([^}]*)\}', line)
            if match:
                nums = re.findall(r'\d+', match.group(1))
                spec.survive = set(int(n) for n in nums)
        
        elif line.startswith('[') and ']' in line:
            # Color definition: [state] = color
            match = re.search(r'\[(\d+)\]\s*=\s*(.+)', line)
            if match:
                state = int(match.group(1))
                color_str = match.group(2).strip()
                spec.colors[state] = parse_color(color_str)
    
    return spec


class RuleTableEngine:
    """
    Engine that executes rules from .rule or .table files.
    
    Supports multi-state rules and custom neighborhoods.
    """
    
    name = "ruletable"
    display_mode = "image"
    
    def __init__(
        self,
        width: int,
        height: int,
        wrap: bool = False,
        rule_file: str | Path | None = None,
    ):
        self.width = width
        self.height = height
        self.wrap = wrap
        
        if rule_file:
            self.spec = load_ruletable_file(rule_file)
        else:
            self.spec = RuleTableSpec(name="conway")
            self.spec.birth = {3}
            self.spec.survive = {2, 3}
        
        self.states = self.spec.states
        # Initialize board with state 0 (dead)
        self.board = np.zeros((height, width), dtype=np.uint8)
        
        # Precompute neighbor counts for all possible values
        self._setup_transition_cache()
    
    def _setup_transition_cache(self) -> None:
        """Pre-compute transition table for performance."""
        # Maximum possible neighbors (9 for Moore, 5 for von Neumann radius 1)
        max_neighbors = 9 if self.spec.neighborhood == 'moore' else 5
        if self.spec.radius > 1:
            max_neighbors = (2 * self.spec.radius + 1) ** 2
        
        self._transition = np.zeros((max_neighbors + 1, self.states), dtype=np.uint8)
        
        if self.spec.is_ruletable:
            # Use explicit transitions from rule file
            for (current, neighbors), output in self.spec.transitions.items():
                if neighbors <= max_neighbors and output < self.states:
                    self._transition[neighbors, current] = output
        else:
            # Standard B/S rules
            for neighbors in range(max_neighbors + 1):
                for current in range(self.states):
                    if current == 0:
                        # Dead cell becomes alive if birth condition met
                        self._transition[neighbors, current] = 1 if neighbors in self.spec.birth else 0
                    else:
                        # Alive cell survives if survive condition met
                        self._transition[neighbors, current] = 1 if neighbors in self.spec.survive else 0
    
    def clear(self) -> None:
        self.board.fill(0)
    
    def randomize(self, density: float, seed: int) -> None:
        self.clear()
        rng = np.random.default_rng(seed)
        self.board = (rng.random((self.height, self.width)) < density).astype(np.uint8)
    
    def seed_pattern(self, pattern: Iterator[tuple[int, int]], anchor_x: int = 0, anchor_y: int = 0) -> None:
        self.clear()
        for dx, dy in pattern:
            x = anchor_x + dx
            y = anchor_y + dy
            if self.wrap:
                self.board[y % self.height, x % self.width] = 1
            elif 0 <= x < self.width and 0 <= y < self.height:
                self.board[y, x] = 1
    
    def step(self) -> None:
        """Execute one simulation step."""
        h, w = self.height, self.width
        
        # Count neighbors using convolution-like approach
        # For efficiency, do manual neighbor counting
        new_board = np.zeros_like(self.board)
        
        # Define neighbor offsets based on neighborhood type
        if self.spec.neighborhood == 'vonneumann':
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if self.spec.radius > 1:
                # Add more radius
                for r in range(2, self.spec.radius + 1):
                    offsets.extend([(-r, 0), (r, 0), (0, -r), (0, r)])
        else:  # moore
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            if self.spec.radius > 1:
                for dy in range(-self.spec.radius, self.spec.radius + 1):
                    for dx in range(-self.spec.radius, self.spec.radius + 1):
                        if dy != 0 or dx != 0:
                            offsets.append((dy, dx))
        
        # Count neighbors
        neighbor_count = np.zeros((h, w), dtype=np.int16)
        
        for dy, dx in offsets:
            if self.wrap:
                ny = (np.arange(h) + dy) % h
                nx = (np.arange(w) + dx) % w
            else:
                ny = np.arange(h) + dy
                nx = np.arange(w) + dx
                # Mask for valid positions
                valid_y = (ny >= 0) & (ny < h)
                valid_x = (nx >= 0) & (nx < w)
            
            # Add neighbor contributions
            if self.wrap:
                neighbor_count += self.board[np.ix_(ny, nx)]
            else:
                for y_idx, y in enumerate(ny):
                    for x_idx, x in enumerate(nx):
                        if 0 <= y < h and 0 <= x < w:
                            neighbor_count[y_idx, x_idx] += self.board[y, x]
        
        # Apply transition table
        np.clip(neighbor_count, 0, self._transition.shape[0] - 1, out=neighbor_count)
        new_board = self._transition[neighbor_count, self.board]
        
        self.board = new_board
    
    def alive_count(self) -> int:
        return int(np.sum(self.board > 0))
    
    def alive_points(self) -> np.ndarray:
        points = np.argwhere(self.board > 0)
        if points.size == 0:
            return np.empty((0, 2), dtype=float)
        return np.column_stack((points[:, 1], points[:, 0])).astype(float)
    
    def board_view(self) -> np.ndarray:
        return self.board.copy()


def list_available_rules(rules_dir: str | Path = "rules") -> list[str]:
    """List all available .rule and .table files in the rules directory."""
    rules_dir = Path(rules_dir)
    if not rules_dir.exists():
        return []
    
    rules = []
    for ext in ['*.rule', '*.table', '*.rul']:
        rules.extend(f.name for f in rules_dir.glob(ext))
    return sorted(rules)