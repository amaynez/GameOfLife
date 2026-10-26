from __future__ import annotations

from pathlib import Path
import time
from typing import Iterable

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None
    TORCH_AVAILABLE = False

from gameoflife.cli import AutoAdaptiveEngine, HashLifeTreeEngine, LifeEngine

ACTION_SET = np.array([1, 2, 4, 8, 16, 32], dtype=np.int32)


class RLBackendEngine(LifeEngine):
    """RL-driven backend that chooses jump size and advances HashLifeTree."""

    name = "rl"
    display_mode = "image"

    def __init__(self, width: int, height: int, wrap: bool = False, agent_path: str | None = None) -> None:
        super().__init__(width=width, height=height, wrap=wrap)
        self._hashlife = HashLifeTreeEngine(width=width, height=height, wrap=wrap)
        self._fallback: AutoAdaptiveEngine | None = None
        self._agent = None
        self._generation = 0
        self._prev_alive = 0
        self._infer_count = 0
        self._infer_avg_ms = 0.0
        self._infer_max_ms = 0.0
        self._infer_last_ms = 0.0
        self._warn_threshold_ms = 5.0
        self._warn_interval = 120
        self._stats_enabled = False

        self._agent_path = Path(agent_path) if agent_path else Path(__file__).with_name("rl_agent.pt")
        self._load_agent_or_fallback()

    def set_stats(self, enabled: bool = True, interval: int | None = None) -> None:
        self._stats_enabled = bool(enabled)
        if interval is not None:
            self._warn_interval = max(1, int(interval))

    def get_inference_stats(self) -> dict[str, float]:
        return {
            "count": float(self._infer_count),
            "avg_ms": float(self._infer_avg_ms),
            "max_ms": float(self._infer_max_ms),
            "last_ms": float(self._infer_last_ms),
        }

    def _load_agent_or_fallback(self) -> None:
        if not TORCH_AVAILABLE:
            print("[warn] torch unavailable; rl backend falling back to auto.")
            self._fallback = AutoAdaptiveEngine(width=self.width, height=self.height, wrap=self.wrap)
            self.name = "rl->auto"
            return

        if not self._agent_path.exists():
            print("RL agent not trained yet. Run: gameoflife-train-rl")
            self._fallback = AutoAdaptiveEngine(width=self.width, height=self.height, wrap=self.wrap)
            self.name = "rl->auto"
            return

        try:
            self._agent = torch.jit.load(str(self._agent_path), map_location="cpu")
            self._agent.eval()
            self._fallback = None
            self.name = "rl"
        except Exception as exc:
            print(f"[warn] failed to load rl agent ({exc}); falling back to auto.")
            self._fallback = AutoAdaptiveEngine(width=self.width, height=self.height, wrap=self.wrap)
            self.name = "rl->auto"

    def _active(self) -> LifeEngine:
        if self._fallback is not None:
            self._fallback.wrap = self.wrap
            return self._fallback
        self._hashlife.wrap = self.wrap
        return self._hashlife

    def clear(self) -> None:
        self._generation = 0
        self._prev_alive = 0
        self._active().clear()

    def randomize(self, density: float, seed: int) -> None:
        self._generation = 0
        self._active().randomize(density, seed)
        self._prev_alive = self.alive_count()

    def seed_pattern(self, pattern: Iterable[tuple[int, int]], anchor_x: int = 0, anchor_y: int = 0) -> None:
        self._generation = 0
        self._active().seed_pattern(pattern, anchor_x, anchor_y)
        self._prev_alive = self.alive_count()

    def _obs(self) -> np.ndarray:
        board = self._hashlife.board_view().astype(np.float32)
        h, w = board.shape
        p = 32
        x0 = max(0, (w - p) // 2)
        y0 = max(0, (h - p) // 2)
        patch = np.zeros((p, p), dtype=np.float32)
        x1 = min(w, x0 + p)
        y1 = min(h, y0 + p)
        patch[: (y1 - y0), : (x1 - x0)] = board[y0:y1, x0:x1]

        alive = float(board.sum())
        total = float(board.size)
        alive_ratio = alive / total if total else 0.0
        alive_delta = (alive - float(self._prev_alive)) / total if total else 0.0
        step_ratio = float(self._generation % 500) / 500.0
        features = np.array([alive_ratio, alive_delta, step_ratio], dtype=np.float32)
        return np.concatenate([patch.reshape(-1), features], dtype=np.float32)

    def step(self) -> int:
        active = self._active()
        if self._fallback is not None:
            delta = active.step()
            if not isinstance(delta, int) or delta <= 0:
                delta = 1
            self._generation += delta
            self._prev_alive = active.alive_count()
            return delta

        obs = self._obs()
        obs_t = torch.from_numpy(obs).to(dtype=torch.float32).unsqueeze(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            action_idx_t = self._agent(obs_t)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        self._infer_count += 1
        self._infer_last_ms = infer_ms
        if infer_ms > self._infer_max_ms:
            self._infer_max_ms = infer_ms
        if self._infer_count == 1:
            self._infer_avg_ms = infer_ms
        else:
            # EWMA to smooth transient spikes while still tracking drift.
            self._infer_avg_ms = (0.95 * self._infer_avg_ms) + (0.05 * infer_ms)
        if self._stats_enabled and self._infer_count % self._warn_interval == 0:
            print(
                (
                    f"[rl-stats] infer_avg_ms={self._infer_avg_ms:.3f} "
                    f"infer_max_ms={self._infer_max_ms:.3f} "
                    f"infer_last_ms={self._infer_last_ms:.3f} "
                    f"count={self._infer_count}"
                )
            )
        if self._infer_count % self._warn_interval == 0 and self._infer_avg_ms > self._warn_threshold_ms:
            print(
                f"[warn] rl inference avg={self._infer_avg_ms:.2f}ms "
                f"(target <{self._warn_threshold_ms:.1f}ms)."
            )
        action_idx = int(action_idx_t.item())
        action_idx = max(0, min(len(ACTION_SET) - 1, action_idx))
        jump = int(ACTION_SET[action_idx])

        delta = self._hashlife.advance(jump)
        if not isinstance(delta, int) or delta <= 0:
            delta = 1
        self._generation += delta
        self._prev_alive = self._hashlife.alive_count()
        return delta

    def advance(self, generations: int) -> int:
        generations = max(0, int(generations))
        advanced = 0
        while advanced < generations:
            advanced += self.step()
        return advanced

    def alive_count(self) -> int:
        return self._active().alive_count()

    def alive_points(self) -> np.ndarray:
        return self._active().alive_points()

    def board_view(self) -> np.ndarray:
        return self._active().board_view()
