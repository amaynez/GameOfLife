from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover - optional at runtime
    raise RuntimeError(
        "RL environment requires gymnasium. Install with: uv pip install -e '.[torch]'"
    ) from exc

from gameoflife.cli import build_engine

ACTION_SET = np.array([1, 2, 4, 8, 16, 32], dtype=np.int32)


@dataclass
class JumpMetrics:
    jump: int
    error: float
    reward: float
    alive: int
    density: float


class AdaptiveJumpEnv(gym.Env):
    """Gym environment for adaptive multi-generation jump selection."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        density: float = 0.2,
        seed: int = 42,
        backend: str = "hashlife-tree",
        wrap: bool = False,
        max_generations: int = 500,
        patch_size: int = 32,
        model_view_size: int = 64,
        lambda_penalty: float = 0.1,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.backend = str(backend)
        self.wrap = bool(wrap)
        self.initial_density = float(density)
        self.initial_seed = int(seed)
        self.max_generations = int(max_generations)
        self.patch_size = int(patch_size)
        self.model_view_size = int(model_view_size)
        self.lambda_penalty = float(lambda_penalty)

        self.action_space = spaces.Discrete(len(ACTION_SET))
        obs_dim = (self.patch_size * self.patch_size) + 3
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.engine = build_engine(
            backend=self.backend,
            width=self.width,
            height=self.height,
            wrap=self.wrap,
            rule="B3/S23",
            rule_file=None,
            rule_preset=None,
        )

        self.generation = 0
        self.last_alive = 0
        self.last_jump = 1

        self._predictor: Callable[[np.ndarray, int], np.ndarray] | None = None
        self.last_metrics = JumpMetrics(jump=1, error=0.0, reward=0.0, alive=0, density=0.0)

    def set_predictor(self, predictor: Callable[[np.ndarray, int], np.ndarray] | None) -> None:
        self._predictor = predictor

    def set_lambda_penalty(self, value: float) -> None:
        self.lambda_penalty = float(value)

    def _board(self) -> np.ndarray:
        return self.engine.board_view().astype(np.float32)

    def _center_patch(self, board: np.ndarray) -> np.ndarray:
        h, w = board.shape
        p = self.patch_size
        x0 = max(0, (w - p) // 2)
        y0 = max(0, (h - p) // 2)
        x1 = min(w, x0 + p)
        y1 = min(h, y0 + p)
        patch = np.zeros((p, p), dtype=np.float32)
        patch[: (y1 - y0), : (x1 - x0)] = board[y0:y1, x0:x1]
        return patch

    def _downsample(self, board: np.ndarray, side: int) -> np.ndarray:
        if board.shape == (side, side):
            return board.astype(np.float32)
        ys = np.linspace(0, board.shape[0] - 1, side, dtype=np.int32)
        xs = np.linspace(0, board.shape[1] - 1, side, dtype=np.int32)
        return board[np.ix_(ys, xs)].astype(np.float32)

    def _obs(self, board: np.ndarray) -> np.ndarray:
        patch = self._center_patch(board)
        alive = float(board.sum())
        total = float(board.size)
        alive_ratio = alive / total if total else 0.0
        alive_delta = (alive - float(self.last_alive)) / total if total else 0.0
        step_ratio = float(self.generation) / float(max(1, self.max_generations))
        features = np.array([alive_ratio, alive_delta, step_ratio], dtype=np.float32)
        return np.concatenate([patch.reshape(-1), features], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.initial_seed = int(seed)

        density = self.initial_density
        if options and "density" in options:
            density = float(options["density"])

        self.engine.randomize(density, self.initial_seed)
        self.generation = 0
        board = self._board()
        self.last_alive = int(board.sum())
        self.last_jump = 1
        obs = self._obs(board)
        info = {"generation": self.generation, "alive": self.last_alive}
        return obs, info

    def current_model_input(self) -> np.ndarray:
        return self._downsample(self._board(), self.model_view_size)

    def step(self, action: int):
        return self._step_impl(action=action, predicted=None, use_predictor=True)

    def step_with_prediction(self, action: int, predicted: np.ndarray | None):
        return self._step_impl(action=action, predicted=predicted, use_predictor=False)

    def _step_impl(self, action: int, predicted: np.ndarray | None, use_predictor: bool):
        action_idx = int(action)
        jump = int(ACTION_SET[action_idx])

        model_in = self.current_model_input()

        pred = predicted
        if use_predictor and pred is None and self._predictor is not None:
            pred = self._predictor(model_in, jump)

        advanced = self.engine.advance(jump)
        self.generation += int(advanced)

        board_after = self._board()
        target = self._downsample(board_after, self.model_view_size)

        if pred is None:
            error = 0.0
        else:
            error = float(np.mean(np.abs(pred.astype(np.float32) - target.astype(np.float32))))

        reward = float(jump - (self.lambda_penalty * error))

        self.last_alive = int(board_after.sum())
        self.last_jump = jump

        done = self.generation >= self.max_generations
        obs = self._obs(board_after)

        density = float(self.last_alive) / float(board_after.size)
        self.last_metrics = JumpMetrics(
            jump=jump,
            error=error,
            reward=reward,
            alive=self.last_alive,
            density=density,
        )

        info = {
            "generation": self.generation,
            "jump": jump,
            "error": error,
            "alive": self.last_alive,
            "density": density,
            "model_input": model_in,
            "model_target": target,
            "action_index": action_idx,
        }
        return obs, reward, done, False, info

    def render(self):
        return self._board()
