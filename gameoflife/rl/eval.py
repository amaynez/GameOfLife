from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import animation

from gameoflife.rl.env import ACTION_SET, AdaptiveJumpEnv
from gameoflife.rl.train import AdaptiveJumpLightning


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_eval(cfg: dict, checkpoint: str, out_path: str, steps: int) -> None:
    model = AdaptiveJumpLightning.load_from_checkpoint(checkpoint, cfg=cfg, map_location="cpu")
    model.eval()

    env = AdaptiveJumpEnv(**cfg["env"])
    obs, _ = env.reset(seed=cfg["env"].get("seed", 42))

    boards: list[np.ndarray] = []
    jumps: list[int] = []
    errors: list[float] = []

    for _ in range(steps):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        patch, stats = model._split_obs(obs_t)
        logits, _ = model.policy(patch, stats)
        action_idx = int(torch.argmax(logits, dim=1).item())

        obs, _, done, _, info = env.step(action_idx)
        boards.append(env.render())
        jumps.append(int(ACTION_SET[action_idx]))
        errors.append(float(info["error"]))

        if done:
            obs, _ = env.reset()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(boards[0], cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    txt = ax.text(0.01, 1.02, "", transform=ax.transAxes)
    ax.set_title("Adaptive Jump Policy (overlay: chosen jump)")
    ax.set_xticks([])
    ax.set_yticks([])

    def _update(i: int):
        im.set_data(boards[i])
        txt.set_text(f"frame={i} jump={jumps[i]} error={errors[i]:.4f}")
        return im, txt

    ani = animation.FuncAnimation(fig, _update, frames=len(boards), interval=120, blit=False)
    if out_path.endswith(".gif"):
        ani.save(out_path, writer="pillow", fps=8)
    else:
        ani.save(out_path, fps=8)

    plt.close(fig)
    print(f"saved_eval={out_path}")
    print(f"mean_jump={np.mean(jumps):.2f} mean_error={np.mean(errors):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate adaptive jump policy and export visualization")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="adaptive_jump_eval.gif")
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    run_eval(cfg=cfg, checkpoint=args.checkpoint, out_path=args.out, steps=max(1, args.steps))


if __name__ == "__main__":
    main()
