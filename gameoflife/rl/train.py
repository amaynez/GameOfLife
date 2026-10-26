from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

from gameoflife.rl.env import ACTION_SET, AdaptiveJumpEnv
from gameoflife.rl.models import ForwardModelCNN, JumpPolicyValueNet, ScriptedJumpPolicy


class _DummyDataset(Dataset):
    def __init__(self, length: int) -> None:
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> int:
        return idx


class AdaptiveJumpLightning(pl.LightningModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        env_cfg = cfg["env"]
        train_cfg = cfg["train"]

        self.patch_size = int(env_cfg.get("patch_size", 32))
        self.num_envs = max(1, int(train_cfg.get("num_envs", 8)))
        base_seed = int(env_cfg.get("seed", 42))

        self.envs: list[AdaptiveJumpEnv] = []
        for i in range(self.num_envs):
            c = dict(env_cfg)
            c["seed"] = base_seed + i
            self.envs.append(AdaptiveJumpEnv(**c))

        self.policy = JumpPolicyValueNet(
            patch_size=self.patch_size,
            n_actions=len(ACTION_SET),
            stats_dim=3,
        )
        self.forward_model = ForwardModelCNN(
            n_actions=len(ACTION_SET),
            hidden_channels=int(train_cfg.get("forward_hidden_channels", 24)),
        )

        self.rollout_steps = int(train_cfg.get("rollout_steps", 128))
        self.ppo_epochs = int(train_cfg.get("ppo_epochs", 4))
        self.minibatch_size = int(train_cfg.get("minibatch_size", 128))
        self.forward_warmup_epochs = int(train_cfg.get("forward_warmup_epochs", 5))
        self.forward_batches_per_epoch = int(train_cfg.get("forward_batches_per_epoch", 8))

        self.gamma = float(train_cfg.get("gamma", 0.99))
        self.gae_lambda = float(train_cfg.get("gae_lambda", 0.95))
        self.clip_eps = float(train_cfg.get("clip_eps", 0.2))
        self.entropy_coef = float(train_cfg.get("entropy_coef", 0.01))
        self.value_coef = float(train_cfg.get("value_coef", 0.5))
        self.forward_coef = float(train_cfg.get("forward_coef", 1.0))
        self.accumulate_steps = max(1, int(train_cfg.get("accumulate_grad_batches", 1)))

        self.lambda_start = float(train_cfg.get("lambda_start", 0.1))
        self.lambda_end = float(train_cfg.get("lambda_end", 0.3))

        initial_obs = []
        for i, env in enumerate(self.envs):
            obs_i, _ = env.reset(seed=base_seed + i)
            initial_obs.append(obs_i)
        self._obs = np.asarray(initial_obs, dtype=np.float32)

        self.automatic_optimization = False
        self._best_reward = float("-inf")
        self._best_policy_state: dict[str, torch.Tensor] | None = None

    def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.patch_size
        patch = obs[:, : p * p].reshape(-1, 1, p, p)
        stats = obs[:, p * p :]
        return patch, stats

    def _set_lambda(self) -> None:
        if self.current_epoch < self.forward_warmup_epochs:
            lam = self.lambda_start
        else:
            policy_epoch = self.current_epoch - self.forward_warmup_epochs
            policy_total = max(1, int(self.cfg["train"]["epochs"]) - self.forward_warmup_epochs)
            frac = min(1.0, float(policy_epoch) / float(policy_total))
            lam = self.lambda_start + (self.lambda_end - self.lambda_start) * frac
        for env in self.envs:
            env.set_lambda_penalty(lam)
        self.log("curriculum/lambda_penalty", lam, on_step=False, on_epoch=True, prog_bar=True)

    def _collect_forward_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_inputs = []
        action_indices = []
        targets = []

        for _ in range(self.rollout_steps):
            actions = np.random.randint(0, len(ACTION_SET), size=self.num_envs)
            for i, env in enumerate(self.envs):
                obs_i, _, done, _, info = env.step_with_prediction(int(actions[i]), predicted=None)
                model_inputs.append(info["model_input"].astype(np.float32))
                targets.append(info["model_target"].astype(np.float32))
                action_indices.append(int(actions[i]))
                if done:
                    obs_i, _ = env.reset()
                self._obs[i] = obs_i

        x = torch.tensor(np.asarray(model_inputs), dtype=torch.float32, device=self.device).unsqueeze(1)
        a = torch.tensor(action_indices, dtype=torch.long, device=self.device)
        y = torch.tensor(np.asarray(targets), dtype=torch.float32, device=self.device).unsqueeze(1)
        return x, a, y

    @torch.no_grad()
    def _predict_forward_batch(self, model_inputs: np.ndarray, action_indices: torch.Tensor) -> np.ndarray:
        board_t = torch.from_numpy(model_inputs).to(device=self.device, dtype=torch.float32).unsqueeze(1)
        logits = self.forward_model(board_t, action_indices.to(self.device))
        return torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)

    def _collect_ppo_rollout(self) -> dict[str, torch.Tensor | float]:
        n_env = self.num_envs
        obs_dim = self._obs.shape[1]
        side = self.envs[0].model_view_size

        obs_buf = np.zeros((self.rollout_steps, n_env, obs_dim), dtype=np.float32)
        actions_buf = np.zeros((self.rollout_steps, n_env), dtype=np.int64)
        logp_buf = np.zeros((self.rollout_steps, n_env), dtype=np.float32)
        values_buf = np.zeros((self.rollout_steps, n_env), dtype=np.float32)
        rewards_buf = np.zeros((self.rollout_steps, n_env), dtype=np.float32)
        dones_buf = np.zeros((self.rollout_steps, n_env), dtype=np.float32)
        model_input_buf = np.zeros((self.rollout_steps, n_env, side, side), dtype=np.float32)
        model_target_buf = np.zeros_like(model_input_buf)

        jump_vals = []
        error_vals = []

        # FIX 1: Entire rollout collection must run under no_grad.
        # Without this, every policy forward pass during data collection
        # builds a computation graph — 1024 graphs for rollout_steps=128,
        # num_envs=8 — blowing up VRAM and making rollout 3-5x slower.
        with torch.no_grad():
            for t in range(self.rollout_steps):
                obs_t = torch.from_numpy(self._obs).to(self.device, dtype=torch.float32)
                patch_t, stats_t = self._split_obs(obs_t)

                logits, values = self.policy(patch_t, stats_t)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logp = dist.log_prob(actions)

                model_inputs = np.asarray([env.current_model_input() for env in self.envs], dtype=np.float32)
                preds = self._predict_forward_batch(model_inputs, actions)

                next_obs = np.zeros_like(self._obs)
                for i, env in enumerate(self.envs):
                    obs_i, reward_i, done_i, _, info_i = env.step_with_prediction(int(actions[i].item()), preds[i])
                    obs_buf[t, i] = self._obs[i]
                    actions_buf[t, i] = int(actions[i].item())
                    logp_buf[t, i] = float(logp[i].item())
                    values_buf[t, i] = float(values[i].item())
                    rewards_buf[t, i] = float(reward_i)
                    dones_buf[t, i] = float(done_i)
                    model_input_buf[t, i] = info_i["model_input"]
                    model_target_buf[t, i] = info_i["model_target"]

                    jump_vals.append(float(info_i["jump"]))
                    error_vals.append(float(info_i["error"]))

                    if done_i:
                        obs_i, _ = env.reset()
                    next_obs[i] = obs_i

                self._obs = next_obs

            next_obs_t = torch.from_numpy(self._obs).to(self.device, dtype=torch.float32)
            next_patch_t, next_stats_t = self._split_obs(next_obs_t)
            _, next_values = self.policy(next_patch_t, next_stats_t)
            next_values_np = next_values.detach().cpu().numpy().astype(np.float32)

        adv = np.zeros((self.rollout_steps, n_env), dtype=np.float32)
        ret = np.zeros((self.rollout_steps, n_env), dtype=np.float32)
        gae = np.zeros(n_env, dtype=np.float32)

        for t in reversed(range(self.rollout_steps)):
            not_done = 1.0 - dones_buf[t]
            v_next = next_values_np if t == self.rollout_steps - 1 else values_buf[t + 1]
            delta = rewards_buf[t] + (self.gamma * v_next * not_done) - values_buf[t]
            gae = delta + (self.gamma * self.gae_lambda * not_done * gae)
            adv[t] = gae
            ret[t] = gae + values_buf[t]

        flat_obs = obs_buf.reshape(-1, obs_dim)
        flat_actions = actions_buf.reshape(-1)
        flat_logp = logp_buf.reshape(-1)
        flat_ret = ret.reshape(-1)
        flat_adv = adv.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        obs = torch.tensor(flat_obs, dtype=torch.float32, device=self.device)
        patch, stats = self._split_obs(obs)

        return {
            "obs": obs,
            "patch": patch,
            "stats": stats,
            "actions": torch.tensor(flat_actions, dtype=torch.long, device=self.device),
            "old_logp": torch.tensor(flat_logp, dtype=torch.float32, device=self.device),
            "returns": torch.tensor(flat_ret, dtype=torch.float32, device=self.device),
            "advantages": torch.tensor(flat_adv, dtype=torch.float32, device=self.device),
            "model_input": torch.tensor(
                model_input_buf.reshape(-1, side, side), dtype=torch.float32, device=self.device
            ).unsqueeze(1),
            "model_target": torch.tensor(
                model_target_buf.reshape(-1, side, side), dtype=torch.float32, device=self.device
            ).unsqueeze(1),
            "mean_jump": float(np.mean(jump_vals)) if jump_vals else 0.0,
            "mean_error": float(np.mean(error_vals)) if error_vals else 0.0,
            "mean_reward": float(np.mean(rewards_buf)),
        }

    def training_step(self, batch, batch_idx):
        del batch, batch_idx
        self._set_lambda()
        opt = self.optimizers()

        if self.current_epoch < self.forward_warmup_epochs:
            losses = []
            opt.zero_grad()
            accum_i = 0
            for _ in range(self.forward_batches_per_epoch):
                x, a, y = self._collect_forward_batch()
                pred = self.forward_model(x, a)
                loss = nn.functional.binary_cross_entropy_with_logits(pred, y) / float(self.accumulate_steps)
                self.manual_backward(loss)
                accum_i += 1
                if accum_i % self.accumulate_steps == 0:
                    nn.utils.clip_grad_norm_(self.forward_model.parameters(), max_norm=1.0)
                    opt.step()
                    opt.zero_grad()
                losses.append(float(loss.detach().item()))
            if accum_i % self.accumulate_steps != 0:
                nn.utils.clip_grad_norm_(self.forward_model.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad()
            self.log("phase", 0.0, on_step=False, on_epoch=True)
            self.log(
                "loss/forward_warmup",
                float(np.mean(losses) * float(self.accumulate_steps)),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            return

        rollout = self._collect_ppo_rollout()

        all_policy_loss = 0.0
        all_value_loss = 0.0
        all_entropy = 0.0
        all_forward_loss = 0.0

        n = int(rollout["obs"].shape[0])
        opt.zero_grad()
        accum_i = 0
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.minibatch_size):
                idx = perm[start : start + self.minibatch_size]

                patch = rollout["patch"][idx]
                stats = rollout["stats"][idx]
                actions = rollout["actions"][idx]
                old_logp = rollout["old_logp"][idx]
                advantages = rollout["advantages"][idx]
                returns = rollout["returns"][idx]

                logits, values = self.policy(patch, stats)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - old_logp)
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
                value_loss = nn.functional.mse_loss(values, returns)

                fm_logits = self.forward_model(rollout["model_input"][idx], actions)
                forward_loss = nn.functional.binary_cross_entropy_with_logits(
                    fm_logits, rollout["model_target"][idx]
                )

                loss = (
                    policy_loss
                    + (self.value_coef * value_loss)
                    - (self.entropy_coef * entropy)
                    + (self.forward_coef * forward_loss)
                ) / float(self.accumulate_steps)
                self.manual_backward(loss)
                accum_i += 1
                if accum_i % self.accumulate_steps == 0:
                    nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    opt.step()
                    opt.zero_grad()

                all_policy_loss += float(policy_loss.detach().item())
                all_value_loss += float(value_loss.detach().item())
                all_entropy += float(entropy.detach().item())
                all_forward_loss += float(forward_loss.detach().item())

        if accum_i % self.accumulate_steps != 0:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()

        self.log("phase", 1.0, on_step=False, on_epoch=True)
        self.log("rollout/mean_jump", float(rollout["mean_jump"]), on_step=False, on_epoch=True, prog_bar=True)
        self.log("rollout/mean_error", float(rollout["mean_error"]), on_step=False, on_epoch=True, prog_bar=True)
        self.log("rollout/reward", float(rollout["mean_reward"]), on_step=False, on_epoch=True, prog_bar=True)
        if float(rollout["mean_reward"]) > self._best_reward:
            self._best_reward = float(rollout["mean_reward"])
            self._best_policy_state = {k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()}

        self.log("loss/policy", all_policy_loss, on_step=False, on_epoch=True)
        self.log("loss/value", all_value_loss, on_step=False, on_epoch=True)
        self.log("loss/entropy", all_entropy, on_step=False, on_epoch=True)
        self.log("loss/forward", all_forward_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        lr = float(self.cfg["train"].get("learning_rate", 3e-4))
        return torch.optim.Adam(self.parameters(), lr=lr)

    def train_dataloader(self):
        updates = int(self.cfg["train"].get("updates_per_epoch", 4))
        requested_workers = self.cfg["train"].get("num_workers")
        if requested_workers is None:
            cpu = os.cpu_count() or 1
            requested_workers = min(8, max(1, cpu - 1))
        num_workers = max(0, int(requested_workers))
        pin_memory = bool(self.cfg["train"].get("pin_memory", torch.cuda.is_available()))
        persistent_workers = bool(self.cfg["train"].get("persistent_workers", True)) and num_workers > 0
        loader_kwargs = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        }
        if num_workers > 0:
            prefetch_factor = max(1, int(self.cfg["train"].get("prefetch_factor", 2)))
            loader_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(_DummyDataset(updates), **loader_kwargs)

    def export_torchscript(
        self,
        agent_path: str,
        forward_path: str | None = None,
        quantize_policy: bool = True,
    ) -> tuple[int, int]:
        self.eval()
        if self._best_policy_state is not None:
            self.policy.load_state_dict(self._best_policy_state)

        policy_adapter = ScriptedJumpPolicy(self.policy, patch_size=self.patch_size).cpu().eval()

        if quantize_policy:
            try:
                try:
                    from torch.ao.quantization import quantize_dynamic as ao_quantize_dynamic
                    policy_adapter = ao_quantize_dynamic(policy_adapter, {nn.Linear}, dtype=torch.qint8)
                except ImportError:
                    policy_adapter = torch.quantization.quantize_dynamic(
                        policy_adapter, {nn.Linear}, dtype=torch.qint8
                    )
            except Exception as exc:
                print(f"[warn] policy quantization failed ({exc}); exporting non-quantized policy.")

        # FIX 2: torch.jit.script fails on dynamically quantized models because
        # quantized ops are not scriptable. Use torch.jit.trace with a dummy
        # input instead — this works for both quantized and non-quantized cases.
        dummy_obs = torch.zeros(1, self.patch_size * self.patch_size + 3)
        scripted_policy = torch.jit.trace(policy_adapter, dummy_obs)
        scripted_policy.save(agent_path)

        forward_bytes = 0
        if forward_path:
            # ForwardModelCNN has no quantization applied, so script works fine.
            scripted_forward = torch.jit.script(self.forward_model.cpu().eval())
            scripted_forward.save(forward_path)
            forward_bytes = Path(forward_path).stat().st_size

        agent_bytes = Path(agent_path).stat().st_size
        return agent_bytes, forward_bytes


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_cfg() -> dict:
    return {
        "env": {
            "width": 256,
            "height": 256,
            "density": 0.2,
            "seed": 42,
            "backend": "hashlife-tree",
            "wrap": False,
            "max_generations": 500,
            "patch_size": 32,
            "model_view_size": 64,
            "lambda_penalty": 0.1,
        },
        "train": {
            "seed": 42,
            "epochs": 24,
            "updates_per_epoch": 4,
            "num_envs": 8,
            "rollout_steps": 128,
            "ppo_epochs": 4,
            "minibatch_size": 128,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "forward_coef": 1.0,
            "forward_warmup_epochs": 5,
            "forward_batches_per_epoch": 8,
            "forward_hidden_channels": 24,
            "lambda_start": 0.1,
            "lambda_end": 0.3,
            "quantize_policy": True,
            "num_workers": 7,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "matmul_precision": "high",
            "accumulate_grad_batches": 1,
            "log_dir": "runs",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL adaptive step jumper and export TorchScript agent")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_cfg(str(cfg_path)) if cfg_path.exists() else _default_cfg()
    matmul_precision = str(cfg.get("train", {}).get("matmul_precision", "high")).lower()
    if matmul_precision in {"high", "medium"}:
        try:
            torch.set_float32_matmul_precision(matmul_precision)
            print(f"matmul_precision={matmul_precision}")
        except Exception as exc:
            print(f"[warn] could not set matmul precision ({exc})")

    pl.seed_everything(int(cfg["train"].get("seed", 42)), workers=True)
    model = AdaptiveJumpLightning(cfg)

    logger = pl.loggers.TensorBoardLogger(save_dir=cfg["train"].get("log_dir", "runs"), name="train-rl")

    trainer = pl.Trainer(
        max_epochs=int(cfg["train"].get("epochs", 24)),
        logger=logger,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        default_root_dir=cfg["train"].get("log_dir", "runs"),
    )
    trainer.fit(model)

    out_agent = Path("gameoflife/backends/rl_agent.pt")
    out_forward = Path("gameoflife/backends/rl_forward.pt")
    out_agent.parent.mkdir(parents=True, exist_ok=True)

    agent_bytes, forward_bytes = model.export_torchscript(
        str(out_agent),
        str(out_forward),
        quantize_policy=bool(cfg["train"].get("quantize_policy", True)),
    )
    total = agent_bytes + forward_bytes

    print(f"exported_agent={out_agent} bytes={agent_bytes}")
    print(f"exported_forward={out_forward} bytes={forward_bytes}")
    print(f"exported_total_bytes={total}")
    if total > (2 * 1024 * 1024):
        print("[warn] exported RL artifacts exceed 2MB target; reduce model sizes or quantize further.")


if __name__ == "__main__":
    main()