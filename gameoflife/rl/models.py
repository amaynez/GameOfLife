from __future__ import annotations

import torch
from torch import nn


class ForwardModelCNN(nn.Module):
    def __init__(self, n_actions: int = 6, hidden_channels: int = 24) -> None:
        super().__init__()
        self.action_embed = nn.Embedding(n_actions, 4)
        self.net = nn.Sequential(
            nn.Conv2d(1 + 4, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, board: torch.Tensor, action_index: torch.Tensor) -> torch.Tensor:
        emb = self.action_embed(action_index)
        emb_map = emb[:, :, None, None].expand(-1, -1, board.shape[-2], board.shape[-1])
        x = torch.cat([board, emb_map], dim=1)
        return self.net(x)


class JumpPolicyValueNet(nn.Module):
    def __init__(self, patch_size: int = 32, n_actions: int = 6, stats_dim: int = 3) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        encoded_side = self.patch_size // 8
        hidden_in = (24 * encoded_side * encoded_side) + stats_dim

        self.trunk = nn.Sequential(
            nn.Linear(hidden_in, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, n_actions)
        self.value_head = nn.Linear(64, 1)

    def forward(self, patch: torch.Tensor, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(patch)
        x = x.flatten(start_dim=1)
        x = torch.cat([x, stats], dim=1)
        h = self.trunk(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


class ScriptedJumpPolicy(nn.Module):
    """Small TorchScript-exportable adapter: obs_flat -> action_index."""

    def __init__(self, policy: JumpPolicyValueNet, patch_size: int = 32) -> None:
        super().__init__()
        self.policy = policy
        self.patch_size = int(patch_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, patch_size*patch_size + 3]
        p2 = self.patch_size * self.patch_size
        patch = obs[:, :p2].reshape(-1, 1, self.patch_size, self.patch_size)
        stats = obs[:, p2:]
        logits, _ = self.policy(patch, stats)
        return torch.argmax(logits, dim=1)
