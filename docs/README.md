# Game of Life

A modern, high-performance Conway's Game of Life implementation with multiple simulation backends and RL-adaptive stepping.

![Screenshots](screenshots/gameoflife_gui.png)

## Features

- **Multiple Backends**: Choose from QuickLife, HashLife, Numba, Torch, RL-adaptive, and more
- **Multiple Rule Families**: Standard Life, Generations, Larger-than-Life, von Neumann
- **RL-Adaptive Stepping**: AI learns optimal multi-generation jump sizes
- **Rich CLI**: Interactive GUI and terminal TUI modes
- **Benchmarking**: Built-in performance testing

## Quick Start

```bash
# Install
uv pip install -e .

# Run with auto backend selection
gameoflife

# Run specific backend
gameoflife --backend quicklife --pattern glider-gun
```

## Installation

### Basic
```bash
uv pip install -e .
```

### With Torch Support (for RL backend)
```bash
uv pip install -e '.[torch]'
```

## Usage

### Basic Commands

```bash
# Random pattern
gameoflife --pattern random --density 0.2

# Load a pattern
gameoflife --pattern glider-gun

# TUI mode (terminal)
gameoflife --ui tui --backend quicklife

# GUI mode (default)
gameoflife --ui gui
```

### Controls (TUI)

| Key | Action |
|-----|--------|
| `Space` | Pause/Resume |
| `n` | Single step (when paused) |
| `r` | Reset to random |
| `g` | Load Gosper glider gun |
| `c` | Clear board |
| `w` | Toggle wrapping |
| `q` | Quit |

### Controls (GUI)

- Same as TUI plus mouse interactions

## Backends

### Auto Backend (`--backend auto`)
Automatically profiles and selects the fastest backend for your hardware.

### QuickLife (`--backend quicklife`)
Fast dense grid update engine. Good balance of speed and simplicity.

### HashLife (`--backend hashlife`)
Memoized simulation for very fast long-run patterns.

### HashLife-Tree (`--backend hashlife-tree`)
Quadtree-based implementation with adaptive boundary handling.

### Numba (`--backend numba`)
JIT-compiled dense kernel. Requires Numba installed.

### Torch (`--backend torch`)
CPU convolution backend using PyTorch.

### RL (`--backend rl`)
Adaptive jump backend using trained RL agent. Requires trained model.

### JVN (`--backend jvn`)
von Neumann neighborhood (B/S rules).

### Generations (`--backend generations`)
Multi-state rules like Brian's Brain.

### LargerLife (`--backend largerlife`)
Larger-than-Life family rules.

## Rules

### Standard Life-like Rules

```bash
# Standard Conway's Life
gameoflife --backend jvn --rule B3/S23

# HighLife (has replicators)
gameoflife --backend jvn --rule B36/S23

# Day & Night
gameoflife --backend jvn --rule B3678/S34678
```

### Generations Rules

```bash
# Brian's Brain
gameoflife --backend generations --rule B2/S/C3

# 4-state variant
gameoflife --backend generations --rule B2/S/C4
```

### Larger-than-Life Rules

```bash
# Radius 2
gameoflife --backend largerlife --rule R2,B3/S23

# With range
gameoflife --backend largerlife --rule R3,B34-45/S34-56

# Presets
gameoflife --backend largerlife --rule-preset bosco
gameoflife --backend largerlife --rule-preset coral
```

### Custom JSON Rules

```bash
gameoflife --backend ruleloader --rule-file rules/example.rule.json
```

See `rules/` directory for more rule examples.

## Benchmarking

```bash
# Single backend
gameoflife --backend quicklife --benchmark-steps 300

# All backends
gameoflife --benchmark-all
```

## RL Training

Train your own adaptive jump agent:

```bash
gameoflife-train-rl --config config.yaml
```

This creates:
- `gameoflife/backends/rl_agent.pt` - Runtime policy
- `gameoflife/backends/rl_forward.pt` - Forward model

Evaluate:
```bash
gameoflife-rl-eval --config config.yaml --checkpoint runs/.../last.ckpt --out eval.gif
```

## Configuration

Edit `config.yaml` for RL training parameters:
- `train.num_envs` - Parallel environments
- `train.num_workers` - Dataloader workers
- `train.matmul_precision` - Tensor Core optimization
- `train.quantize_policy` - Policy quantization

## Project Structure

```
GameOfLife/
├── gameoflife/          # Main package
│   ├── backends/        # Simulation engines
│   ├── cli.py          # CLI interface
│   └── rl/             # RL components
├── rules/               # Rule definitions
├── config.yaml          # Training config
└── setup.py            # Package config
```

## License

MIT