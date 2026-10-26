# Installation Guide

## Requirements

- Python 3.10+
- uv (package manager)

## Installation Steps

### 1. Install uv (if not already)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

### 2. Clone the Repository

```bash
git clone https://github.com/Kiy-K/GameOfLife.git
cd GameOfLife
```

### 3. Install Dependencies

#### Basic Installation
```bash
uv pip install -e .
```

#### With Torch Support (for RL backend)
```bash
uv pip install -e '.[torch]'
```

This installs:
- NumPy (for array operations)
- Numba (JIT compilation)
- PyTorch (for RL and convolution backends)
- Gymnasium (RL environment)
- PyTorch Lightning (RL training)
- Matplotlib (GUI)

### 4. Verify Installation

```bash
# Check CLI works
gameoflife --help

# Run doctor check
gameoflife --doctor
```

## Optional Dependencies

### Development
```bash
# Install dev dependencies
uv pip install -e '.[dev]'
# Or all extras
uv pip install -e '.[dev,torch]'
```

## Troubleshooting

### Numba not found
```bash
pip install numba
```

### Torch not found
```bash
pip install torch
```

### GUI not working
Requires a display. For headless servers, use TUI mode:
```bash
gameoflife --ui tui
```

## Uninstallation

```bash
uv pip uninstall gameoflife
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 8GB+ |
| CPU | Any x86_64 | Multi-core for Numba |
| GPU | Optional | CUDA-capable for RL |

## Running Without Installation

```bash
# Run directly from source
python -m gameoflife.cli --help
```