# CLI Reference

Complete reference for `gameoflife` command-line interface.

## Usage

```bash
gameoflife [OPTIONS]
```

## Options

### UI Options
| Option | Description | Default |
|--------|-------------|---------|
| `--ui {gui,tui}` | UI mode | gui |

### Backend Options
| Option | Description | Default |
|--------|-------------|---------|
| `--backend {auto,jvn,generations,largerlife,ruleloader,quicklife,hashlife,hashlife-tree,rl,numba,torch}` | Simulation backend | auto |

### Rule Options

#### For jvn backend:
```
--rule RULESTRING    Rule in B#/S# format (e.g., B3/S23)
```

#### For generations backend:
```
--rule RULESTRING    Rule in B#/S#/C# format (e.g., B2/S/C3)
```

#### For largerlife backend:
```
--rule RULESTRING    Rule in R#,B#-#,S#-# format (e.g., R2,B3/S23)
--rule-preset PRESET  Use preset (bosco, coral, nova, storm)
```

#### For ruleloader backend:
```
--rule-file FILE     Path to JSON rule file
```

### Board Options
| Option | Description | Default |
|--------|-------------|---------|
| `--width N` | Board width | 200 |
| `--height N` | Board height | 200 |
| `--wrap` | Enable wrapping | False |

### Pattern Options
| Option | Description | Default |
|--------|-------------|---------|
| `--pattern {glider-gun,random,empty,gosper-glider-gun}` | Initial pattern | random |
| `--density N` | Random density (0.0-1.0) | 0.2 |
| `--seed N` | Random seed | random |

### Simulation Options
| Option | Description | Default |
|--------|-------------|---------|
| `--interval N` | Milliseconds between steps | 50 |
| `--max-generations N` | Stop after N generations | unlimited |

### Performance Options
| Option | Description | Default |
|--------|-------------|---------|
| `--benchmark-steps N` | Run N steps and report time | - |
| `--benchmark-all` | Benchmark all backends | - |
| `--doctor` | Run diagnostics | - |
| `--rl-stats` | Show RL inference latency | - |

### Help
| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--version` | Show version |

## Examples

### Basic Usage
```bash
# Random pattern, default settings
gameoflife

# Glider gun pattern
gameoflife --pattern glider-gun

# Specify board size
gameoflife --width 400 --height 300
```

### Different Backends
```bash
# Auto-select fastest
gameoflife --backend auto

# QuickLife (fastest for most)
gameoflife --backend quicklife

# Numba JIT-compiled
gameoflife --backend numba

# Torch convolution
gameoflife --backend torch
```

### Different Rules
```bash
# Standard Life
gameoflife --backend jvn --rule B3/S23

# HighLife
gameoflife --backend jvn --rule B36/S23

# Brian's Brain
gameoflife --backend generations --rule B2/S/C3

# Larger-than-Life
gameoflife --backend largerlife --rule R3,B34-45/S34-56
```

### Custom Rule Files
```bash
gameoflife --backend ruleloader --rule-file rules/myrule.rule.json
```

### Benchmarking
```bash
# Benchmark specific backend
gameoflife --backend quicklife --benchmark-steps 500

# Benchmark all
gameoflife --benchmark-all

# With custom board size
gameoflife --width 1000 --height 1000 --benchmark-steps 100
```

### Terminal Mode (TUI)
```bash
gameoflife --ui tui --backend quicklife --pattern random
```

## Keyboard Controls (TUI)

| Key | Action |
|-----|--------|
| `Space` | Pause/Resume |
| `n` | Next generation (when paused) |
| `r` | Reset to random |
| `g` | Load Gosper glider gun |
| `c` | Clear board |
| `w` | Toggle wrap |
| `q` | Quit |
| `+/-` | Adjust speed |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GAMEOFLIFE_BACKEND` | Default backend |
| `GAMEOFLIFE_WIDTH` | Default width |
| `GAMEOFLIFE_HEIGHT` | Default height |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error |