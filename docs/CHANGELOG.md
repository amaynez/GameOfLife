# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Added `docs/` directory with comprehensive documentation
  - `README.md` - Main documentation
  - `INSTALL.md` - Installation guide
  - `RULES.md` - Rules reference
  - `CLI.md` - CLI reference
- Added new rules in JSON format:
  - `rules/day_and_night.rule.json`
  - `rules/highlife.rule.json`
  - `rules/morley.rule.json`
  - `rules/seeds.rule.json`
  - `rules/brians_brain.rule.json`
  - `rules/serviensaver.rule.json`
  - `rules/replicator.rule.json`
- Added `rules/README.md` with usage guide

### Fixed
- Various rule file formats standardized

## [1.0.0] - 2024-10-28

### Added
- Modernized Matplotlib window-title handling
- Backend architecture for adaptive workload selection
- Dense vectorized stepping (NumPy)
- Dense JIT stepping (Numba)
- Interactive controls and configurable CLI
- Multiple simulation backends:
  - `jvn` - von Neumann neighborhood
  - `generations` - Multi-state rules
  - `largerlife` - Larger-than-Life rules
  - `ruleloader` - JSON rule files
  - `quicklife` - Fast dense engine
  - `hashlife` - Memoized transitions
  - `hashlife-tree` - Quadtree implementation
  - `rl` - RL adaptive jumping
  - `numba` - JIT-compiled
  - `torch` - CPU convolution
- GUI and Terminal TUI modes
- RL training system with PyTorch Lightning
- Benchmarking utilities
- Rule presets for Larger-than-Life

### Fixed
- Off-by-one coordinate bugs on board boundaries