# Game of Life - Additional Rules

This directory contains additional rules for the Game of Life simulator.

## Usage

### Using ruleloader backend (loads JSON files):
```bash
gameoflife --backend ruleloader --rule-file rules/day_and_night.rule.json --pattern random
```

### Using built-in backends (via CLI):
```bash
# Standard Life (B3/S23)
gameoflife --backend jvn --rule B3/S23

# HighLife
gameoflife --backend jvn --rule B36/S23

# Brian's Brain (Generations)
gameoflife --backend generations --rule B2/S/C3
```

## Rules Collection

### Life-like (B3/S23 variants)

| Rule | Description | File |
|------|-------------|------|
| Day & Night | Symmetrical, many patterns | `day_and_night.rule.json` |
| HighLife | Like Life + B6, has replicators | `highlife.rule.json` |
| Morley | Many ships and puffers | `morley.rule.json` |
| Seeds | Explosive, all cells die | `seeds.rule.json` |

### Generations Rules

Use with `--backend generations`:

| Rule | States | Description |
|------|--------|-------------|
| B2/S/C3 | 3 | Brian's Brain - flying wires |
| B2/S/C4 | 4 | 4-state variant |

### Larger-than-Life Rules

Use with `--backend largerlife`:

```bash
# R2 example
gameoflife --backend largerlife --rule R2,B3/S23

# R3 example  
gameoflife --backend largerlife --rule R3,B34-45/S34-56
```

## Adding New Rules

Create a JSON file with:
```json
{
  "name": "RuleName",
  "description": "Description of the rule",
  "neighborhood": "moore",
  "radius": 1,
  "birth": [3],
  "survive": [2, 3],
  "states": 2
}
```

Place in this directory and use with `--backend ruleloader --rule-file rules/yourrule.rule.json`