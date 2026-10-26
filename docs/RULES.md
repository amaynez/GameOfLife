# Rules Guide

This guide covers all rule types supported by Game of Life.

## Rule Syntax

### Birth/Survive (B/S)
```
B<birth_digits>/S<survive_digits>
```

- `B3` = Born if 3 neighbors
- `S23` = Survives with 2-3 neighbors

Example: `B3/S23` = Conway's Life

## Rule Types

### 1. Life-like (Moore Neighborhood)

Standard 2-state rules using 8-neighbor Moore neighborhood.

```bash
gameoflife --backend jvn --rule B3/S23          # Conway's Life
gameoflife --backend jvn --rule B36/S23         # HighLife  
gameoflife --backend jvn --rule B3678/S34678    # Day & Night
gameoflife --backend jvn --rule B2/S             # Seeds
gameoflife --backend jvn --rule B2/S0            # Serviensaver
```

#### Popular Life-like Rules

| Rule | B/S | Description |
|------|-----|-------------|
| Conway's Life | B3/S23 | Classic |
| HighLife | B36/S23 | Has replicators |
| Day & Night | B3678/S34678 | Symmetric |
| Morley | B368/S245 | Many ships |
| Seeds | B2/S | Explosive |
| Live Free or Die | B2/S0 | Rule 0 |

### 2. Generations (Multi-state)

Multi-state rules where cells cycle through states.

```bash
gameoflife --backend generations --rule B2/S/C3     # Brian's Brain
gameoflife --backend generations --rule B2/S/C4     # 4-state variant
gameoflife --backend generations --rule B3/S23/C5   # 5-state Life
```

**Format:** `B<birth>/S<survive>/C<states>`

#### Popular Generations Rules

| Rule | B/S/C | Description |
|------|-------|-------------|
| Brian's Brain | B2/S/C3 | Flying wires |
| Day & Night | B3678/S34678/C2 | (can use Generations) |
| Star Wars | B2/S/C4 | Complex patterns |

### 3. von Neumann Neighborhood

Uses 4-neighbor (up/down/left/right) instead of 8.

```bash
gameoflife --backend jvn --rule B2/S12            # Default (R1)
gameoflife --backend jvn --rule R2,B3/S23         # Radius 2
gameoflife --backend jvn --rule R3,B3/S23         # Radius 3
```

**Format:** `R<radius>,B<birth>/S<survive>`

Default radius is 1 (4 neighbors max).

### 4. Larger-than-Life (LTL)

Extended range cellular automata.

```bash
gameoflife --backend largerlife --rule R2,B3/S23     # R2 default
gameoflife --backend largerlife --rule R3,B34-45/S34-56
gameoflife --backend largerlife --rule-preset bosco  # Preset
```

**Format:** `R<radius>,B<min-max>,S<min-max>`

#### Presets

```bash
--rule-preset bosco   # R5, B30-44, S28-52
--rule-preset coral  # R5, B36-46, S55-68  
--rule-preset nova   # R5, B36-54, S54-72
--rule-preset storm  # R5, B34-56, S46-58
```

### 5. Custom JSON Rules

Create rule files in `rules/` directory:

```json
{
  "name": "MyRule",
  "neighborhood": "moore",
  "radius": 1,
  "birth": [3],
  "survive": [2, 3],
  "states": 2
}
```

Load with:
```bash
gameoflife --backend ruleloader --rule-file rules/myrule.rule.json
```

## Rule Files

See `rules/` directory for examples:

| File | Rule | Type |
|------|------|------|
| `example.rule.json` | B3/S23 | JSON |
| `day_and_night.rule.json` | B3678/S34678 | JSON |
| `highlife.rule.json` | B36/S23 | JSON |
| `morley.rule.json` | B368/S245 | JSON |
| `seeds.rule.json` | B2/S | JSON |

## Understanding the Rules

### Birth Conditions
Which neighbor counts cause a dead cell to become alive.

### Survive Conditions
Which neighbor counts keep a live cell alive.

### Example: Conway's Life
- **Birth**: Cell becomes alive with exactly 3 neighbors
- **Survive**: Cell stays alive with 2 or 3 neighbors
- **Otherwise**: Cell dies or stays dead

## Performance by Rule Type

| Backend | Best For |
|---------|----------|
| quicklife | Standard Life |
| hashlife | Long simulations |
| numba | Large grids |
| torch | GPU acceleration |
| generations | Multi-state |
| largerlife | Extended range |