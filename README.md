# NFL GPP Simulation Optimizer

A data-driven NFL GPP (Guaranteed Prize Pool) optimizer as a Python package with a small CLI. This package implements projection blending, lineup optimization with stacking rules, and chalk/pivot reporting for DraftKings NFL contests.

## Features

- **Projection Blending**: 50/50 blend of original Fantasy points and sim_mean (if provided), rounded half up, with value recompute
- **DST Position Normalization**: DST positions normalized internally (D → DST for processing, DST → D for output)
- **Lineup Optimization**: Generates optimized lineups with:
  - Salary cap constraints (up to $50,000)
  - Position requirements (1 QB, 2+ RB, 3+ WR, 1+ TE, 1 DST)
  - Team stacking rules (QB + receiving option from same team)
  - Defensive constraints (no player vs opposing DST for RB, no QB/WR/TE with own DST)
  - Max 4 players per team (excluding DST)
- **Ownership-Aware Optimization**: Uses roster percentage (RST%) for chalk avoidance and pivot identification
- **Game Environment Scoring**: Incorporates Over/Under and spread data for game script analysis
- **Chalk/Pivot Analysis**: Identifies high-ownership players and suggests lower-owned alternatives

## Installation

```bash
pip install -e .
```

## Usage

The package provides a CLI with two main commands:

### 1. Compare and Apply Projection Blending

Blend simulation projections with original projections and generate comparison reports:

```bash
nfl-gpp compare-apply --players players.csv --sims sims.csv --outdir out/
```

**Arguments:**
- `--players` (required): CSV with player data including Fantasy points, SAL, RST%, Position columns
- `--sims` (optional): CSV with simulation data including sim_mean column
- `--defcsv` (optional): Defense CSV for DST sync
- `--outdir` (default: "out"): Output directory for results

**Outputs:**
- `players_blended.csv`: Players with updated projections
- `def_adjusted.csv`: Defense data adjusted to match DST projections (if provided)
- `projection_deltas.csv`: Report showing projection changes

### 2. Build Optimized Lineups

Generate optimized lineups with stacking and ownership rules:

```bash
nfl-gpp build-lineups --players players.csv --preset large --n 150 --outdir out/
```

**Arguments:**
- `--players` (required): CSV with player data
- `--preset` (default: "large"): Optimization preset - choices: se, mid, large
- `--n` (default: 150): Number of lineups to generate
- `--corrcsv` (optional): Correlation data for advanced stacking
- `--outdir` (default: "out"): Output directory for results

**Presets:**
- `se` (Small Entry): Conservative, low ownership penalty, tighter salary requirements
- `mid` (Mid-Stakes): Balanced approach with moderate ownership penalty
- `large` (Large Field): Aggressive, higher ownership penalty, more salary flexibility

**Outputs:**
- `lineups.csv`: Generated lineup combinations with projections and ownership
- `chalk_players.csv`: High-ownership players identified as chalk
- `pivot_options.csv`: Lower-owned alternatives to chalk players

## Input Data Format

### Players CSV
Required columns:
- `Name`: Player name
- `Position`: Player position (QB, RB, WR, TE, D)
- `Team`: Player's team abbreviation
- `SAL`: DraftKings salary
- `Fantasy points`: Projected fantasy points
- `RST%`: Roster percentage (ownership projection)

Optional columns:
- `Opp`: Opposing team abbreviation
- `O/U`: Game over/under total
- `SPRD`: Game spread
- `player_id`: Unique player identifier

### Sims CSV (Optional)
- `player_id` or `Name` + `Position`: Player identifier
- `mean` (or `sim_mean`): Simulated projection mean
- `stdev_sim`: Standard deviation of simulations
- `floor_sim`: Simulated floor projection
- `ceil_sim`: Simulated ceiling projection

## Package Structure

```
src/nfl_gpp_sim_optimizer/
├── __init__.py          # Package initialization
├── cli.py               # Command-line interface
├── pipeline.py          # Main pipeline functions
├── projections.py       # Projection blending logic
├── optimizer.py         # Lineup optimization and constraints
├── chalk.py            # Chalk identification and pivot finding
└── io.py               # File I/O operations
```

## Requirements

- Python ≥ 3.10
- pandas ≥ 2.2
- numpy ≥ 1.26
- pulp ≥ 2.7

## License

This project is released under the MIT License.