# NFL GPP Sim Optimizer - Master Reference

## Overview

The NFL GPP Sim Optimizer is a comprehensive tool for generating optimized lineups for NFL Guaranteed Prize Pool (GPP) contests using Monte Carlo simulation and advanced projection techniques.

## Methodology

### Monte Carlo Simulation

Our simulation engine implements position-calibrated normal/lognormal distributions with the following features:

- **Seeded RNG**: Reproducible results using numpy Generator with explicit seed control
- **Position-specific variance**: Different statistical models for QB, RB, WR, TE, DST positions
- **Clamp bounds**: Ensures no negative fantasy points in simulations
- **DraftKings scoring**: Applies bonuses for performance thresholds (passing/rushing/receiving yards)

### Projection Sources

1. **Historical Priors (2023-2024)**: Usage and efficiency data collapsed into points means
2. **Site Projections**: Fallback for rookies and new players
3. **Vegas Integration**: Mild adjustment based on O/U and spread data

### Key Metrics

- **Floor/Ceiling**: p10, p75, p90, p95 quantiles from simulation distribution
- **Boom Probability**: Likelihood of exceeding position-specific boom thresholds (default p90)
- **Beat Site Probability**: Chance of outperforming site projections
- **Value Metrics**: Value per $1K salary, ceiling per $1K
- **Boom Score**: Composite 1-100 score combining boom probability and value factors

## GPP Strategy Framework

Our GPP approach balances:

- **Stack construction** with bring-back opportunities
- **Ownership differentiation** through dart plays and contrarian selections  
- **Value optimization** across salary constraints
- **Boom upside** targeting for tournament success

See [GPP Strategy Blueprint](gpp_strategy_blueprint.md) for detailed implementation.

## Research Foundation

Detailed methodology documentation available at:
- [Monte Carlo Methodology](research/monte_carlo_methodology.md)
- [Research PDF](research/monte_carlo_football.pdf) *(to be uploaded)*

## File Outputs

### Simulation Results
- `sim_players.csv`: Core simulation results with quantiles and probabilities
- `compare.csv`: Site comparison with deltas and value metrics
- `diagnostics_summary.csv`: Model validation statistics
- `flags.csv`: Data quality and extreme delta alerts
- `metadata.json`: Run configuration and statistics
- `simulator_outputs.zip`: Complete results bundle

### Baseline Data
- `data/baseline/team_priors.csv`: Historical team performance data
- `data/baseline/player_priors.csv`: Historical player performance data  
- `data/baseline/boom_thresholds.json`: Position-specific boom thresholds

## CLI Usage

### Build Baseline Data
```bash
python scripts/build_baseline.py --start 2023 --end 2024 --out data
python scripts/build_boom_thresholds.py --start 2023 --end 2024 --out data/baseline/boom_thresholds.json --quantile 0.90
```

### Run Weekly Simulation
```bash
python -m src.projections.run_week_from_site_players \
  --season 2025 --week 1 \
  --players-site path/to/players.csv \
  --team-priors data/baseline/team_priors.csv \
  --player-priors data/baseline/player_priors.csv \
  --boom-thresholds data/baseline/boom_thresholds.json \
  --sims 10000 --out data/sim_week
```

## Streamlit UI

### Simulator Tab
- File upload for players.csv
- Column mapping and validation
- Simulation parameter control (sims, seed)
- Results preview with filtering and sorting
- Downloadable outputs (CSV and ZIP)
- Methodology documentation

### Optimizer Tab
- **GPP Presets**: Small/Mid/Large configurations
- **Constraint Controls**: Ownership bands, boom thresholds, value requirements
- **Stack Settings**: Bring-back enforcement, mini-stacks, dart requirements
- **Lineup Generation**: *(full optimization in follow-up PR)*

## Extension Points

The architecture supports future enhancements:

- **Advanced Models**: PROE, xpass, XYAC integration
- **Correlation Modeling**: Player and team correlation matrices
- **DST Adapter**: Defense-specific projection adjustments
- **Stack Optimization**: Advanced correlation-aware stacking
- **Live Updates**: Real-time injury and weather adjustments

## Data Requirements

### Site Players CSV Format
Required columns (with synonym mapping):
- `PLAYER` (Player, Name)
- `POS` (Position) 
- `TEAM` (Team, Tm)
- `OPP` (Opponent, Opp)
- `SAL` (Salary, Sal, DK Salary)
- `FPTS` (Projection, Proj, DK Points)
- `RST%` (Ownership, Own, Roster%)
- `O/U` (Over/Under, Total)
- `SPRD` (Spread, Line)

### Output Schema
All outputs follow consistent naming conventions with player_id as the primary key (`TEAM_POS_NORMALIZEDNAME`).