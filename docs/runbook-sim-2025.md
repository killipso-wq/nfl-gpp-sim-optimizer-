# NFL Simulator Runbook (2025)

## Overview

This guide explains how to use the NFL Simulator to build 2023-2024 baseline priors and generate 2025 week projections.

## Quick Start

### 1. Build Baseline (One-time setup)

Either use the Streamlit UI or run via CLI:

```bash
# Via CLI
python scripts/build_baseline.py --start 2023 --end 2024 --out data

# This creates:
# - data/baseline/team_priors.csv
# - data/baseline/player_priors.csv
```

### 2. Build Boom Thresholds (Optional)

```bash
python scripts/build_boom_thresholds.py --start 2023 --end 2024 --out data/baseline/boom_thresholds.json --quantile 0.90
```

### 3. Run Week Simulation

Upload your 2025 players.csv with these columns:
- PLAYER (name) 
- POS (position)
- TEAM  
- OPP (opponent)
- O/U (total line)
- SPRD (spread)
- SAL (salary)
- RST% (ownership %)
- FPTS (optional site projection)

#### Via Streamlit UI:
1. Go to "Simulator" tab
2. Upload players.csv
3. Set season/week parameters
4. Run simulation
5. Download results

#### Via CLI:
```bash
python -m src.projections.run_week_from_site_players \
  --season 2025 --week 1 \
  --players-site players_week1.csv \
  --team-priors data/baseline/team_priors.csv \
  --player-priors data/baseline/player_priors.csv \
  --boom-thresholds data/baseline/boom_thresholds.json \
  --sims 10000 \
  --out results/week1
```

## Output Files

### Core Projections (sim_players.csv)
- `proj_mean`: Main projection
- `p10, p75, p90, p95`: Percentiles (floor/ceiling)
- `std`: Standard deviation

### Comparison Analysis (compare.csv)
- `value_per_1k`: Value per $1000 salary
- `ceil_per_1k`: Ceiling value per $1000
- `boom_prob`: Probability of boom performance
- `boom_score`: 1-100 boom score with ownership/value adjustments
- `dart_flag`: Low ownership + high boom potential
- `delta_vs_site`: Difference vs site projection
- `beat_site_prob`: Probability of beating site projection

### Diagnostics (diagnostics_summary.csv)
- `correlation`: Correlation with site projections
- `mae`: Mean absolute error
- `rmse`: Root mean squared error  
- `coverage_p10_p90`: % of site projections within our p10-p90 range

### Flags (flags.csv)
Players with large projection differences for manual review.

## Understanding the Metrics

### Value Metrics
- **value_per_1k**: Points per $1000 salary (3.0+ is typically good value)
- **ceil_per_1k**: Ceiling value using p90 projection

### Boom Analysis
- **boom_prob**: Probability player reaches boom threshold for position
- **boom_score**: 1-100 score combining boom probability, value, and ownership
- **dart_flag**: RST% ≤ 5% AND boom_score ≥ 70

### Boom Thresholds (Default)
- QB: 25.0 points
- RB: 20.0 points  
- WR: 18.0 points
- TE: 15.0 points
- DST: 10.0 points

Player booms if: `sim_points ≥ max(position_threshold, 1.2×site_proj, site_proj+5)`

## Troubleshooting

### Common Issues

**"nfl_data_py not available" warning:**
- The system uses mock data when nfl_data_py can't be installed
- Mock data is sufficient for testing but use real data for production

**"No team priors available":**
- Build baseline first using the "Build Baseline" button
- Or upload existing team_priors.csv file

**"Large absolute difference" flags:**
- Review flagged players - large differences may indicate:
  - Site projection errors
  - Missing injury/status information
  - Different game environment assumptions

**Low correlation with site:**
- Expected for contrarian strategies
- Review diagnostics to understand systematic differences

### Performance Notes

- 10,000 simulations typical for accuracy
- Runtime scales with number of players and simulations
- Use fewer sims (1,000-5,000) for faster testing

## Advanced Usage

### Custom Priors
Upload your own team_priors.csv or player_priors.csv to override baseline data.

### Custom Boom Thresholds
Upload boom_thresholds.json with custom thresholds by position.

### Batch Processing
Use the CLI scripts for processing multiple weeks or automated workflows.

## File Formats

### players.csv (Input)
```csv
PLAYER,POS,TEAM,OPP,O/U,SPRD,SAL,RST%,FPTS
Patrick Mahomes,QB,KC,BUF,47.5,-3.0,9000,15.2,22.5
Josh Allen,QB,BUF,KC,47.5,3.0,8800,18.1,21.8
```

### team_priors.csv
```csv
team,plays_per_game_prior,neutral_pass_rate_prior,proe_neutral_prior,...
KC,65.2,0.582,0.023,...
BUF,67.8,0.601,-0.012,...
```

### boom_thresholds.json
```json
{
  "quantile": 0.90,
  "thresholds": {
    "QB": 25.0,
    "RB": 20.0,
    "WR": 18.0,
    "TE": 15.0,
    "DST": 10.0
  }
}
```