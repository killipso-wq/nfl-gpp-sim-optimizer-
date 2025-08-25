# Research Documentation

## Overview

This directory contains research artifacts and methodology documentation for the NFL GPP Sim Optimizer.

## Research Artifacts

### Core Methodology
- **[monte_carlo_methodology.md](monte_carlo_methodology.md)**: Detailed explanation of simulation approach
- **[monte_carlo_football.pdf](monte_carlo_football.pdf)**: Comprehensive research PDF *(to be uploaded)*

## Implementation Mapping

### Research → Code Mapping

| Research Component | Implementation Module | Description |
|-------------------|----------------------|-------------|
| Position Distributions | `src/sim/game_simulator.py` | Normal/lognormal by position |
| Boom Thresholds | `scripts/build_boom_thresholds.py` | Historical p90 calculations |
| Player Priors | `scripts/build_baseline.py` | 2023-2024 historical data |
| Site Integration | `src/ingest/site_players.py` | CSV parsing and normalization |
| Value Metrics | `src/projections/value_metrics.py` | Value per $1K calculations |
| Diagnostics | `src/projections/diagnostics.py` | Model validation metrics |

### Data Flow Architecture

```
Historical Data (nfl_data_py)
    ↓
Baseline Scripts (build_baseline.py, build_boom_thresholds.py)
    ↓
Baseline Priors (team_priors.csv, player_priors.csv, boom_thresholds.json)
    ↓
Site Players Input (players.csv)
    ↓
Monte Carlo Simulator (game_simulator.py)
    ↓
Projection Outputs (sim_players.csv, compare.csv, diagnostics.csv)
```

## Statistical Foundation

### Distribution Selection
- **QB**: Lognormal (right-skewed due to big play potential)
- **RB**: Normal (more consistent distribution)
- **WR**: Lognormal (boom/bust nature)
- **TE**: Normal (limited target variance)
- **DST**: Lognormal (defensive scoring volatility)

### Variance Calibration
Position-specific variance derived from historical 2023-2024 weekly data:
- Standard deviation calculated from actual weekly performances
- Adjusted for game script and matchup factors
- Clamped at 0 minimum to prevent negative scoring

### Prior Integration
- **Historical mean**: 2023-2024 season average per player
- **Vegas adjustment**: Mild team-level multiplier from O/U and spread
- **Rookie fallback**: Site projection when no historical data available

## Validation Methodology

### Diagnostic Metrics
- **MAE (Mean Absolute Error)**: Average prediction accuracy
- **RMSE (Root Mean Squared Error)**: Prediction variance penalty
- **Correlation**: Linear relationship strength
- **Coverage**: Actual outcomes within predicted confidence intervals

### Position-Level Analysis
Diagnostics calculated separately by position to identify:
- Model bias patterns
- Prediction confidence levels
- Position-specific accuracy

### Exclusion Rules
- **Rookies**: Excluded from diagnostic calculations (flagged separately)
- **Injured players**: Filtered from validation set
- **Limited snaps**: <20% snap share excluded

## Future Research Directions

### Advanced Modeling
1. **Player Correlation**: Quantify same-team correlation matrices
2. **Game Environment**: Weather, dome/outdoor, time zone effects
3. **Opponent Strength**: Defensive unit adjustments
4. **Injury Impact**: Snap share redistribution modeling

### Optimization Enhancement
1. **Stack Correlation**: Mathematical correlation in optimizer constraints
2. **Ownership Prediction**: Dynamic ownership forecasting
3. **Late Swap**: Real-time lineup adjustment algorithms
4. **Multi-Slate**: Tournament selection optimization

### Data Integration
1. **Advanced Metrics**: PROE, xpass, XYAC incorporation
2. **Tracking Data**: Route running and target distribution
3. **Coaching Tendencies**: Play-calling situation analysis
4. **Market Inefficiencies**: Pricing lag exploitation

## Contributing to Research

To contribute new research or methodology improvements:

1. Document methodology in this directory
2. Map implementation to specific code modules
3. Include validation results and diagnostic analysis
4. Update this README with new artifacts

All research should maintain reproducibility through:
- Explicit seed control
- Version-controlled data sources
- Documented parameter choices
- Clear statistical assumptions