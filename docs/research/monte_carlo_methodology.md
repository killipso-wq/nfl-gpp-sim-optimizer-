# Monte Carlo Methodology

## Overview

Our Monte Carlo simulation approach models NFL player fantasy scoring using position-specific probability distributions calibrated to historical performance data.

## Core Methodology

### 1. Distribution Selection by Position

**Quarterbacks (QB)**
- **Distribution**: Lognormal
- **Rationale**: Right-skewed due to big play potential (long TDs, rushing scores)
- **Parameters**: μ (log-scale mean), σ (log-scale standard deviation)

**Running Backs (RB)**
- **Distribution**: Normal
- **Rationale**: More consistent touch-based scoring with predictable floor
- **Parameters**: μ (mean), σ (standard deviation)

**Wide Receivers (WR)**
- **Distribution**: Lognormal  
- **Rationale**: High boom/bust variance, big play dependency
- **Parameters**: μ (log-scale mean), σ (log-scale standard deviation)

**Tight Ends (TE)**
- **Distribution**: Normal
- **Rationale**: Limited target variance, more predictable role
- **Parameters**: μ (mean), σ (standard deviation)

**Defense/Special Teams (DST)**
- **Distribution**: Lognormal
- **Rationale**: Defensive scoring highly volatile (turnovers, TDs)
- **Parameters**: μ (log-scale mean), σ (log-scale standard deviation)

### 2. Parameter Estimation

**Historical Prior Calculation**
```python
# For players with 2023-2024 data
player_games = historical_data[historical_data['player_id'] == player_id]
prior_mean = player_games['fantasy_points'].mean()
prior_std = player_games['fantasy_points'].std()
```

**Position Baseline Variance**
```python
# Position-level variance for rookie fallback
position_games = historical_data[historical_data['position'] == position]
position_variance = position_games['fantasy_points'].var()
```

**Vegas Integration**
```python
# Mild team-level adjustment from game totals
team_multiplier = (game_total / season_avg_total) ** 0.3  # Conservative weighting
adjusted_mean = prior_mean * team_multiplier
```

### 3. Simulation Process

**Monte Carlo Execution**
```python
def simulate_player(mean, std, position, n_sims=10000, seed=None):
    rng = np.random.Generator(np.random.PCG64(seed))
    
    if position in ['QB', 'WR', 'DST']:
        # Lognormal distribution
        log_mean = np.log(mean**2 / np.sqrt(std**2 + mean**2))
        log_std = np.sqrt(np.log(1 + std**2 / mean**2))
        samples = rng.lognormal(log_mean, log_std, n_sims)
    else:
        # Normal distribution (RB, TE)
        samples = rng.normal(mean, std, n_sims)
    
    # Clamp at 0 minimum
    samples = np.maximum(samples, 0)
    
    # Apply DraftKings bonuses (approximate)
    samples = apply_dk_bonuses(samples, position)
    
    return samples
```

**DraftKings Bonus Application**
```python
def apply_dk_bonuses(samples, position):
    if position == 'QB':
        # 300+ passing yards: +3 points
        samples += np.where(samples >= 18, 3, 0)  # Approx threshold
    elif position in ['RB', 'WR']:
        # 100+ rushing/receiving yards: +3 points
        samples += np.where(samples >= 12, 3, 0)  # Approx threshold
    return samples
```

### 4. Output Metrics

**Quantile Calculations**
```python
def calculate_quantiles(samples):
    return {
        'p10': np.percentile(samples, 10),
        'p50': np.percentile(samples, 50),  # Median
        'p75': np.percentile(samples, 75),
        'p90': np.percentile(samples, 90),
        'p95': np.percentile(samples, 95),
        'sim_mean': np.mean(samples)
    }
```

**Probability Metrics**
```python
def calculate_probabilities(samples, boom_threshold, site_projection=None):
    probs = {
        'boom_prob': np.mean(samples >= boom_threshold)
    }
    
    if site_projection is not None:
        probs['beat_site_prob'] = np.mean(samples >= site_projection)
    
    return probs
```

## Implementation Mapping

### Code Structure
- **Core Engine**: `src/sim/game_simulator.py`
- **Parameter Building**: `scripts/build_baseline.py`
- **Boom Thresholds**: `scripts/build_boom_thresholds.py`
- **CLI Interface**: `src/projections/run_week_from_site_players.py`

### Data Pipeline
1. **Historical Data Ingestion**: nfl_data_py weekly stats (2023-2024)
2. **Prior Calculation**: Player and position-level statistics
3. **Site Integration**: CSV parsing with synonym mapping
4. **Simulation Execution**: Monte Carlo with seeded RNG
5. **Output Generation**: Quantiles, probabilities, diagnostics

### Validation Framework
- **Diagnostic Module**: `src/projections/diagnostics.py`
- **Metrics**: MAE, RMSE, correlation, coverage analysis
- **Position Segmentation**: Separate validation by position
- **Rookie Exclusion**: Historical validation excludes rookie fallback

## Statistical Assumptions

### 1. Independence
- Player performances assumed independent within lineup
- Future enhancement: correlation modeling for same-team players

### 2. Stationarity  
- Historical variance patterns persist into current season
- Adjustment for major rule changes or injury impacts

### 3. Distribution Appropriateness
- Normal/lognormal chosen based on empirical fit to historical data
- Position-specific selection based on skewness patterns

### 4. Parameter Stability
- Player skill level relatively stable across seasons
- Game environment effects captured through Vegas integration

## Limitations and Extensions

### Current Limitations
1. **No correlation modeling** between same-team players
2. **Simplified bonus structure** approximation
3. **Limited game environment factors** (weather, dome effects)
4. **Static opponent adjustments** (no defensive strength modeling)

### Future Extensions
1. **Correlation Matrices**: Copula-based same-team correlation
2. **Advanced Bonuses**: Precise yardage-based bonus calculation
3. **Environmental Models**: Weather, altitude, surface effects
4. **Opponent Strength**: Defensive unit rate statistics
5. **Dynamic Variance**: Injury/target share redistribution

## Research Validation

### Backtesting Methodology
1. **Out-of-sample testing**: Validate on holdout weeks
2. **Cross-validation**: K-fold validation across season chunks
3. **Position accuracy**: Separate model evaluation by position
4. **Extreme outcome coverage**: Validate tail probability predictions

### Key Findings
- Lognormal distributions better capture boom outcomes for skill positions
- Position-specific variance essential for accurate confidence intervals
- Vegas integration provides marginal improvement over pure historical priors
- Model performs best for established players with full season history

### Performance Benchmarks
- **Overall MAE**: Target <15% improvement over site projections
- **Correlation**: Target >0.6 correlation with actual outcomes
- **Coverage**: 80% confidence intervals should contain 80% of outcomes
- **Boom Prediction**: ROC-AUC >0.65 for boom outcome prediction

This methodology provides the foundation for all simulation outputs while maintaining computational efficiency and statistical rigor.