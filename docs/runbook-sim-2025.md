# NFL DFS Simulation Pipeline - Local Development Runbook

This document provides step-by-step instructions for running the NFL DFS simulation pipeline locally.

## Prerequisites

1. Python 3.8+ installed
2. Required Python packages (see requirements.txt)
3. Internet connection for initial NFL data download

## Installation

```bash
# Clone repository
git clone <repository-url>
cd nfl-gpp-sim-optimizer-

# Install dependencies
pip install -r requirements.txt
```

## Step 1: Build Baseline Data (One-time setup)

This step downloads historical NFL data from 2023-2024 seasons and calculates team/player priors.

```bash
# Build team and player priors
python scripts/build_baseline.py --start 2023 --end 2024 --out data

# Build boom thresholds
python scripts/build_boom_thresholds.py --start 2023 --end 2024 --out data/baseline/boom_thresholds.json --quantile 0.90
```

**Expected outputs:**
- `data/baseline/team_priors.csv` (~32 teams)
- `data/baseline/player_priors.csv` (~300+ players) 
- `data/baseline/boom_thresholds.json` (position boom cutoffs)

**Note:** First run will download ~500MB of NFL data and may take 5-10 minutes.

## Step 2: Prepare Player Data

Create or obtain a players.csv file with the following columns:
- `PLAYER` (required): Player name
- `POS` (required): Position (QB, RB, WR, TE, K, DST/D)
- `TEAM` (required): Team abbreviation
- `OPP`: Opponent (format: @GB, vs GB, or GB)
- `SAL`: DraftKings salary
- `FPTS`: Site projection (optional, for comparison)
- `RST%`: Ownership percentage
- `O/U`: Game over/under
- `SPRD`: Point spread
- `ML`, `TM/P`, `VAL`: Additional fields (preserved in output)

**Example:**
```csv
PLAYER,POS,TEAM,OPP,O/U,SPRD,SAL,FPTS,RST%
Josh Allen,QB,BUF,@GB,48.5,-3.5,8400,24.5,15.2
Derrick Henry,RB,BAL,vs CLE,45.0,-7,7800,18.2,8.5
```

## Step 3: Run Simulation

```bash
python -m src.projections.run_week_from_site_players \
    --season 2025 \
    --week 1 \
    --players-site path/to/players.csv \
    --team-priors data/baseline/team_priors.csv \
    --player-priors data/baseline/player_priors.csv \
    --boom-thresholds data/baseline/boom_thresholds.json \
    --sims 10000 \
    --out data/sim_week
```

**Parameters:**
- `--sims`: Number of Monte Carlo simulations (more = more accurate, slower)
- `--seed`: Random seed for reproducible results
- `--verbose`: Enable detailed logging

## Step 4: Launch Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

The web interface will be available at http://localhost:8501

## Output Files

After running simulations, the following files are created in `data/sim_week/`:

### Core Outputs
- **sim_players.csv**: Raw simulation results
  - `proj_mean`, `floor` (p10), `ceiling` (p90), `p95`
  - `boom_prob`, `beat_site_prob`

- **compare.csv**: Enhanced analysis with all metrics
  - Projections + salary, ownership, site data
  - `value_per_1k`, `ceil_per_1k`, `boom_score`
  - `dart_flag` for ≤5% owned, boom score ≥70

### Analysis Outputs
- **dart_throws.csv**: Low-owned, high-boom candidates
- **value_plays.csv**: Players with value_per_1k ≥ 3.0
- **diagnostics_summary.csv**: Accuracy metrics vs site
- **flags.csv**: Large projection discrepancies

## Troubleshooting

### Data Download Issues
```bash
# Clear cache and retry
python -c "from src.metrics.sources import get_global_loader; get_global_loader().clear_cache()"
```

### Memory Issues (Large Simulations)
```bash
# Reduce simulation count
--sims 5000

# Or run position-by-position
```

### Missing Player Priors
The system gracefully handles players not in historical data by using position averages. Check logs for "using defaults" messages.

## Performance Notes

- **Initial baseline build**: 5-10 minutes (downloads NFL data)
- **10K simulations**: 30-60 seconds for typical slate
- **50K simulations**: 3-5 minutes (recommended for final analysis)

## File Structure After Setup

```
data/
├── baseline/
│   ├── team_priors.csv       # Team pace, pass rate, EPA priors
│   ├── player_priors.csv     # Player efficiency, share priors  
│   └── boom_thresholds.json  # Position boom cutoffs
├── intermediate/             # Raw metrics (optional)
└── sim_week/                # Simulation outputs
    ├── sim_players.csv
    ├── compare.csv
    ├── dart_throws.csv
    └── diagnostics_summary.csv
```

## Next Steps

1. Upload `compare.csv` to your DFS optimizer
2. Target dart throw candidates for low-owned exposure
3. Use value metrics for cash game builds
4. Review flags.csv for projection discrepancies

## Advanced Usage

### Custom Boom Thresholds
```bash
python scripts/build_boom_thresholds.py --quantile 0.85  # 85th percentile instead of 90th
```

### Position-Specific Analysis
```python
# In Python console
import pandas as pd
results = pd.read_csv('data/sim_week/compare.csv')

# Top QB values
qb_values = results[results['position']=='QB'].nlargest(5, 'value_per_1k')
print(qb_values[['player', 'proj_mean', 'salary', 'value_per_1k']])
```

### Correlation Analysis
The simulation models player correlations within teams (QB-WR stacks get correlated game scripts) but doesn't model opponent correlations. For advanced analysis, consider manual adjustments based on flags.csv discrepancies.