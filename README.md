# NFL GPP Sim Optimizer

A Streamlit web application for NFL GPP (Guaranteed Prize Pool) optimization with simulation blend projections.

## Features

- **Optimizer**: Upload CSV files for players, simulations, defense, and stack configurations
- **Weekly Review**: Generate optimized lineups with preset configurations (se/mid/large) 
- **Simulator**: Build 2023-2024 baseline priors and run 2025 week Monte Carlo simulations
- Download results as CSV or ZIP files
- Real-time deployment tracking with branch/commit info

### Simulator Features

- Build team and player priors from historical NFL data (2023-2024) using nfl_data_py
- Monte Carlo simulation engine for 2025 week projections
- Value metrics: points per $1k salary, ceiling/floor analysis
- Boom probability and boom scores with ownership adjustments  
- Dart flags for low-owned, high-upside plays
- Projection diagnostics comparing to site data (MAE, RMSE, correlation)
- Interactive charts and comprehensive CSV exports

## Deployment

### Render.com (Recommended)

This app is configured for automatic deployment on Render.com using the included `render.yaml` configuration.

**Auto Deploy**: Connect your GitHub repository to Render and it will automatically deploy on every push to main.

**Manual Deploy/Rebuild**: In your Render dashboard, click "Manual Deploy" → "Deploy latest commit" to trigger a rebuild.

### Configuration Options

The `render.yaml` includes an optional environment variable to increase upload size limits:

```yaml
# Uncomment to allow uploads up to 300MB (default is ~200MB)
- key: STREAMLIT_SERVER_MAX_UPLOAD_SIZE
  value: "300"
```

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

## Using the Simulator

### Quick Start

1. **Build Baseline**: Go to "Simulator" tab, click "Build Baseline" to generate 2023-2024 priors
2. **Upload Players**: Upload your site players.csv with columns: PLAYER, POS, TEAM, OPP, O/U, SPRD, SAL, RST%, FPTS (optional)
3. **Run Simulation**: Click "Run Simulation" to generate projections with 10,000 Monte Carlo iterations
4. **Analyze Results**: Review projections, value metrics, boom scores, and download CSV files

### Command Line Usage

```bash
# Build baseline priors
python scripts/build_baseline.py --start 2023 --end 2024 --out data

# Build boom thresholds  
python scripts/build_boom_thresholds.py --start 2023 --end 2024 --out data/baseline/boom_thresholds.json

# Run week simulation
python -m src.projections.run_week_from_site_players \
  --season 2025 --week 1 \
  --players-site docs/sample_players_2025.csv \
  --team-priors data/baseline/team_priors.csv \
  --player-priors data/baseline/player_priors.csv \
  --sims 10000 --out results/week1
```

See `docs/runbook-sim-2025.md` for detailed instructions.

## File Structure

- `app.py` - Entry point for Render deployment
- `streamlit_app.py` - Main Streamlit application code
- `src/` - Core simulation and analysis modules
  - `metrics/` - NFL data processing and team/player metrics
  - `sim/` - Monte Carlo game simulation engine  
  - `projections/` - Value metrics, boom scoring, diagnostics
  - `ingest/` - Site player data ingestion and normalization
- `scripts/` - CLI scripts for baseline building and automation
- `docs/` - Documentation and sample files
- `render.yaml` - Render.com deployment configuration
- `requirements.txt` - Python dependencies