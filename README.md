# NFL GPP Sim Optimizer

A Streamlit web application for NFL GPP (Guaranteed Prize Pool) optimization with simulation blend projections.

## Features

- Upload CSV files for players, simulations, defense, and stack configurations
- Generate optimized lineups with preset configurations (se/mid/large)
- Download results as CSV or ZIP files
- Real-time deployment tracking with branch/commit info
- Simulator tab to run per-player Monte Carlo simulations and download:
  - sim_players.csv
  - compare.csv
  - diagnostics_summary.csv
  - flags.csv

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

### Simulator Tab (new)

1. Open the app and navigate to the "Simulator" tab.
2. Upload your Players CSV with columns similar to: PLAYER, POS (D→DST), TEAM, OPP, FPTS.
3. Choose number of simulations per player and a random seed.
4. Click "Run simulation" to generate results.
5. Download the four files for downstream use: sim_players.csv, compare.csv, diagnostics_summary.csv, flags.csv.

## File Structure

- `app.py` - Entry point for Render deployment
- `streamlit_app.py` - Main Streamlit application code
- `render.yaml` - Render.com deployment configuration
- `requirements.txt` - Python dependencies