# NFL GPP Sim Optimizer

A Streamlit web application for NFL GPP (Guaranteed Prize Pool) optimization with simulation blend projections.

## Methodology

This optimizer uses a Monte Carlo simulation approach to generate fantasy football projections. The methodology combines historical performance data (2023-2024 seasons) with current week site projections to create robust, position-calibrated fantasy point distributions.

**Key Features:**
- Position-specific statistical models (normal/lognormal distributions)
- Seeded Monte Carlo simulation for reproducible results
- Historical baseline priors with rookie fallback to site projections
- Boom probability analysis using quantile-based thresholds
- Comprehensive diagnostic validation

**Documentation:**
- [Master Reference](docs/master_reference.md) - Complete methodology overview
- [Research PDF](docs/research/monte_carlo_football.pdf) - Detailed statistical methodology *(to be uploaded)*
- [Monte Carlo Methodology](docs/research/monte_carlo_methodology.md) - Implementation details

## Features

- Upload CSV files for players, simulations, defense, and stack configurations
- Generate optimized lineups with preset configurations (se/mid/large)
- Download results as CSV or ZIP files
- Real-time deployment tracking with branch/commit info

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

## File Structure

- `app.py` - Entry point for Render deployment
- `streamlit_app.py` - Main Streamlit application code
- `render.yaml` - Render.com deployment configuration
- `requirements.txt` - Python dependencies