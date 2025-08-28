# GitHub Copilot Instructions for NFL GPP Sim Optimizer

## Project Overview

The NFL GPP Sim Optimizer is a Streamlit web application for NFL Guaranteed Prize Pool (GPP) optimization with simulation blend projections. The application builds on historical 2023-2024 NFL data to create baseline priors, then uses Monte Carlo simulation to generate 2025 week projections with advanced analytics including value metrics, boom probability, and projection diagnostics.

## Key Features & Architecture

### 🏈 Baseline Data Pipeline
- **Historical Analysis**: Processes 2023-2024 NFL play-by-play and weekly data using nfl_data_py
- **Team Metrics**: Calculates pace, neutral pass rates, PROE (Pass Rate Over Expected), EPA/play, success rates
- **Player Metrics**: Generates usage shares, efficiency rates, TD rates, WOPR/RACR for receivers, XYAC components
- **Empirical Bayes Shrinkage**: Applies statistical shrinkage toward league/position means for robust priors

### ⚡ Monte Carlo Simulation Engine
- **Game Environment Modeling**: Simulates pace, pass rates adjusted by O/U lines and spreads
- **Position-Specific Logic**: Tailored simulation for QB, RB, WR, TE, and DST with realistic correlations
- **DraftKings Scoring**: Implements PPR scoring with 3-point yardage bonuses
- **Statistical Distributions**: Uses appropriate distributions (Normal, Poisson, Binomial) for different metrics

### 💎 Advanced Analytics
- **Value Metrics**: Calculates points per $1k salary, ceiling/floor value analysis
- **Boom Analysis**: Position-based boom thresholds with ownership-adjusted boom scores (1-100 scale)
- **Dart Flags**: Identifies low-ownership (≤5%) + high-boom (≥70 score) contrarian plays
- **Site Comparison**: Delta analysis, beat-site probability, coverage diagnostics

### 🖥️ Streamlit Integration
- **Three-Tab Interface**: Optimizer, Weekly Review (MVP), and Simulator tabs
- **File Upload/Download**: Supports CSV inputs/outputs with robust header normalization
- **Interactive Charts**: Projection distributions, sim vs site scatter plots using Altair
- **Results Visualization**: Tabbed interface showing projections, value analysis, diagnostics, and flags

## Technical Implementation

### Directory Structure (Planned)
```
src/
├── metrics/          # NFL data pipeline and team/player metrics
├── projections/      # Value metrics, boom scoring, diagnostics  
├── ingest/          # Site player data normalization
└── sim/             # Monte Carlo game simulation engine

scripts/             # CLI automation tools
docs/               # Comprehensive documentation and samples
```

### Current File Structure
```
.
├── .github/
│   ├── workflows/
│   └── copilot-instructions.md
├── app.py                    # Entry point for Render deployment
├── streamlit_app.py         # Main Streamlit application
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project configuration
├── render.yaml             # Render.com deployment config
└── README.md               # Project documentation
```

### CLI Tools (To Be Implemented)
- `scripts/build_baseline.py` - Generate historical priors from 2023-2024 data
- `scripts/build_boom_thresholds.py` - Calculate position-specific boom cutoffs
- `src/projections/run_week_from_site_players.py` - Full week simulation pipeline

## Data Flow

1. **Baseline Building**: Historical data → Team/Player metrics → Empirical Bayes priors
2. **Week Simulation**: Site players.csv → Game environment simulation → Player performance → Fantasy scoring
3. **Analytics Layer**: Raw projections → Value metrics → Boom analysis → Diagnostics → Flags

## Key Outputs

### Core Files
- `sim_players.csv` - Projections with mean, p10/p75/p90/p95 quantiles
- `compare.csv` - Value metrics, boom scores, site comparison, dart flags
- `diagnostics_summary.csv` - MAE, RMSE, correlation, coverage by position
- `flags.csv` - Players with large projection mismatches for review

### Key Metrics
- **Value per $1k**: Fantasy points per $1000 salary (3.0+ typically good value)
- **Boom Probability**: P(player reaches position boom threshold)
- **Boom Score**: 1-100 scale with ownership/value adjustments
- **Beat Site Probability**: P(simulation exceeds site projection)

## Monte Carlo Simulation Methodology (PDF-Guided)

The simulation engine should mirror the flow and parameterization from the provided PDF:

### Core Parameters
- **Trials**: Number of Monte Carlo simulations (CLI flag: `--trials`)
- **RNG Seeding**: For reproducible results (CLI flag: `--seed`)
- **Per-Player Outcome Modeling**: Position-specific distributions and correlations

### Distribution Guidelines
- Use PDF guidance for outcome distributions and parameter fitting
- Fall back to bootstrap/resampling for rookies/players with no history
- Implement appropriate statistical distributions (Normal, Poisson, Binomial)

### Adjustments & Modifiers
- Opponent strength/home-away/tempo adjustments (optional toggles)
- Weather/Vegas line integration (stubs if data not available)
- Game environment factors (pace, pass rates, O/U adjustments)

### Output Requirements
- Mean, variance, percentiles (P5/P50/P95)
- Threshold exceedance probabilities
- Both CLI and Streamlit "Simulator" tab reporting
- Deterministic runs via `--seed` parameter
- Configuration recording alongside results

## Development Guidelines

### Code Style
- Follow existing patterns in `streamlit_app.py`
- Use type hints where appropriate
- Include docstrings for functions and classes
- Maintain backward compatibility with existing optimizer functionality

### Error Handling & Validation
- **Robust CSV Parsing**: Handle various header formats (PLAYER/Name, SAL/Salary, etc.)
- **Mock Data Fallback**: Use synthetic data when nfl_data_py unavailable for testing
- **Input Validation**: Comprehensive checks for data quality and consistency
- **Graceful Degradation**: System continues with reduced functionality if components fail

### Testing Strategy
- Test complete pipeline from baseline building through final outputs
- Verify Streamlit UI functionality across all tabs
- Validate CLI scripts and core functionality
- Ensure file downloads and uploads work correctly
- Test with sample data and edge cases

### Dependencies
- `pandas>=2.2` - Data manipulation and analysis
- `numpy>=1.26` - Numerical computations
- `pulp>=2.7` - Linear programming for optimization
- `streamlit>=1.29,<2` - Web application framework
- `nfl_data_py` - NFL data access (to be added)
- `altair` - Interactive visualizations (already included via streamlit)

## Implementation Priorities

### Phase 1: Core Infrastructure
1. Set up `src/` directory structure
2. Implement basic Monte Carlo simulation engine
3. Add data ingestion and normalization utilities
4. Create baseline data pipeline framework

### Phase 2: Simulation Engine
1. Implement position-specific simulation logic
2. Add statistical distribution modeling
3. Integrate game environment factors
4. Build reproducible seeding system

### Phase 3: Analytics Layer
1. Implement value metrics calculations
2. Add boom analysis and scoring
3. Build site comparison capabilities
4. Create dart flag identification

### Phase 4: UI Integration
1. Add "Simulator" tab to Streamlit app
2. Implement file upload/download workflows
3. Create interactive visualizations
4. Add results display and export

### Phase 5: CLI Tools
1. Build baseline generation scripts
2. Create boom threshold calculation tools
3. Implement week simulation pipeline
4. Add configuration and parameter management

## Coding Best Practices

When working on this project:

1. **Maintain Minimal Changes**: Make surgical, precise modifications to existing code
2. **Follow Existing Patterns**: Use the established patterns from `streamlit_app.py`
3. **Test Incrementally**: Run and verify changes frequently
4. **Document Changes**: Include clear docstrings and comments
5. **Handle Edge Cases**: Robust error handling and validation
6. **Preserve Functionality**: Ensure existing optimizer features remain intact

## Current Application State

The application currently has:
- Two tabs: "Optimizer" and "Weekly Review (MVP)"
- File upload functionality for players, sims, defense, and stack configurations
- Placeholder optimization logic that generates example lineups
- CSV and ZIP download capabilities with metadata
- Exposure analysis and projection calibration features

## Next Steps for Simulator Implementation

1. Create the "Simulator" tab in the main Streamlit interface
2. Implement the Monte Carlo simulation engine following PDF guidelines
3. Build the data pipeline for historical NFL data processing
4. Add analytics capabilities for value metrics and boom analysis
5. Create CLI tools for baseline building and week simulation
6. Integrate with existing optimizer workflow

## Documentation Requirements

All new features should include:
- Inline code documentation and type hints
- User-facing documentation in `docs/` directory
- Sample data and usage examples
- Configuration parameter explanations
- Methodology documentation referencing the PDF guidelines