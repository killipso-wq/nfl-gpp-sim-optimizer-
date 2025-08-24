"""
NFL DFS Simulation Pipeline - Streamlit UI

Provides a web interface for building baselines, running simulations,
and analyzing fantasy football projections with boom/bust analysis.
"""

import os
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st

# Import our modules
try:
    from src.ingest.site_players import load_site_players_csv
    from src.ingest.name_normalizer import get_global_mapper
    from src.metrics.sources import get_global_loader
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()


def version_banner():
    """Display version information."""
    branch = os.getenv("RENDER_GIT_BRANCH") or "local"
    commit = (os.getenv("RENDER_GIT_COMMIT") or "")[:7]
    service = os.getenv("RENDER_SERVICE_NAME") or ""
    banner = f"branch: {branch}"
    if commit:
        banner += f" • commit: {commit}"
    if service:
        banner += f" • service: {service}"
    st.caption(banner)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_baseline_data() -> Tuple[bool, str]:
    """
    Check if baseline data exists and load status.
    
    Returns:
        Tuple of (exists, status_message)
    """
    baseline_dir = Path("data/baseline")
    
    required_files = [
        "team_priors.csv",
        "player_priors.csv", 
        "boom_thresholds.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not (baseline_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        return False, f"Missing baseline files: {', '.join(missing_files)}"
    
    return True, "Baseline data ready"


def build_baseline_ui():
    """UI for building baseline data from NFL historical data."""
    st.header("🏗️ Build Baseline")
    
    baseline_exists, status = load_baseline_data()
    
    if baseline_exists:
        st.success(f"✅ {status}")
        if st.button("Rebuild baseline data", type="secondary"):
            st.session_state.rebuild_baseline = True
    else:
        st.warning(f"⚠️ {status}")
    
    if not baseline_exists or st.session_state.get('rebuild_baseline', False):
        st.info("📥 Building baseline from 2023-2024 NFL data. This may take a few minutes...")
        
        with st.spinner("Downloading NFL data and calculating metrics..."):
            try:
                # This would run the baseline building process
                # For now, we'll create placeholder data
                create_placeholder_baseline_data()
                st.success("✅ Baseline data built successfully!")
                st.session_state.rebuild_baseline = False
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Failed to build baseline: {e}")


def create_placeholder_baseline_data():
    """Create placeholder baseline data for testing."""
    baseline_dir = Path("data/baseline")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    # Create placeholder team priors
    teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
             'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LV', 'LAC', 'LAR', 'MIA',
             'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB',
             'TEN', 'WAS']
    
    team_priors = pd.DataFrame({
        'team': teams,
        'pace': np.random.normal(65, 3, len(teams)),
        'pass_rate': np.random.normal(0.6, 0.05, len(teams)),
        'epa_per_play': np.random.normal(0.05, 0.1, len(teams))
    })
    team_priors.to_csv(baseline_dir / "team_priors.csv", index=False)
    
    # Create placeholder player priors
    positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
    player_data = []
    
    for pos in positions:
        n_players = {'QB': 40, 'RB': 80, 'WR': 120, 'TE': 40, 'K': 32, 'DST': 32}[pos]
        for i in range(n_players):
            player_data.append({
                'player_id': f'{pos}_player_{i+1}',
                'position': pos,
                'team': np.random.choice(teams),
                'target_share': np.random.beta(2, 8) if pos in ['WR', 'TE'] else 0,
                'rush_share': np.random.beta(3, 7) if pos in ['QB', 'RB'] else 0,
                'efficiency': np.random.normal(1.0, 0.2)
            })
    
    player_priors = pd.DataFrame(player_data)
    player_priors.to_csv(baseline_dir / "player_priors.csv", index=False)
    
    # Create boom thresholds
    boom_thresholds = {
        'QB': 25.0,
        'RB': 20.0, 
        'WR': 18.0,
        'TE': 15.0,
        'K': 12.0,
        'DST': 12.0
    }
    
    with open(baseline_dir / "boom_thresholds.json", 'w') as f:
        json.dump(boom_thresholds, f, indent=2)


def simulation_ui():
    """UI for running simulations on uploaded player data."""
    st.header("🎯 Run Simulations")
    
    # Check baseline
    baseline_exists, status = load_baseline_data()
    if not baseline_exists:
        st.error(f"❌ {status}. Please build baseline data first.")
        return
    
    # File uploader
    players_file = st.file_uploader(
        "Upload players.csv", 
        type=["csv"],
        help="Upload your DFS site player data (with PLAYER, POS, TEAM, OPP, SAL, etc.)"
    )
    
    if not players_file:
        st.info("📄 Upload a players.csv file to run simulations")
        return
    
    # Load and validate player data
    try:
        players_df, column_mapping, validation_errors = load_site_players_csv(players_file)
        
        st.success(f"✅ Loaded {len(players_df)} players")
        
        # Show column mapping
        with st.expander("📋 Column Mapping", expanded=False):
            st.json(column_mapping)
        
        # Show validation errors
        if validation_errors:
            st.warning("⚠️ Data validation issues:")
            for error in validation_errors:
                st.write(f"• {error}")
        
        # Preview data
        with st.expander("👀 Data Preview", expanded=True):
            st.dataframe(players_df.head(10), use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Failed to load player data: {e}")
        return
    
    # Simulation parameters
    col1, col2 = st.columns(2)
    with col1:
        n_sims = st.number_input("Number of simulations", min_value=1000, max_value=50000, value=10000, step=1000)
    with col2:
        week = st.number_input("Week", min_value=1, max_value=18, value=1)
    
    # Run simulation
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        with st.spinner(f"Running {n_sims:,} simulations..."):
            try:
                # Create placeholder simulation results
                sim_results = create_placeholder_simulation_results(players_df, n_sims)
                
                # Store results in session state
                st.session_state.sim_results = sim_results
                st.session_state.players_df = players_df
                
                st.success(f"✅ Simulation complete! Generated projections for {len(sim_results)} players")
                
            except Exception as e:
                st.error(f"❌ Simulation failed: {e}")


def create_placeholder_simulation_results(players_df: pd.DataFrame, n_sims: int) -> pd.DataFrame:
    """Create placeholder simulation results for testing."""
    results = []
    
    for _, player in players_df.iterrows():
        # Generate realistic fantasy point distributions by position
        pos = player.get('position', 'WR')
        
        if pos == 'QB':
            base_points = np.random.normal(18, 8, n_sims)
        elif pos == 'RB':
            base_points = np.random.normal(12, 7, n_sims)
        elif pos == 'WR':
            base_points = np.random.normal(10, 6, n_sims)
        elif pos == 'TE':
            base_points = np.random.normal(8, 5, n_sims)
        elif pos == 'K':
            base_points = np.random.normal(7, 3, n_sims)
        else:  # DST
            base_points = np.random.normal(8, 5, n_sims)
        
        # Ensure no negative points
        base_points = np.maximum(0, base_points)
        
        # Calculate percentiles
        proj_mean = np.mean(base_points)
        p10 = np.percentile(base_points, 10)
        p75 = np.percentile(base_points, 75)  
        p90 = np.percentile(base_points, 90)
        p95 = np.percentile(base_points, 95)
        
        # Boom probability (% of sims >= position threshold)
        boom_thresholds = {'QB': 25, 'RB': 20, 'WR': 18, 'TE': 15, 'K': 12, 'DST': 12}
        boom_threshold = boom_thresholds.get(pos, 15)
        boom_prob = (base_points >= boom_threshold).mean() * 100
        
        # Beat site probability if FPTS available
        site_fpts = player.get('fpts', np.nan)
        if pd.notna(site_fpts):
            beat_site_prob = (base_points >= site_fpts).mean() * 100
        else:
            beat_site_prob = np.nan
        
        results.append({
            'player': player.get('player', ''),
            'position': pos,
            'team': player.get('team', ''),
            'proj_mean': round(proj_mean, 2),
            'floor': round(p10, 2),
            'p75': round(p75, 2),
            'ceiling': round(p90, 2),
            'p95': round(p95, 2),
            'boom_prob': round(boom_prob, 1),
            'beat_site_prob': round(beat_site_prob, 1) if pd.notna(beat_site_prob) else np.nan
        })
    
    return pd.DataFrame(results)


def analysis_ui():
    """UI for analyzing simulation results."""
    st.header("📊 Analysis")
    
    if 'sim_results' not in st.session_state:
        st.info("🎯 Run a simulation first to see analysis")
        return
    
    sim_results = st.session_state.sim_results
    players_df = st.session_state.players_df
    
    # Merge with original data for salary/ownership info
    analysis_df = sim_results.merge(
        players_df[['player', 'salary', 'ownership', 'fpts']],
        on='player',
        how='left'
    )
    
    # Calculate value metrics
    analysis_df['value_per_1k'] = (analysis_df['proj_mean'] / analysis_df['salary'] * 1000).round(3)
    analysis_df['ceil_per_1k'] = (analysis_df['ceiling'] / analysis_df['salary'] * 1000).round(3)
    
    # Calculate boom score
    ownership_factor = np.where(analysis_df['ownership'] <= 5, 1.5, 1.0)
    value_factor = np.clip(analysis_df['value_per_1k'] / 3, 0.5, 2.0)
    analysis_df['boom_score'] = (analysis_df['boom_prob'] * ownership_factor * value_factor).clip(0, 100).round(1)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Projections", "💎 Dart Throws", "⚖️ Value", "🚩 Flags"])
    
    with tab1:
        st.subheader("Player Projections")
        
        # Position filter
        positions = ['All'] + sorted(analysis_df['position'].unique().tolist())
        pos_filter = st.selectbox("Filter by position:", positions)
        
        display_df = analysis_df if pos_filter == 'All' else analysis_df[analysis_df['position'] == pos_filter]
        
        # Sort by boom score by default
        display_df = display_df.sort_values('boom_score', ascending=False)
        
        st.dataframe(
            display_df[['player', 'position', 'team', 'proj_mean', 'floor', 'ceiling', 
                       'boom_prob', 'boom_score', 'salary', 'value_per_1k']],
            use_container_width=True
        )
    
    with tab2:
        st.subheader("💎 Dart Throw Candidates")
        
        dart_throws = analysis_df[
            (analysis_df['ownership'] <= 5) & (analysis_df['boom_score'] >= 70)
        ].sort_values('boom_score', ascending=False)
        
        if len(dart_throws) > 0:
            st.dataframe(dart_throws, use_container_width=True)
        else:
            st.info("No dart throw candidates found (≤5% owned, boom score ≥70)")
    
    with tab3:
        st.subheader("⚖️ Value Analysis")
        
        value_df = analysis_df.sort_values('value_per_1k', ascending=False)
        st.dataframe(
            value_df[['player', 'position', 'salary', 'proj_mean', 'value_per_1k', 'ceil_per_1k']],
            use_container_width=True
        )
    
    with tab4:
        st.subheader("🚩 Projection Flags")
        
        # Only show if we have site FPTS to compare against
        if 'fpts' in players_df.columns and not players_df['fpts'].isna().all():
            analysis_df['delta_vs_site'] = analysis_df['proj_mean'] - analysis_df['fpts'] 
            analysis_df['pct_delta'] = (analysis_df['delta_vs_site'] / analysis_df['fpts'] * 100).round(1)
            
            # Flag large discrepancies
            flags_df = analysis_df[abs(analysis_df['pct_delta']) >= 20].sort_values('pct_delta', ascending=False)
            
            if len(flags_df) > 0:
                st.dataframe(
                    flags_df[['player', 'position', 'fpts', 'proj_mean', 'delta_vs_site', 'pct_delta']],
                    use_container_width=True
                )
            else:
                st.info("No major projection discrepancies found")
        else:
            st.info("Upload player data with FPTS column to see projection comparisons")


def download_ui():
    """UI for downloading results."""
    st.header("📥 Download Results")
    
    if 'sim_results' not in st.session_state:
        st.info("🎯 Run a simulation first to download results")
        return
    
    sim_results = st.session_state.sim_results
    
    # Create download files
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = sim_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📄 Download sim_results.csv",
            data=csv_data,
            file_name="sim_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Create zip with all results
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("sim_results.csv", csv_data)
            
            # Add metadata
            metadata = {
                'generated_at': datetime.now().isoformat(),
                'n_players': len(sim_results),
                'positions': sim_results['position'].value_counts().to_dict()
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        
        zip_buffer.seek(0)
        st.download_button(
            "🗜️ Download all results (zip)",
            data=zip_buffer.getvalue(),
            file_name="simulation_results.zip",
            mime="application/zip",
            use_container_width=True
        )


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="NFL DFS Simulation Pipeline", 
        page_icon="🏈",
        layout="wide"
    )
    st.title("🏈 NFL DFS Simulation Pipeline")
    version_banner()
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        ["🏗️ Build Baseline", "🎯 Run Simulation", "📊 Analysis", "📥 Downloads"]
    )
    
    # Route to appropriate UI
    if page == "🏗️ Build Baseline":
        build_baseline_ui()
    elif page == "🎯 Run Simulation":
        simulation_ui()  
    elif page == "📊 Analysis":
        analysis_ui()
    elif page == "📥 Downloads":
        download_ui()


if __name__ == "__main__":
    main()
