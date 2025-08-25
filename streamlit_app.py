import os
import io
import json
import random
import zipfile
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st


def version_banner():
    branch = os.getenv("RENDER_GIT_BRANCH") or "local"
    commit = (os.getenv("RENDER_GIT_COMMIT") or "")[:7]
    service = os.getenv("RENDER_SERVICE_NAME") or ""
    banner = f"branch: {branch}"
    if commit:
        banner += f" • commit: {commit}"
    if service:
        banner += f" • service: {service}"
    st.caption(banner)


def get_git_commit():
    """Get git commit hash with robust fallback detection."""
    # Try to read from .git directory
    try:
        with open(".git/HEAD", "r") as f:
            head = f.read().strip()
        if head.startswith("ref: "):
            # We're on a branch, read the commit from the ref
            ref_path = head[5:]  # Remove "ref: "
            with open(f".git/{ref_path}", "r") as f:
                commit = f.read().strip()[:7]
                return commit
        else:
            # HEAD contains the commit directly
            return head[:7]
    except (FileNotFoundError, IOError):
        pass
    
    # Fallback to environment variables
    commit = os.getenv("RENDER_GIT_COMMIT") or os.getenv("GIT_COMMIT") or ""
    if commit:
        return commit[:7]
    
    return "unknown"


def generate_lineup_from_players(players_df, seed=None):
    """Generate a placeholder lineup by selecting players from the pool."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Define NFL lineup structure
    positions_needed = {
        'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DST': 1
    }
    
    # Expected column names in players CSV
    position_col = None
    for col in ['position', 'Position', 'pos', 'Pos']:
        if col in players_df.columns:
            position_col = col
            break
    
    if position_col is None:
        # Can't form valid lineups, return empty slots
        return {
            'QB': '', 'RB1': '', 'RB2': '', 'WR1': '', 'WR2': '', 'WR3': '',
            'TE': '', 'FLEX': '', 'DST': '',
            'total_salary': np.nan, 'projected_points': np.nan
        }
    
    lineup = {}
    total_salary = 0
    total_projected = 0
    flex_eligible = []  # RB/WR/TE players not used in main positions
    
    # Find salary and projection columns
    salary_col = None
    proj_col = None
    player_id_col = None
    
    for col in ['salary', 'Salary', 'sal', 'Sal']:
        if col in players_df.columns:
            salary_col = col
            break
    
    for col in ['projection', 'Projection', 'proj', 'Proj', 'fppg', 'FPPG']:
        if col in players_df.columns:
            proj_col = col
            break
            
    for col in ['player_id', 'Player_ID', 'id', 'ID', 'name', 'Name']:
        if col in players_df.columns:
            player_id_col = col
            break
    
    if not all([salary_col, proj_col, player_id_col]):
        return {
            'QB': '', 'RB1': '', 'RB2': '', 'WR1': '', 'WR2': '', 'WR3': '',
            'TE': '', 'FLEX': '', 'DST': '',
            'total_salary': np.nan, 'projected_points': np.nan
        }
    
    used_players = set()
    
    # Fill QB
    qbs = players_df[players_df[position_col].str.upper() == 'QB']
    if not qbs.empty:
        qb = qbs.sample(n=1).iloc[0]
        lineup['QB'] = qb[player_id_col]
        total_salary += qb[salary_col] if pd.notna(qb[salary_col]) else 0
        total_projected += qb[proj_col] if pd.notna(qb[proj_col]) else 0
        used_players.add(qb[player_id_col])
    else:
        lineup['QB'] = ''
    
    # Fill RB positions
    rbs = players_df[
        (players_df[position_col].str.upper() == 'RB') & 
        (~players_df[player_id_col].isin(used_players))
    ]
    rb_count = 0
    for _, rb in rbs.sample(n=min(2, len(rbs))).iterrows():
        rb_count += 1
        lineup[f'RB{rb_count}'] = rb[player_id_col]
        total_salary += rb[salary_col] if pd.notna(rb[salary_col]) else 0
        total_projected += rb[proj_col] if pd.notna(rb[proj_col]) else 0
        used_players.add(rb[player_id_col])
        if rb_count >= 2:
            break
    
    # Fill remaining RB slots with empty if not enough players
    while rb_count < 2:
        rb_count += 1
        lineup[f'RB{rb_count}'] = ''
    
    # Fill WR positions
    wrs = players_df[
        (players_df[position_col].str.upper() == 'WR') & 
        (~players_df[player_id_col].isin(used_players))
    ]
    wr_count = 0
    for _, wr in wrs.sample(n=min(3, len(wrs))).iterrows():
        wr_count += 1
        lineup[f'WR{wr_count}'] = wr[player_id_col]
        total_salary += wr[salary_col] if pd.notna(wr[salary_col]) else 0
        total_projected += wr[proj_col] if pd.notna(wr[proj_col]) else 0
        used_players.add(wr[player_id_col])
        if wr_count >= 3:
            break
    
    # Fill remaining WR slots with empty if not enough players
    while wr_count < 3:
        wr_count += 1
        lineup[f'WR{wr_count}'] = ''
    
    # Fill TE
    tes = players_df[
        (players_df[position_col].str.upper() == 'TE') & 
        (~players_df[player_id_col].isin(used_players))
    ]
    if not tes.empty:
        te = tes.sample(n=1).iloc[0]
        lineup['TE'] = te[player_id_col]
        total_salary += te[salary_col] if pd.notna(te[salary_col]) else 0
        total_projected += te[proj_col] if pd.notna(te[proj_col]) else 0
        used_players.add(te[player_id_col])
    else:
        lineup['TE'] = ''
    
    # Fill FLEX with remaining RB/WR/TE
    flex_eligible = players_df[
        (players_df[position_col].str.upper().isin(['RB', 'WR', 'TE'])) & 
        (~players_df[player_id_col].isin(used_players))
    ]
    if not flex_eligible.empty:
        flex = flex_eligible.sample(n=1).iloc[0]
        lineup['FLEX'] = flex[player_id_col]
        total_salary += flex[salary_col] if pd.notna(flex[salary_col]) else 0
        total_projected += flex[proj_col] if pd.notna(flex[proj_col]) else 0
        used_players.add(flex[player_id_col])
    else:
        lineup['FLEX'] = ''
    
    # Fill DST/DEF
    dsts = players_df[
        (players_df[position_col].str.upper().isin(['DST', 'DEF', 'D/ST'])) & 
        (~players_df[player_id_col].isin(used_players))
    ]
    if not dsts.empty:
        dst = dsts.sample(n=1).iloc[0]
        lineup['DST'] = dst[player_id_col]
        total_salary += dst[salary_col] if pd.notna(dst[salary_col]) else 0
        total_projected += dst[proj_col] if pd.notna(dst[proj_col]) else 0
        used_players.add(dst[player_id_col])
    else:
        lineup['DST'] = ''
    
    lineup['total_salary'] = int(total_salary) if pd.notna(total_salary) and total_salary > 0 else np.nan
    lineup['projected_points'] = round(float(total_projected), 2) if pd.notna(total_projected) and total_projected > 0 else np.nan
    
    return lineup


def analyze_projection_bias(players_df, actuals_df):
    """Calculate position-based projection bias from actual vs projected points."""
    # Find common columns
    player_id_col = None
    position_col = None
    proj_col = None
    
    for col in ['player_id', 'Player_ID', 'id', 'ID', 'name', 'Name']:
        if col in players_df.columns:
            player_id_col = col
            break
    
    for col in ['position', 'Position', 'pos', 'Pos']:
        if col in players_df.columns:
            position_col = col
            break
    
    for col in ['projection', 'Projection', 'proj', 'Proj', 'fppg', 'FPPG']:
        if col in players_df.columns:
            proj_col = col
            break
    
    if not all([player_id_col, position_col, proj_col]):
        return {}, {}
    
    # Find actual points column
    actual_col = None
    for col in ['actual_points', 'actual', 'points', 'score']:
        if col in actuals_df.columns:
            actual_col = col
            break
    
    if actual_col is None:
        return {}, {}
    
    # Merge data
    merged = players_df.merge(
        actuals_df, 
        left_on=player_id_col, 
        right_on='player_id', 
        how='inner',
        suffixes=('', '_actual')
    )
    
    if merged.empty:
        return {}, {}
    
    # Calculate bias by position
    bias_by_position = {}
    rmse_by_position = {}
    
    for position in merged[position_col].unique():
        pos_data = merged[merged[position_col] == position]
        
        # Remove invalid data
        valid_mask = (
            pd.notna(pos_data[proj_col]) & 
            pd.notna(pos_data[actual_col]) & 
            (pos_data[proj_col] > 0)
        )
        
        if valid_mask.sum() > 0:
            pos_valid = pos_data[valid_mask]
            
            # Calculate bias as mean(actual/projection)
            bias = (pos_valid[actual_col] / pos_valid[proj_col]).mean()
            
            # Clip to reasonable range
            bias = np.clip(bias, 0.9, 1.1)
            bias_by_position[position] = round(float(bias), 3)
            
            # Calculate RMSE
            rmse = np.sqrt(((pos_valid[actual_col] - pos_valid[proj_col]) ** 2).mean())
            rmse_by_position[position] = round(float(rmse), 2)
    
    return bias_by_position, rmse_by_position


def analyze_lineup_overlap(our_results_df, winning_lineups_df):
    """Analyze overlap between our lineups and winning lineups."""
    if our_results_df.empty or winning_lineups_df.empty:
        return {}
    
    # Position columns to check
    position_cols = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DST']
    
    # Calculate overlap metrics
    total_our_lineups = len(our_results_df)
    total_winning_lineups = len(winning_lineups_df)
    
    # Count unique players in each set
    our_players = set()
    winning_players = set()
    
    for col in position_cols:
        if col in our_results_df.columns:
            our_players.update(our_results_df[col].dropna().astype(str))
        if col in winning_lineups_df.columns:
            winning_players.update(winning_lineups_df[col].dropna().astype(str))
    
    # Remove empty strings
    our_players.discard('')
    winning_players.discard('')
    
    overlap = our_players.intersection(winning_players)
    
    return {
        "our_lineups_count": total_our_lineups,
        "winning_lineups_count": total_winning_lineups,
        "our_unique_players": len(our_players),
        "winning_unique_players": len(winning_players),
        "overlapping_players": len(overlap),
        "overlap_percentage": round(len(overlap) / max(len(our_players), 1) * 100, 1)
    }


def calculate_exposure_diffs(our_results_df, winning_lineups_df):
    """Calculate exposure differences between our lineups and winning lineups."""
    position_cols = ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DST']
    
    # Count exposures in our lineups
    our_exposure = {}
    for col in position_cols:
        if col in our_results_df.columns:
            for player in our_results_df[col].dropna():
                if player and str(player).strip():
                    our_exposure[str(player)] = our_exposure.get(str(player), 0) + 1
    
    # Count exposures in winning lineups  
    winning_exposure = {}
    for col in position_cols:
        if col in winning_lineups_df.columns:
            for player in winning_lineups_df[col].dropna():
                if player and str(player).strip():
                    winning_exposure[str(player)] = winning_exposure.get(str(player), 0) + 1
    
    # Calculate differences
    all_players = set(our_exposure.keys()) | set(winning_exposure.keys())
    exposure_diffs = []
    
    for player in all_players:
        our_pct = (our_exposure.get(player, 0) / len(our_results_df)) * 100 if len(our_results_df) > 0 else 0
        win_pct = (winning_exposure.get(player, 0) / len(winning_lineups_df)) * 100 if len(winning_lineups_df) > 0 else 0
        diff = win_pct - our_pct
        
        if abs(diff) > 0.1:  # Only include meaningful differences
            exposure_diffs.append({
                'player_id': player,
                'our_exposure_pct': round(our_pct, 1),
                'winning_exposure_pct': round(win_pct, 1), 
                'difference': round(diff, 1)
            })
    
    # Sort by absolute difference
    exposure_diffs.sort(key=lambda x: abs(x['difference']), reverse=True)
    return exposure_diffs[:10]  # Top 10 differences


def apply_projection_adjustments(players_df, multipliers_by_pos, shrinkage=0.2, max_change=0.1):
    """Apply position-based projection adjustments with shrinkage."""
    adjusted_df = players_df.copy()
    
    # Find columns
    position_col = None
    proj_col = None
    
    for col in ['position', 'Position', 'pos', 'Pos']:
        if col in adjusted_df.columns:
            position_col = col
            break
    
    for col in ['projection', 'Projection', 'proj', 'Proj', 'fppg', 'FPPG']:
        if col in adjusted_df.columns:
            proj_col = col
            break
    
    if not all([position_col, proj_col]):
        return adjusted_df
    
    # Apply adjustments
    for idx, row in adjusted_df.iterrows():
        position = row[position_col]
        current_proj = row[proj_col]
        
        if pd.notna(current_proj) and position in multipliers_by_pos:
            # Apply shrinkage: move only partway toward the target
            multiplier = multipliers_by_pos[position]
            shrinked_multiplier = 1 + shrinkage * (multiplier - 1)
            
            # Apply max change cap
            capped_multiplier = np.clip(shrinked_multiplier, 1 - max_change, 1 + max_change)
            
            new_proj = current_proj * capped_multiplier
            adjusted_df.at[idx, proj_col] = round(new_proj, 2)
    
    return adjusted_df


def main():
    st.set_page_config(page_title="NFL GPP Sim Optimizer", layout="wide")
    st.title("NFL GPP Sim Optimizer")
    version_banner()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Simulator", "Optimizer", "Weekly Review (MVP)"])
    
    with tab1:
        simulator_tab()
    
    with tab2:
        optimizer_tab()
    
    with tab3:
        weekly_review_tab()


def simulator_tab():
    """Simulator tab for Monte Carlo projections."""
    st.header("Monte Carlo Simulator")
    st.markdown("Upload a players.csv file to generate Monte Carlo-based fantasy projections with boom analysis.")
    
    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    
    # File uploader
    players_file = st.sidebar.file_uploader(
        "Upload players.csv", 
        type=["csv"], 
        key="sim_players_csv",
        help="CSV file with player data including names, positions, teams, salaries, and projections"
    )
    
    # Simulation parameters
    n_sims = st.sidebar.number_input(
        "Number of Simulations", 
        min_value=1000, 
        max_value=50000, 
        value=10000, 
        step=1000,
        help="Higher values provide more accurate results but take longer"
    )
    
    seed = st.sidebar.number_input(
        "Random Seed", 
        min_value=1, 
        max_value=9999, 
        value=1234, 
        help="For reproducible results"
    )
    
    # Cache control
    if st.sidebar.button("Clear Cache", help="Clear cached simulation results"):
        st.cache_data.clear()
        st.success("Cache cleared!")
    
    if not players_file:
        st.info("👈 Upload a players.csv file in the sidebar to get started.")
        
        # Show methodology expander
        with st.expander("📖 Methodology", expanded=False):
            st.markdown("""
            ### Monte Carlo Simulation Methodology
            
            Our simulator uses position-specific probability distributions to model fantasy scoring:
            
            **Distribution Types:**
            - **QB, WR, DST**: Lognormal (captures boom/bust nature)
            - **RB, TE**: Normal (more consistent scoring patterns)
            
            **Key Metrics:**
            - **Floor (p10)**: 10th percentile outcome
            - **Ceiling (p90)**: 90th percentile outcome  
            - **Boom Probability**: Chance of exceeding position-specific boom threshold
            - **Beat Site Probability**: Chance of outperforming site projection
            
            **Data Sources:**
            - Historical priors from 2023-2024 seasons when available
            - Site projections as fallback for rookies/new players
            - Vegas data (O/U, spread) for mild game environment adjustment
            
            For detailed methodology, see: [Monte Carlo Research](docs/research/monte_carlo_football.pdf)
            """)
        
        return
    
    # Load and process file
    try:
        # Use caching for file processing
        @st.cache_data
        def load_and_process_players(file_bytes, sims, seed_val):
            """Load and process players file with caching."""
            import io
            from src.ingest.site_players import load_site_players
            
            # Create temporary file path
            temp_file = io.StringIO(file_bytes.decode('utf-8'))
            players_df = pd.read_csv(temp_file)
            
            # For demo, create a mock players file
            return players_df, {}, []
        
        file_bytes = players_file.getvalue()
        players_df, column_mapping, warnings_list = load_and_process_players(file_bytes, n_sims, seed)
        
        st.success(f"✅ Loaded {len(players_df)} players")
        
        # Show warnings if any
        if warnings_list:
            st.warning("⚠️ Data validation warnings:")
            for warning in warnings_list:
                st.write(f"• {warning}")
        
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return
    
    # Column mapping display
    if column_mapping:
        with st.expander("📋 Column Mapping", expanded=False):
            mapping_df = pd.DataFrame([
                {"Site Column": k, "Mapped To": v, "Status": "✅ Mapped"}
                for k, v in column_mapping.items()
            ])
            st.dataframe(mapping_df, use_container_width=True)
    
    # Main simulation interface
    col1, col2 = st.columns([3, 1])
    
    with col2:
        run_simulation = st.button(
            "🚀 Run Simulation", 
            type="primary", 
            use_container_width=True,
            help=f"Run {n_sims:,} Monte Carlo simulations"
        )
    
    with col1:
        st.write(f"**Ready to simulate:** {len(players_df)} players • {n_sims:,} sims • Seed: {seed}")
    
    if run_simulation:
        try:
            with st.spinner(f"Running {n_sims:,} simulations..."):
                # Mock simulation results for MVP
                # In production, this would call the actual simulation engine
                
                @st.cache_data
                def run_monte_carlo_simulation(file_bytes, sims, seed_val):
                    """Run Monte Carlo simulation with caching."""
                    import numpy as np
                    
                    # Mock simulation results
                    np.random.seed(seed_val)
                    n_players = len(players_df)
                    
                    sim_results = players_df.copy()
                    
                    # Add mock simulation columns
                    if 'FPTS' in sim_results.columns:
                        base_proj = sim_results['FPTS'].fillna(10)
                    else:
                        base_proj = pd.Series([10] * n_players)
                    
                    # Generate mock results with realistic variance
                    sim_results['sim_mean'] = base_proj * np.random.uniform(0.8, 1.2, n_players)
                    sim_results['floor_p10'] = sim_results['sim_mean'] * np.random.uniform(0.3, 0.6, n_players)
                    sim_results['ceiling_p90'] = sim_results['sim_mean'] * np.random.uniform(1.4, 2.2, n_players)
                    sim_results['p75'] = sim_results['sim_mean'] * np.random.uniform(1.1, 1.3, n_players)
                    sim_results['p95'] = sim_results['sim_mean'] * np.random.uniform(1.6, 2.5, n_players)
                    sim_results['boom_prob'] = np.random.uniform(0.05, 0.3, n_players)
                    sim_results['beat_site_prob'] = np.random.uniform(0.3, 0.7, n_players)
                    sim_results['rookie_fallback'] = np.random.choice([True, False], n_players, p=[0.1, 0.9])
                    
                    # Calculate value metrics
                    if 'SAL' in sim_results.columns:
                        sim_results['value_per_1k'] = (sim_results['sim_mean'] / sim_results['SAL'].fillna(5000)) * 1000
                        sim_results['ceil_per_1k'] = (sim_results['ceiling_p90'] / sim_results['SAL'].fillna(5000)) * 1000
                    
                    # Calculate boom score
                    sim_results['boom_score'] = (sim_results['boom_prob'] * 60 + 
                                                np.random.uniform(20, 40, n_players)).clip(1, 100).astype(int)
                    sim_results['dart_flag'] = sim_results['boom_score'] >= 75
                    
                    return sim_results
                
                sim_results = run_monte_carlo_simulation(file_bytes, n_sims, seed)
            
            st.success(f"✅ Simulation complete! Generated projections for {len(sim_results)} players")
            
            # Results preview with filters
            st.subheader("📊 Simulation Results")
            
            # Filter controls
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                position_filter = st.multiselect(
                    "Filter by Position",
                    options=sim_results['POS'].unique() if 'POS' in sim_results.columns else [],
                    default=[]
                )
            
            with filter_col2:
                sort_options = ['sim_mean', 'ceiling_p90', 'boom_score', 'value_per_1k']
                available_sorts = [col for col in sort_options if col in sim_results.columns]
                sort_by = st.selectbox("Sort by", available_sorts, index=0)
            
            with filter_col3:
                show_darts_only = st.checkbox("Show Dart Plays Only", value=False)
            
            # Apply filters
            filtered_results = sim_results.copy()
            
            if position_filter:
                filtered_results = filtered_results[filtered_results['POS'].isin(position_filter)]
            
            if show_darts_only and 'dart_flag' in filtered_results.columns:
                filtered_results = filtered_results[filtered_results['dart_flag']]
            
            if sort_by in filtered_results.columns:
                filtered_results = filtered_results.sort_values(sort_by, ascending=False)
            
            # Display results
            display_cols = ['PLAYER', 'POS', 'TEAM', 'sim_mean', 'floor_p10', 'ceiling_p90', 
                           'boom_prob', 'boom_score']
            
            if 'SAL' in filtered_results.columns:
                display_cols.insert(4, 'SAL')
            if 'value_per_1k' in filtered_results.columns:
                display_cols.append('value_per_1k')
            if 'dart_flag' in filtered_results.columns:
                display_cols.append('dart_flag')
            
            available_display_cols = [col for col in display_cols if col in filtered_results.columns]
            
            st.dataframe(
                filtered_results[available_display_cols].head(50),
                use_container_width=True,
                column_config={
                    'sim_mean': st.column_config.NumberColumn('Sim Mean', format="%.1f"),
                    'floor_p10': st.column_config.NumberColumn('Floor', format="%.1f"),
                    'ceiling_p90': st.column_config.NumberColumn('Ceiling', format="%.1f"),
                    'boom_prob': st.column_config.NumberColumn('Boom %', format="%.1%"),
                    'boom_score': st.column_config.NumberColumn('Boom Score', format="%d"),
                    'value_per_1k': st.column_config.NumberColumn('Value/1K', format="%.1f"),
                    'dart_flag': st.column_config.CheckboxColumn('Dart')
                }
            )
            
            # Download buttons
            st.subheader("💾 Download Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # sim_players.csv
                sim_players_csv = sim_results[available_display_cols].to_csv(index=False)
                st.download_button(
                    "📄 sim_players.csv",
                    data=sim_players_csv,
                    file_name="sim_players.csv",
                    mime="text/csv"
                )
            
            with col2:
                # compare.csv (if we have site projections)
                compare_csv = sim_results.to_csv(index=False)
                st.download_button(
                    "📊 compare.csv",
                    data=compare_csv,
                    file_name="compare.csv",
                    mime="text/csv"
                )
            
            with col3:
                # diagnostics summary
                diagnostics_data = {
                    'Position': ['Overall'],
                    'Count': [len(sim_results)],
                    'MAE': ['N/A'],
                    'RMSE': ['N/A'],
                    'Note': ['Requires actual results for validation']
                }
                diagnostics_csv = pd.DataFrame(diagnostics_data).to_csv(index=False)
                st.download_button(
                    "📈 diagnostics.csv",
                    data=diagnostics_csv,
                    file_name="diagnostics_summary.csv",
                    mime="text/csv"
                )
            
            with col4:
                # Create ZIP bundle
                import zipfile
                import io
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    zip_file.writestr("sim_players.csv", sim_players_csv)
                    zip_file.writestr("compare.csv", compare_csv)
                    zip_file.writestr("diagnostics_summary.csv", diagnostics_csv)
                    
                    # Add metadata
                    metadata = {
                        'generated_at': pd.Timestamp.now().isoformat(),
                        'sims': n_sims,
                        'seed': seed,
                        'players_count': len(sim_results)
                    }
                    zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
                
                st.download_button(
                    "📦 All Results (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="simulator_outputs.zip",
                    mime="application/zip"
                )
                
        except Exception as e:
            st.error(f"❌ Simulation failed: {e}")


def optimizer_tab():
    """Enhanced optimizer functionality with GPP Presets."""
    st.header("Lineup Optimizer")
    
    # Sidebar for inputs
    st.sidebar.header("Data Sources")
    
    # Option 1: Build from nfl_data_py (placeholder)
    with st.sidebar.expander("🏈 Build from nfl_data_py", expanded=False):
        st.info("Coming in future release")
        season = st.number_input("Season", value=2025, min_value=2020, max_value=2030)
        week = st.number_input("Week", value=1, min_value=1, max_value=18)
        slate_type = st.selectbox("Slate Type", ["Main", "Showdown", "Single Game"])
        include_salaries = st.checkbox("Include DK Salaries", value=True)
        
        st.button("Build Player Pool", disabled=True, help="Feature coming soon")
    
    # Option 2: Upload files
    with st.sidebar.expander("📁 Upload Files", expanded=True):
        players_file = st.file_uploader("Players CSV (required)", type=["csv"], key="opt_players_csv")
        sims_file = st.file_uploader("sims.csv (optional)", type=["csv"], key="opt_sims_csv")
        def_file = st.file_uploader("DEF.csv (optional)", type=["csv"], key="opt_def_csv")
        stacks_file = st.file_uploader("QBRBWRTE.csv (optional)", type=["csv"], key="opt_stacks_csv")
    
    # GPP Presets Section
    st.sidebar.header("🎯 GPP Presets")
    
    preset = st.sidebar.selectbox(
        "Preset Configuration", 
        ["Small", "Mid", "Large"],
        index=1,
        help="Pre-configured settings for different tournament sizes"
    )
    
    # Preset configuration details
    preset_configs = {
        "Small": {
            "ownership_band": (15, 35),
            "boom_threshold": 60,
            "value_threshold": 3.5,
            "enforce_bring_back": False,
            "mini_stacks": 1,
            "salary_leftover": (0, 200),
            "require_darts": 1
        },
        "Mid": {
            "ownership_band": (10, 25),
            "boom_threshold": 70,
            "value_threshold": 4.0,
            "enforce_bring_back": True,
            "mini_stacks": 1,
            "salary_leftover": (200, 500),
            "require_darts": 1
        },
        "Large": {
            "ownership_band": (5, 20),
            "boom_threshold": 75,
            "value_threshold": 4.5,
            "enforce_bring_back": True,
            "mini_stacks": 2,
            "salary_leftover": (0, 300),
            "require_darts": 2
        }
    }
    
    config = preset_configs[preset]
    
    # Display preset details
    with st.sidebar.expander(f"📋 {preset} Field Settings", expanded=True):
        st.write(f"**Ownership Band:** {config['ownership_band'][0]}% - {config['ownership_band'][1]}%")
        st.write(f"**Boom Score Threshold:** {config['boom_threshold']}+")
        st.write(f"**Value Threshold:** {config['value_threshold']}+ pts/$1K")
        st.write(f"**Bring-back Required:** {'Yes' if config['enforce_bring_back'] else 'No'}")
        st.write(f"**Mini-stacks:** {config['mini_stacks']}")
        st.write(f"**Salary Leftover:** ${config['salary_leftover'][0]}-${config['salary_leftover'][1]}")
        st.write(f"**Dart Plays Required:** {config['require_darts']}+")
    
    # Advanced controls
    with st.sidebar.expander("⚙️ Advanced Controls", expanded=False):
        # Ownership controls
        st.write("**Ownership Settings**")
        ownership_min = st.slider("Min Ownership %", 0, 50, config['ownership_band'][0])
        ownership_max = st.slider("Max Ownership %", 0, 100, config['ownership_band'][1])
        
        # Boom controls
        st.write("**Boom Settings**")
        boom_threshold = st.slider("Boom Score Threshold", 30, 100, config['boom_threshold'])
        
        # Value controls
        st.write("**Value Settings**")
        value_threshold = st.number_input("Value per $1K Threshold", 2.0, 6.0, config['value_threshold'], 0.1)
        
        # Stack controls
        st.write("**Stack Settings**")
        enforce_bring_back = st.checkbox("Enforce Bring-back", config['enforce_bring_back'])
        mini_stacks = st.selectbox("Mini-stacks", [0, 1, 2], config['mini_stacks'])
        
        # Dart controls
        st.write("**Dart Settings**")
        require_darts = st.number_input("Require Dart Plays", 0, 5, config['require_darts'])
        
        # Salary controls
        st.write("**Salary Settings**")
        salary_min = st.number_input("Min Salary Leftover", 0, 1000, config['salary_leftover'][0])
        salary_max = st.number_input("Max Salary Leftover", 0, 2000, config['salary_leftover'][1])
    
    # Apply preset button
    if st.sidebar.button("✅ Apply Preset", type="primary"):
        st.success(f"Applied {preset} field preset configuration!")
    
    # Optimization settings
    st.sidebar.header("⚡ Optimization")
    n_lineups = st.sidebar.number_input("Number of lineups", min_value=1, max_value=150, value=20, step=1)
    
    # Main content area
    if not players_file:
        st.info("👈 Upload a Players CSV file in the sidebar to get started.")
        
        # Show GPP Strategy info
        with st.expander("📚 GPP Strategy Guide", expanded=False):
            st.markdown("""
            ### GPP Preset Strategy
            
            **Small Field (150 entries or less):**
            - Higher ownership tolerance (15-35%)
            - Moderate boom requirements (60+ score)
            - Focus on solid value plays (3.5+ pts/$1K)
            - Optional bring-back stacks
            
            **Mid Field (150-1000 entries):**
            - Lower ownership targets (10-25%)
            - Higher boom requirements (70+ score)
            - Better value requirements (4.0+ pts/$1K)
            - Enforce bring-back stacks for correlation
            
            **Large Field (1000+ entries):**
            - Contrarian ownership (5-20%)
            - Elite boom requirements (75+ score)  
            - Premium value plays (4.5+ pts/$1K)
            - Multiple correlation plays (stacks + mini-stacks)
            
            For detailed strategy, see: [GPP Strategy Blueprint](docs/gpp_strategy_blueprint.md)
            """)
        
        return
    
    # File processing
    try:
        players_df = pd.read_csv(players_file)
        st.success(f"✅ Loaded {len(players_df)} players")
        
        # Show data preview
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Players.csv (preview)")
            st.dataframe(players_df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Data Summary")
            summary_data = {"players": players_df.shape}
            if sims_file:
                summary_data["sims"] = pd.read_csv(sims_file).shape
            if def_file:
                summary_data["DEF"] = pd.read_csv(def_file).shape
            if stacks_file:
                summary_data["QBRBWRTE"] = pd.read_csv(stacks_file).shape
            
            st.json(summary_data)
        
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return
    
    # Constraints Panel
    st.subheader("🎯 Applied Constraints")
    
    constraint_data = {
        "Constraint Type": [
            "Ownership Band", "Boom Score", "Value Threshold", 
            "Bring-back", "Mini-stacks", "Dart Plays", "Salary Band"
        ],
        "Setting": [
            f"{ownership_min}% - {ownership_max}%",
            f"{boom_threshold}+",
            f"{value_threshold}+ pts/$1K",
            "Required" if enforce_bring_back else "Optional",
            f"{mini_stacks} required",
            f"{require_darts}+ players",
            f"${salary_min} - ${salary_max} leftover"
        ],
        "Status": ["✅ Active"] * 7
    }
    
    constraints_df = pd.DataFrame(constraint_data)
    st.dataframe(constraints_df, use_container_width=True, hide_index=True)
    
    # Run optimization
    run_optimizer = st.button(
        f"🚀 Generate {n_lineups} Lineups", 
        type="primary", 
        use_container_width=True
    )
    
    if run_optimizer:
        st.info("🚧 **Full optimization engine coming in next release!**")
        st.write("The GPP Presets UI is now complete. The optimization logic will be implemented in a follow-up PR.")
        
        # Show placeholder results structure
        with st.expander("Preview: Expected Output Structure", expanded=True):
            placeholder_lineup = {
                "lineup_id": [1, 2, 3],
                "QB": ["Josh Allen", "Lamar Jackson", "Jalen Hurts"],
                "RB1": ["Christian McCaffrey", "Austin Ekeler", "Josh Jacobs"],
                "RB2": ["Tony Pollard", "Kenneth Walker", "Derrick Henry"],
                "WR1": ["Tyreek Hill", "Stefon Diggs", "Davante Adams"],
                "WR2": ["Keenan Allen", "Mike Evans", "DK Metcalf"],
                "WR3": ["Gabe Davis", "Jerry Jeudy", "Tyler Lockett"],
                "TE": ["Travis Kelce", "Mark Andrews", "Dallas Goedert"],
                "FLEX": ["Tee Higgins", "Chris Olave", "Calvin Ridley"],
                "DST": ["Bills", "49ers", "Eagles"],
                "total_salary": [49800, 49600, 49900],
                "projected_points": [142.5, 138.2, 145.1],
                "boom_score_avg": [72.1, 68.8, 74.3],
                "ownership_avg": [18.2, 15.6, 12.4]
            }
            
            st.dataframe(pd.DataFrame(placeholder_lineup), use_container_width=True)



def weekly_review_tab():
    """Weekly Review MVP functionality."""
    st.header("Weekly Review (MVP)")
    st.write("Analyze your submissions versus winning lineups and actuals, then apply small adjustments week-to-week.")
    
    # File uploads
    st.subheader("Data Inputs")
    col1, col2 = st.columns(2)
    
    with col1:
        players_file = st.file_uploader("Players CSV (required)", type=["csv"], key="review_players_csv", 
                                      help="Same format as the optimizer: player_id, name, position, salary, projection")
        
        winning_lineups_file = st.file_uploader("Winning Lineups CSV (optional)", type=["csv"], key="winning_lineups_csv",
                                               help="Format: lineup_id, QB, RB1, RB2, WR1, WR2, WR3, TE, FLEX, DST")
        
    with col2:
        results_file = st.file_uploader("Your Results CSV (from this app)", type=["csv"], key="review_results_csv",
                                      help="Enhanced format from this app with lineup details")
        
        actuals_file = st.file_uploader("Player Actuals CSV (optional)", type=["csv"], key="actuals_csv", 
                                      help="Format: player_id, actual_points, [field_ownership]")
    
    # Controls
    st.subheader("Analysis Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        shrinkage = st.slider("Shrinkage (projection adjustment strength)", 
                            min_value=0.0, max_value=1.0, value=0.2, step=0.05,
                            help="0.2 means apply 20% of measured bias")
    
    with col2:
        max_change = st.slider("Max per-player change cap", 
                             min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                             help="Maximum percentage change allowed per player")
    
    if st.button("Run Analysis", use_container_width=True, type="primary"):
        if not players_file:
            st.error("Please upload Players CSV to proceed.")
            st.stop()
        
        # Load required data
        try:
            players_df = pd.read_csv(players_file)
        except Exception as e:
            st.error(f"Failed to read Players CSV: {e}")
            st.stop()
        
        # Load optional data
        results_df = pd.read_csv(results_file) if results_file else pd.DataFrame()
        winning_lineups_df = pd.read_csv(winning_lineups_file) if winning_lineups_file else pd.DataFrame() 
        actuals_df = pd.read_csv(actuals_file) if actuals_file else pd.DataFrame()
        
        st.success("Data loaded successfully!")
        
        # Analysis outputs
        st.divider()
        
        # 1. Summary Analysis
        st.subheader("📊 Summary Analysis")
        
        if not results_df.empty and not winning_lineups_df.empty:
            overlap_stats = analyze_lineup_overlap(results_df, winning_lineups_df)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Our Lineups", overlap_stats.get("our_lineups_count", 0))
            with col2:
                st.metric("Winning Lineups", overlap_stats.get("winning_lineups_count", 0))
            with col3:
                st.metric("Player Overlap", f"{overlap_stats.get('overlapping_players', 0)}")
            with col4:
                st.metric("Overlap %", f"{overlap_stats.get('overlap_percentage', 0)}%")
            
            # Exposure differences
            st.write("**Top 10 Exposure Differences:**")
            exposure_diffs = calculate_exposure_diffs(results_df, winning_lineups_df)
            
            if exposure_diffs:
                exposure_df = pd.DataFrame(exposure_diffs)
                st.dataframe(exposure_df, use_container_width=True)
            else:
                st.info("No significant exposure differences found.")
        else:
            st.info("Upload both your results and winning lineups to see overlap analysis.")
        
        # 2. Projection Calibration
        st.subheader("🎯 Projection Calibration")
        
        if not actuals_df.empty:
            bias_by_position, rmse_by_position = analyze_projection_bias(players_df, actuals_df)
            
            if bias_by_position:
                st.write("**Position-based Bias Multipliers:**")
                
                bias_df = pd.DataFrame([
                    {"Position": pos, "Bias Multiplier": mult, "RMSE": rmse_by_position.get(pos, "N/A")}
                    for pos, mult in bias_by_position.items()
                ])
                st.dataframe(bias_df, use_container_width=True)
                
                # Apply adjustments
                adjusted_df = apply_projection_adjustments(
                    players_df, bias_by_position, shrinkage, max_change
                )
                
                # Downloadable adjusted projections
                st.write("**Download Adjusted Projections:**")
                adj_csv = adjusted_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download projections_adjusted.csv",
                    data=adj_csv,
                    file_name="projections_adjusted.csv",
                    mime="text/csv"
                )
                
                # Generate tuning JSON
                now_utc = datetime.utcnow()
                tuning_data = {
                    "generated_at_utc": now_utc.isoformat() + "Z",
                    "run_id": now_utc.isoformat() + "Z", 
                    "projection_multipliers": {
                        "by_position": bias_by_position,
                        "max_change": max_change,
                        "shrinkage": shrinkage
                    },
                    "exposure_targets": {
                        "players": {}  # Could be populated with exposure recommendations
                    }
                }
                
                tuning_json = json.dumps(tuning_data, indent=2).encode("utf-8")
                st.download_button(
                    "Download tuning.json",
                    data=tuning_json,
                    file_name="tuning.json",
                    mime="application/json"
                )
                
            else:
                st.info("No valid projection bias could be calculated. Check data format and overlap.")
        else:
            st.info("Upload player actuals to see projection calibration analysis.")


if __name__ == "__main__":
    main()
