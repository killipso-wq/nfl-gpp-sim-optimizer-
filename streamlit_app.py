import os
import io
import json
import random
import zipfile
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

from simulator import run_simulation


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
    tab1, tab2, tab3 = st.tabs(["Optimizer", "Simulator", "Weekly Review (MVP)"])
    
    with tab1:
        optimizer_tab()
    
    with tab2:
        simulator_tab()
    
    with tab3:
        weekly_review_tab()


def optimizer_tab():
    """Original optimizer functionality."""
    st.sidebar.header("Inputs")

    players_file = st.sidebar.file_uploader("Players CSV (required)", type=["csv"], key="players_csv")
    sims_file = st.sidebar.file_uploader("sims.csv (optional)", type=["csv"], key="sims_csv")
    def_file = st.sidebar.file_uploader("DEF.csv (optional)", type=["csv"], key="def_csv")
    stacks_file = st.sidebar.file_uploader("QBRBWRTE.csv (optional)", type=["csv"], key="stacks_csv")

    preset = st.sidebar.selectbox("Lineup preset", ["se", "mid", "large"], index=1)
    n_lineups = st.sidebar.number_input("Number of lineups", min_value=1, max_value=150, value=20, step=1)

    st.write("Upload files in the left sidebar, choose a preset and quantity, then click Run optimizer.")

    run = st.button("Run optimizer", use_container_width=True)

    if run:
        if not players_file:
            st.error("Please upload Players CSV to proceed.")
            st.stop()

        # Read the uploads that exist
        try:
            players_df = pd.read_csv(players_file)
        except Exception as e:
            st.error(f"Failed to read Players CSV: {e}")
            st.stop()

        st.success("Inputs loaded.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Players.csv (preview)")
            st.dataframe(players_df.head(20), use_container_width=True)
        with c2:
            st.subheader("Shapes")
            st.write(
                {
                    "players": players_df.shape,
                    "sims": pd.read_csv(sims_file).shape if sims_file else None,
                    "DEF": pd.read_csv(def_file).shape if def_file else None,
                    "QBRBWRTE": pd.read_csv(stacks_file).shape if stacks_file else None,
                }
            )

        st.divider()
        st.subheader("Optimization output (placeholder)")

        # Generate enhanced results with lineup details
        now_utc = datetime.utcnow()
        run_id = now_utc.isoformat() + "Z"
        git_commit = get_git_commit()
        seed = random.randint(1000, 9999)
        
        # Generate placeholder lineups from players data
        lineups = []
        for i in range(int(n_lineups)):
            lineup_seed = seed + i  # Unique seed per lineup
            lineup = generate_lineup_from_players(players_df, seed=lineup_seed)
            
            lineup_data = {
                "lineup_id": i + 1,
                "preset": preset,
                "generated_at_utc": run_id,
                "run_id": run_id,
                "git_commit": git_commit,
                "seed": lineup_seed,
                "total_salary": lineup['total_salary'],
                "projected_points": lineup['projected_points'],
                "QB": lineup['QB'],
                "RB1": lineup['RB1'],
                "RB2": lineup['RB2'], 
                "WR1": lineup['WR1'],
                "WR2": lineup['WR2'],
                "WR3": lineup['WR3'],
                "TE": lineup['TE'],
                "FLEX": lineup['FLEX'],
                "DST": lineup['DST']
            }
            lineups.append(lineup_data)
        
        results = pd.DataFrame(lineups)

        st.write("Generated example results (placeholder). Replace with real optimizer logic later.")
        st.dataframe(results.head(20), use_container_width=True)

        # Generate metadata for ZIP
        metadata = {
            "run_id": run_id,
            "preset": preset,
            "generated_at_utc": run_id,
            "git_commit": git_commit,
            "seed": seed,
            "app_version": None,  # Could be populated if versioning is added
            "counts": {
                "lineups_generated": int(n_lineups),
                "players_in_pool": len(players_df)
            }
        }

        # Single CSV download
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results.csv",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Enhanced Zip download with metadata
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("results.csv", csv_bytes)
            zf.writestr("metadata.json", json.dumps(metadata, indent=2).encode("utf-8"))
        zip_buf.seek(0)
        st.download_button(
            "Download all results (zip)",
            data=zip_buf.getvalue(),
            file_name="results.zip",
            mime="application/zip",
            use_container_width=True,
        )


def simulator_tab():
    st.header("Simulator")
    st.write("Run Monte Carlo simulations based on your Players CSV and download results.")

    # Sidebar inputs
    st.sidebar.header("Simulator Inputs")
    players_file = st.sidebar.file_uploader("Players CSV (required)", type=["csv"], key="sim_players_csv")

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        sims = st.number_input("Simulations per player", min_value=1000, max_value=200000, value=20000, step=1000)
    with col_b:
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

    run = st.button("Run simulation", type="primary", use_container_width=True)

    if not run:
        st.info("Upload your Players CSV and click Run simulation to generate outputs.")
        return

    if not players_file:
        st.error("Players CSV is required.")
        st.stop()

    try:
        players_df = pd.read_csv(players_file)
    except Exception as e:
        st.error(f"Failed to read Players CSV: {e}")
        st.stop()

    with st.spinner("Running simulations..."):
        sim_players_df, compare_df, diagnostics_df, flags_df = run_simulation(players_df, n_sims=int(sims), seed=int(seed))

    st.success("Simulation complete!")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("sim_players.csv (preview)")
        st.dataframe(sim_players_df.head(20), use_container_width=True)
    with c2:
        st.subheader("compare.csv (preview)")
        st.dataframe(compare_df.head(20), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("diagnostics_summary.csv")
        st.dataframe(diagnostics_df, use_container_width=True)
    with c4:
        st.subheader("flags.csv")
        st.dataframe(flags_df if not flags_df.empty else pd.DataFrame([{"info": "No flags"}]), use_container_width=True)

    st.divider()
    st.subheader("Downloads")

    sim_players_bytes = sim_players_df.to_csv(index=False).encode("utf-8")
    compare_bytes = compare_df.to_csv(index=False).encode("utf-8")
    diagnostics_bytes = diagnostics_df.to_csv(index=False).encode("utf-8")
    flags_bytes = flags_df.to_csv(index=False).encode("utf-8")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button("Download sim_players.csv", data=sim_players_bytes, file_name="sim_players.csv", mime="text/csv", use_container_width=True)
    with col2:
        st.download_button("Download compare.csv", data=compare_bytes, file_name="compare.csv", mime="text/csv", use_container_width=True)
    with col3:
        st.download_button("Download diagnostics_summary.csv", data=diagnostics_bytes, file_name="diagnostics_summary.csv", mime="text/csv", use_container_width=True)
    with col4:
        st.download_button("Download flags.csv", data=flags_bytes, file_name="flags.csv", mime="text/csv", use_container_width=True)


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
