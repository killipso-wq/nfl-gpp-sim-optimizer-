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


def main():
    st.set_page_config(page_title="NFL GPP Sim Optimizer", layout="wide")
    st.title("NFL GPP Sim Optimizer")
    version_banner()

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


if __name__ == "__main__":
    main()
