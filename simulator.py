import numpy as np
import pandas as pd
from typing import Dict, Tuple

POS_STD = {
    "QB": 6.5,
    "RB": 5.5,
    "WR": 5.0,
    "TE": 4.0,
    "DST": 3.0,
}

POSSIBLE_NAME_COLS = ["PLAYER", "Player", "Name", "player", "name"]
POSSIBLE_POS_COLS = ["POS", "Position", "pos", "position"]
POSSIBLE_TEAM_COLS = ["TEAM", "Team", "team"]
POSSIBLE_OPP_COLS = ["OPP", "Opp", "opp"]
POSSIBLE_FPTS_COLS = ["FPTS", "Proj", "Projection", "projection", "proj"]

def _find_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _normalize_pos(pos: str) -> str:
    if not isinstance(pos, str):
        return "UNK"
    p = pos.strip().upper()
    return "DST" if p == "D" else p

def prepare_players(players_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    colmap = {}
    name_col = _find_col(players_df, POSSIBLE_NAME_COLS)
    pos_col = _find_col(players_df, POSSIBLE_POS_COLS)
    team_col = _find_col(players_df, POSSIBLE_TEAM_COLS)
    opp_col = _find_col(players_df, POSSIBLE_OPP_COLS)
    fpts_col = _find_col(players_df, POSSIBLE_FPTS_COLS)

    colmap["PLAYER"] = name_col or "PLAYER"
    colmap["POS"] = pos_col or "POS"
    colmap["TEAM"] = team_col or "TEAM"
    colmap["OPP"] = opp_col or "OPP"
    colmap["FPTS"] = fpts_col or "FPTS"

    df = players_df.copy()
    # Ensure the required columns exist; if not, create placeholders to avoid hard crash
    for std_name, src in colmap.items():
        if src not in df.columns:
            df[std_name] = np.nan
        else:
            if std_name != src:
                df.rename(columns={src: std_name}, inplace=True)

    # Normalize
    df["POS"] = df["POS"].apply(_normalize_pos)
    df["FPTS"] = pd.to_numeric(df["FPTS"], errors="coerce")

    return df, colmap

def infer_sigma(pos: str, fpts: float) -> float:
    """Heuristic variance by position with minor scaling by projection."""
    base = POS_STD.get(pos, 5.0)
    if not np.isfinite(fpts) or fpts <= 0:
        return base
    # Slightly scale with magnitude of projection
    scale = 0.15 if fpts >= 20 else 0.10 if fpts >= 12 else 0.08
    return max(1.0, base * (1.0 + scale))

def simulate_players(players_df: pd.DataFrame, n_sims: int = 10000, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    df, _ = prepare_players(players_df)
    records = []
    flags = []

    for idx, row in df.iterrows():
        name = row.get("PLAYER")
        pos = row.get("POS")
        team = row.get("TEAM")
        opp = row.get("OPP")
        mu = row.get("FPTS")

        # Flags for data quality
        if not np.isfinite(mu):
            flags.append({"PLAYER": name, "flag": "missing_fpts", "detail": "FPTS not provided or non-numeric"})
            mu = 0.0
        if pos == "UNK":
            flags.append({"PLAYER": name, "flag": "unknown_position", "detail": "Position not recognized"})
        if pos == "DST":
            # allow DST negative? Keep non-negative after clipping
            pass

        sigma = infer_sigma(pos, mu)
        if sigma > 2 * max(POS_STD.values()):
            flags.append({"PLAYER": name, "flag": "high_variance", "detail": f"Sigma appears high: {sigma:.2f}"})

        # Draw sims, clip at 0 (no negative fantasy points)
        sims = rng.normal(loc=mu, scale=sigma, size=n_sims)
        sims = np.clip(sims, a_min=0.0, a_max=None)

        rec = {
            "PLAYER": name,
            "POS": pos,
            "TEAM": team,
            "OPP": opp,
            "site_fpts": float(mu),
            "sim_mean": float(np.mean(sims)),
            "sim_median": float(np.median(sims)),
            "sim_std": float(np.std(sims, ddof=1)),
            "sim_p10": float(np.percentile(sims, 10)),
            "sim_p90": float(np.percentile(sims, 90)),
            "sim_p95": float(np.percentile(sims, 95)),
        }
        records.append(rec)

    sim_players_df = pd.DataFrame(records)

    # compare.csv: site vs sim deltas
    compare_df = sim_players_df.copy()
    compare_df["delta_mean"] = compare_df["sim_mean"] - compare_df["site_fpts"]

    # diagnostics_summary.csv: simple run metadata and distribution summary
    diagnostics = {
        "num_players": len(sim_players_df),
        "sims_per_player": int(n_sims),
        "avg_site_fpts": float(np.nanmean(df["FPTS"].values.astype(float))),
        "avg_sim_mean": float(sim_players_df["sim_mean"].mean() if not sim_players_df.empty else 0.0),
        "avg_sim_std": float(sim_players_df["sim_std"].mean() if not sim_players_df.empty else 0.0),
        "pct_missing_fpts": float((~np.isfinite(df["FPTS"]).astype(bool)).mean() * 100.0) if len(df) else 0.0,
    }
    diagnostics_summary_df = pd.DataFrame([diagnostics])

    flags_df = pd.DataFrame(flags) if flags else pd.DataFrame(columns=["PLAYER", "flag", "detail"])

    return sim_players_df, compare_df, diagnostics_summary_df, flags_df

# Convenience wrapper used by Streamlit
def run_simulation(players_df: pd.DataFrame, n_sims: int, seed: int):
    return simulate_players(players_df, n_sims=n_sims, seed=seed)