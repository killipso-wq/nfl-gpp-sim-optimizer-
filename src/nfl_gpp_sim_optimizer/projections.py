import pandas as pd
import numpy as np

def round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))

def apply_sim_blend(df_players: pd.DataFrame, sims_path: str | None, df_def: pd.DataFrame | None):
    df = df_players.copy()
    # Keep a copy of original Fpts
    if "Fantasy points" not in df.columns:
        raise ValueError("Players.csv must include 'Fantasy points' column")
    df["Fantasy points_orig"] = df["Fantasy points"]

    sims = None
    if sims_path:
        sims = pd.read_csv(sims_path)

    # Join key
    join_key = None
    if sims is not None:
        if "player_id" in df.columns and "player_id" in sims.columns:
            join_key = "player_id"
        elif "Name" in df.columns and "Name" in sims.columns:
            if "_POS" in df.columns and ("_POS" in sims.columns or "Position" in sims.columns):
                if "Position" in sims.columns and "_POS" not in sims.columns:
                    sims = sims.rename(columns={"Position":"_POS"})
                join_key = ["Name","_POS"]
            else:
                join_key = ["Name"]

    if sims is not None and join_key is not None:
        df = df.merge(
            sims.rename(columns={"mean":"sim_mean"}),
            how="left",
            on=join_key
        )
    else:
        for col in ["sim_mean","stdev_sim","floor_sim","ceil_sim"]:
            if col not in df.columns:
                df[col] = np.nan

    # Blend and round
    sim = df["sim_mean"].where(df["sim_mean"].notna(), df["Fantasy points"]) if "sim_mean" in df.columns else df["Fantasy points"]
    orig = df["Fantasy points"].where(df["Fantasy points"].notna(), sim)
    proj_final_raw = 0.5*sim + 0.5*orig
    df["Fantasy points"] = proj_final_raw.apply(round_half_up)

    # Recompute value
    if "SAL" not in df.columns:
        raise ValueError("Players.csv must include 'SAL'")
    df["value"] = df["Fantasy points"] / (df["SAL"] / 1000.0)

    # DEF sync (if provided)
    df_def_adj = None
    if df_def is not None:
        if "Team" in df.columns and "_POS" in df.columns:
            dst_rows = df[df["_POS"]=="DST"][["Team","Fantasy points"]]
            df_def_adj = df_def.copy()
            if "Team" in df_def_adj.columns:
                df_def_adj = df_def_adj.drop(columns=["Fantasy points"], errors="ignore").merge(dst_rows, on="Team", how="left")

    # Delta report
    delta = pd.DataFrame({
        "Name": df.get("Name", pd.Series(dtype=str)),
        "Position": df.get("Position", pd.Series(dtype=str)),
        "Fantasy points_orig": df["Fantasy points_orig"],
        "Fantasy points_new": df["Fantasy points"],
        "delta": df["Fantasy points"] - df["Fantasy points_orig"]
    })
    return df, df_def_adj, delta
