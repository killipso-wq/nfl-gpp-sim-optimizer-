import pandas as pd
import numpy as np
from .optimizer import compute_game_scores

def identify_chalk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    proj = df["Fantasy points"].astype(float)
    own = df["RST%"].astype(float)
    proj_p = proj.rank(pct=True)
    own_p = own.rank(pct=True)
    df["chalk_score"] = proj_p + own_p
    pos = df.get("_POS", df.get("Position"))
    df["is_chalk"] = False
    df.loc[(pos=="QB") & (own>=15) & (proj_p>=0.70), "is_chalk"] = True
    df.loc[(pos.isin(["RB","WR"])) & (own>=20) & (proj_p>=0.65), "is_chalk"] = True
    df.loc[(pos=="TE") & (own>=15) & (proj_p>=0.60), "is_chalk"] = True
    df.loc[(pos=="DST") & (own>=20) & (proj_p>=0.60), "is_chalk"] = True
    return df

def find_pivots(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_game_scores(df.copy())
    dfc = identify_chalk(df)
    pivots = []
    for _, c in dfc[dfc["is_chalk"]].iterrows():
        same_pos = dfc[dfc["_POS"] == c["_POS"]].copy()
        band = same_pos[
            (same_pos["SAL"].between(c["SAL"]-1000, c["SAL"]+1000)) & (same_pos["Name"] != c["Name"])
        ]
        cand = band[
            (band["RST%"] <= c["RST%"] - 8) &
            (band["value"] >= c["value"] - 0.2) &
            (band["Fantasy points"] >= 0.8 * c["Fantasy points"])
        ].copy()
        if cand.empty:
            continue
        cand["proj_norm"] = cand["Fantasy points"] / cand["Fantasy points"].max()
        cand["val_norm"] = cand["value"] / cand["value"].max()
        own_norm = cand["RST%"] / 100.0
        gs = cand.get("game_score", pd.Series(0.0, index=cand.index))
        gs_norm = (gs - gs.min()) / (gs.max() - gs.min() + 1e-9)

        cand["pivot_score"] = (
            0.4 * cand["proj_norm"] +
            0.35 * cand["val_norm"] +
            0.15 * (1 - own_norm) +
            0.10 * gs_norm
        )
        best = cand.sort_values("pivot_score", ascending=False).head(3)
        for _, p in best.iterrows():
            pivots.append({
                "chalk_name": c.get("Name"),
                "chalk_pos": c.get("_POS"),
                "chalk_sal": c.get("SAL"),
                "chalk_proj": c.get("Fantasy points"),
                "chalk_own": c.get("RST%"),
                "pivot_name": p.get("Name"),
                "pivot_sal": p.get("SAL"),
                "pivot_proj": p.get("Fantasy points"),
                "pivot_own": p.get("RST%"),
                "pivot_value": p.get("value"),
                "pivot_score": p.get("pivot_score"),
            })
    return pd.DataFrame(pivots)

def make_chalk_and_pivots(df: pd.DataFrame):
    chalk_df = identify_chalk(df)
    pivots_df = find_pivots(df)
    return chalk_df, pivots_df
