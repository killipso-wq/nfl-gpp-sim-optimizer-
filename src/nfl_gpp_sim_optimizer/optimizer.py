import pandas as pd
import numpy as np
import pulp

def zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sd = series.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0, index=series.index)
    return (series - mu) / sd

def compute_game_scores(df: pd.DataFrame) -> pd.DataFrame:
    # Requires O/U and SPRD, Team and Opp to group
    if not set(["Team","Opp","O/U","SPRD"]).issubset(df.columns):
        df = df.copy()
        df["game_score"] = 0.0
        return df
    df = df.copy()
    gid = df["Team"].astype(str) + "@" + df["Opp"].astype(str)
    df["_gid"] = gid
    game_agg = df.groupby("_gid").agg({"O/U":"max","SPRD":"max"})
    game_agg["total_z"] = zscore(game_agg["O/U"])
    game_agg["spread_z"] = zscore(game_agg["SPRD"].abs())
    game_agg["game_score"] = game_agg["total_z"] - 0.7*game_agg["spread_z"]
    # Drop existing game_score if present before joining
    if "game_score" in df.columns:
        df = df.drop(columns=["game_score"])
    df = df.join(game_agg["game_score"], on="_gid")
    return df

def _preset_params(preset: str):
    if preset == "se":
        return dict(beta=0.10, min_spend=49700, chalk_cap=2, k_rand=0.05, leave_p=0.0)
    if preset == "mid":
        return dict(beta=0.20, min_spend=49500, chalk_cap=3, k_rand=0.20, leave_p=0.2)
    # large default
    return dict(beta=0.30, min_spend=49200, chalk_cap=4, k_rand=0.30, leave_p=0.4)

def generate_lineups(df_players: pd.DataFrame, corr: pd.DataFrame | None, preset: str, n: int):
    df = df_players.copy()
    if "_POS" not in df.columns:
        df["_POS"] = df.get("Position").replace({"D":"DST"})
    df = compute_game_scores(df)

    params = _preset_params(preset)
    beta = params["beta"]
    min_spend = params["min_spend"]
    chalk_cap = params["chalk_cap"]
    k = params["k_rand"]
    leave_p = params["leave_p"]

    df["idx"] = range(len(df))
    pos = df["_POS"]
    teams = df["Team"].astype(str)
    opps = df["Opp"].astype(str) if "Opp" in df.columns else pd.Series("", index=df.index)
    own = df["RST%"].astype(float)
    salary = df["SAL"].astype(float)
    proj_base = df["Fantasy points"].astype(float)
    value = df["value"] if "value" in df.columns else proj_base / (salary/1000.0)
    # Ensure value is in the dataframe
    df["value"] = value
    stdev = df.get("stdev_sim", pd.Series(0.0, index=df.index)).fillna(0.0).astype(float)
    gs = df.get("game_score", pd.Series(0.0, index=df.index)).astype(float)

    # Identify darts and chalk
    def meets_floor(i):
        p = proj_base.iat[i]; v = value.iat[i]; P = pos.iat[i]
        return ((P=="QB" and (p>=16 or v>=2.2)) or
                (P=="RB" and (p>=9 or v>=2.2)) or
                (P=="WR" and (p>=8 or v>=2.2)) or
                (P=="TE" and (p>=6 or v>=2.2)) or
                (P=="DST" and (p>=6 or v>=2.0)))
    dart_idxs = [i for i in df["idx"] if own.iat[i] <= 5 and meets_floor(i)]
    chalk_idxs = [i for i in df["idx"] if own.iat[i] >= 20]

    rng = np.random.default_rng()

    def build_one(force_leave=False):
        m = pulp.LpProblem("dk_nfl", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"x_{i}", 0, 1, cat='Binary') for i in df["idx"]}
        noise = rng.normal(0.0, k*stdev)
        proj = np.maximum(0.0, proj_base + noise)
        # Objective includes ownership penalty and light game/value terms
        m += pulp.lpSum([x[i]*(proj[i] + 0.8*gs.iat[i] + 1.0*value.iat[i] - beta*own.iat[i]) for i in df["idx"]])

        total_salary = pulp.lpSum([x[i]*salary.iat[i] for i in df["idx"]])
        m += total_salary <= 50000
        if force_leave:
            # encourage leaving salary
            m += total_salary <= 49900
            m += total_salary >= max(min_spend, 49200)
        else:
            m += total_salary >= min_spend

        def pos_count(p):
            return pulp.lpSum([x[i] for i in df["idx"] if pos.iat[i]==p])
        m += pos_count("QB") == 1
        m += pos_count("DST") == 1
        m += pos_count("RB") + pos_count("WR") + pos_count("TE") == 6
        m += pos_count("RB") >= 2
        m += pos_count("WR") >= 3
        m += pos_count("TE") >= 1

        # Team max excluding DST
        for t in teams.unique():
            idxs = [i for i in df["idx"] if teams.iat[i]==t and pos.iat[i] != "DST"]
            if idxs:
                m += pulp.lpSum([x[i] for i in idxs]) <= 4

        # DST interaction rules
        dst_rows = [i for i in df["idx"] if pos.iat[i]=="DST"]
        for i in df["idx"]:
            if pos.iat[i] == "DST":
                continue
            own_t = teams.iat[i]
            # No QB/WR/TE with their own DST (RB + own DST allowed)
            if pos.iat[i] in ("QB","WR","TE"):
                for j in dst_rows:
                    if teams.iat[j] == own_t:
                        m += x[i] + x[j] <= 1
            # No RB vs opposing DST
            if pos.iat[i] == "RB":
                opp = opps.iat[i]
                for j in dst_rows:
                    if opp and teams.iat[j] == opp:
                        m += x[i] + x[j] <= 1

        # Require QB stack (>=1 WR/TE with any chosen QB)
        for t in teams.unique():
            qb_sum = pulp.lpSum([x[i] for i in df["idx"] if pos.iat[i]=="QB" and teams.iat[i]==t])
            pass_catchers = pulp.lpSum([x[i] for i in df["idx"] if teams.iat[i]==t and pos.iat[i] in ("WR","TE")])
            m += pass_catchers >= qb_sum

        if chalk_cap is not None:
            m += pulp.lpSum([x[i] for i in chalk_idxs]) <= chalk_cap
        if dart_idxs:
            m += pulp.lpSum([x[i] for i in dart_idxs]) >= 1

        m.solve(pulp.PULP_CBC_CMD(msg=False))
        return df[df["idx"].map(lambda i: pulp.value(x[i]) > 0.5)]

    # Reports
    from .chalk import make_chalk_and_pivots
    chalk_df, pivots_df = make_chalk_and_pivots(df)

    lineups = []
    used = set()
    for k_iter in range(n):
        force_leave = rng.random() < leave_p
        chosen = build_one(force_leave=force_leave)
        ids = tuple(sorted(chosen.get("Name", chosen.index).tolist()))
        if ids in used:
            chosen = build_one(force_leave=True)
            ids = tuple(sorted(chosen.get("Name", chosen.index).tolist()))
        used.add(ids)
        chosen = chosen.assign(lineup=k_iter)
        chosen["salary_left"] = 50000 - chosen["SAL"].sum()
        chosen["lineup_proj"] = chosen["Fantasy points"].sum()
        chosen["lineup_own_sum"] = chosen["RST%"].sum()
        lineups.append(chosen)

    lineup_df = pd.concat(lineups, ignore_index=True) if lineups else pd.DataFrame()
    return lineup_df, chalk_df, pivots_df
