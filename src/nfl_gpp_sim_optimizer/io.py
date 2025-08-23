import os
import pandas as pd

PROTECTED = ["SAL","ML","RST%"]

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def read_players(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize DST position internally
    if "Position" in df.columns:
        df["_POS"] = df["Position"].replace({"D":"DST"}).fillna(df["Position"])
    else:
        raise ValueError("Players.csv must include Position column")
    return df

def write_players(df: pd.DataFrame, path: str):
    # Write back with original Position labels, preserving D for DST
    if "Position" in df.columns and "_POS" in df.columns:
        df_out = df.copy()
        df_out["Position"] = df_out["_POS"].replace({"DST":"D"})
        df_out.drop(columns=["_POS"], inplace=True)
    else:
        df_out = df
    df_out.to_csv(path, index=False)

def read_def(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if path else None

def write_def(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def read_corr(path: str):
    return pd.read_csv(path)
