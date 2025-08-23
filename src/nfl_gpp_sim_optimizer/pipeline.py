import os
from pathlib import Path

from .io import ensure_outdir, read_players, write_players, read_def, read_corr
from .projections import apply_sim_blend
from .chalk import make_chalk_and_pivots
from .optimizer import generate_lineups


def compare_apply(args):
    """Compare and apply sim blend projections."""
    ensure_outdir(args.outdir)
    
    # Read inputs
    df_players = read_players(args.players)
    df_def = read_def(args.defcsv) if args.defcsv else None
    
    # Apply sim blend
    df_adjusted = apply_sim_blend(df_players, args.sims, df_def)
    
    # Apply chalk/pivot logic
    df_final = make_chalk_and_pivots(df_adjusted)
    
    # Write adjusted players
    output_path = os.path.join(args.outdir, "players_adjusted.csv")
    write_players(df_final, output_path)
    print(f"Adjusted players written to: {output_path}")


def build_lineups(args):
    """Build optimal lineups."""
    ensure_outdir(args.outdir)
    
    # Read adjusted players
    df_players = read_players(args.players)
    
    # Read correlation matrix if provided
    corr = None
    if args.corrcsv:
        corr = read_corr(args.corrcsv)
    
    # Generate lineups
    lineups_df = generate_lineups(df_players, corr, args.preset, args.n)
    
    # Write lineups
    output_path = os.path.join(args.outdir, "lineups.csv")
    lineups_df.to_csv(output_path, index=False)
    print(f"Lineups written to: {output_path}")
