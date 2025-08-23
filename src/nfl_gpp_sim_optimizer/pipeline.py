import os
from .projections import apply_sim_blend
from .optimizer import generate_lineups
from .io import read_players, read_def, read_corr, write_players, write_def, ensure_outdir


def compare_apply(args):
    """Compare original vs sim-blended projections and apply the blend."""
    # Read inputs
    df_players = read_players(args.players)
    df_def = read_def(args.defcsv) if args.defcsv else None
    
    # Apply sim blend
    df_blended, df_def_adj, delta = apply_sim_blend(df_players, args.sims, df_def)
    
    # Ensure output directory
    ensure_outdir(args.outdir)
    
    # Write outputs
    write_players(df_blended, os.path.join(args.outdir, "players_blended.csv"))
    if df_def_adj is not None:
        write_def(df_def_adj, os.path.join(args.outdir, "def_adjusted.csv"))
    delta.to_csv(os.path.join(args.outdir, "projection_deltas.csv"), index=False)
    
    print(f"Applied sim blend. Results written to {args.outdir}/")
    print(f"Average projection change: {delta['delta'].mean():.2f}")


def build_lineups(args):
    """Build optimized lineups with stacking rules."""
    # Read inputs
    df_players = read_players(args.players)
    corr = read_corr(args.corrcsv) if args.corrcsv else None
    
    # Generate lineups
    lineup_df, chalk_df, pivots_df = generate_lineups(df_players, corr, args.preset, args.n)
    
    # Ensure output directory
    ensure_outdir(args.outdir)
    
    # Write outputs
    lineup_df.to_csv(os.path.join(args.outdir, "lineups.csv"), index=False)
    chalk_df[chalk_df["is_chalk"]].to_csv(os.path.join(args.outdir, "chalk_players.csv"), index=False)
    pivots_df.to_csv(os.path.join(args.outdir, "pivot_options.csv"), index=False)
    
    print(f"Generated {args.n} lineups using {args.preset} preset")
    print(f"Results written to {args.outdir}/")
    print(f"Chalk players: {chalk_df['is_chalk'].sum()}")
    print(f"Pivot options: {len(pivots_df)}")
