"""
Minimal CLI module for the NFL GPP Sim Optimizer.
This is a stub implementation to support the Streamlit app.
"""
import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="NFL GPP Sim Optimizer")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # compare-apply command
    compare_parser = subparsers.add_parser("compare-apply", help="Compare and apply projections")
    compare_parser.add_argument("--players", required=True, help="Players CSV file")
    compare_parser.add_argument("--outdir", required=True, help="Output directory")
    compare_parser.add_argument("--sims", help="Sims CSV file")
    compare_parser.add_argument("--defcsv", help="DEF CSV file")
    compare_parser.add_argument("--corrcsv", help="Correlation CSV file")
    
    # build-lineups command
    build_parser = subparsers.add_parser("build-lineups", help="Build lineups")
    build_parser.add_argument("--players", required=True, help="Players CSV file")
    build_parser.add_argument("--preset", required=True, help="Lineup preset")
    build_parser.add_argument("--n", type=int, required=True, help="Number of lineups")
    build_parser.add_argument("--outdir", required=True, help="Output directory")
    build_parser.add_argument("--corrcsv", help="Correlation CSV file")
    
    args = parser.parse_args()
    
    if args.command == "compare-apply":
        return compare_apply(args)
    elif args.command == "build-lineups":
        return build_lineups(args)
    else:
        parser.print_help()
        return 1


def compare_apply(args):
    """Stub implementation for compare-apply."""
    import pandas as pd
    import shutil
    
    print(f"Processing players file: {args.players}")
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    
    # For demo purposes, just copy the players file as "adjusted"
    players_path = Path(args.players)
    if players_path.exists():
        # Read and write back to simulate processing
        df = pd.read_csv(players_path)
        adjusted_path = outdir / "players_adjusted.csv"
        df.to_csv(adjusted_path, index=False)
        print(f"Created adjusted players file: {adjusted_path}")
    else:
        print(f"ERROR: Players file not found: {args.players}")
        return 1
    
    print("compare-apply completed successfully")
    return 0


def build_lineups(args):
    """Stub implementation for build-lineups."""
    import pandas as pd
    import numpy as np
    
    print(f"Building {args.n} lineups with preset: {args.preset}")
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    
    players_path = Path(args.players)
    if not players_path.exists():
        print(f"ERROR: Players file not found: {args.players}")
        return 1
    
    # Create a simple demo lineup
    df_players = pd.read_csv(players_path)
    
    # Create dummy lineup data
    lineup_data = []
    for i in range(min(args.n, 10)):  # Limit to 10 for demo
        lineup = {
            "Lineup": i + 1,
            "QB": f"QB_{i+1}",
            "RB1": f"RB1_{i+1}",
            "RB2": f"RB2_{i+1}",
            "WR1": f"WR1_{i+1}",
            "WR2": f"WR2_{i+1}",
            "WR3": f"WR3_{i+1}",
            "TE": f"TE_{i+1}",
            "FLEX": f"FLEX_{i+1}",
            "DST": f"DST_{i+1}",
            "Salary": 50000 - (i * 100),
            "Projected": 120 + np.random.randn() * 5
        }
        lineup_data.append(lineup)
    
    lineups_df = pd.DataFrame(lineup_data)
    lineups_path = outdir / "lineups.csv"
    lineups_df.to_csv(lineups_path, index=False)
    
    print(f"Created {len(lineup_data)} lineups in: {lineups_path}")
    print("build-lineups completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())