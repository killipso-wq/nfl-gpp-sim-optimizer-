#!/usr/bin/env python3
"""Build baseline team and player priors from historical NFL data.

CLI: python scripts/build_baseline.py --start 2023 --end 2024 --out data
"""

import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics.pipeline import run_baseline_pipeline, create_team_priors, create_player_priors, save_baseline_data


def main():
    parser = argparse.ArgumentParser(description='Build baseline priors from historical NFL data')
    parser.add_argument('--start', type=int, required=True, help='Start season year (e.g., 2023)')  
    parser.add_argument('--end', type=int, required=True, help='End season year (e.g., 2024)')
    parser.add_argument('--out', type=str, required=True, help='Output directory for baseline data')
    
    args = parser.parse_args()
    
    print(f"Building baseline from {args.start}-{args.end} seasons...")
    print(f"Output directory: {args.out}")
    
    try:
        # Run the data pipeline
        team_metrics_df, player_metrics_df, dst_metrics_df = run_baseline_pipeline(
            args.start, args.end
        )
        
        print(f"Generated metrics for {len(team_metrics_df)} team-seasons")
        print(f"Generated metrics for {len(player_metrics_df)} players")  
        print(f"Generated metrics for {len(dst_metrics_df)} defenses")
        
        # Combine player and DST metrics
        all_player_metrics = pd.concat([player_metrics_df, dst_metrics_df], ignore_index=True)
        
        # Create priors
        print("Creating team priors...")
        team_priors_df = create_team_priors(team_metrics_df)
        
        print("Creating player priors...")
        player_priors_df = create_player_priors(all_player_metrics)
        
        # Create output directory
        baseline_dir = os.path.join(args.out, 'baseline')
        os.makedirs(baseline_dir, exist_ok=True)
        
        # Save data
        save_baseline_data(team_priors_df, player_priors_df, baseline_dir)
        
        print("\nBaseline build complete!")
        print(f"Team priors: {len(team_priors_df)} teams")
        print(f"Player priors: {len(player_priors_df)} players")
        print(f"Files saved to {baseline_dir}/")
        
    except Exception as e:
        print(f"Error building baseline: {e}")
        sys.exit(1)


if __name__ == '__main__':
    import pandas as pd  # Import here to avoid issues if not available at module level
    main()