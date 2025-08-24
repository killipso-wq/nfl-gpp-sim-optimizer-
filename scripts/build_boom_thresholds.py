#!/usr/bin/env python3
"""Build boom thresholds from historical fantasy points distributions.

CLI: python scripts/build_boom_thresholds.py --start 2023 --end 2024 --out data/baseline/boom_thresholds.json --quantile 0.90
"""

import argparse
import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics.sources import load_weekly_data
from src.metrics.player_metrics import calculate_player_metrics


def calculate_boom_thresholds(weekly_df, quantile=0.90):
    """Calculate boom thresholds by position from historical data.
    
    Args:
        weekly_df: Weekly player stats DataFrame
        quantile: Quantile to use for boom threshold (0.90 = 90th percentile)
        
    Returns:
        Dictionary of position -> boom threshold
    """
    print(f"Calculating {quantile*100}% boom thresholds by position...")
    
    # Calculate fantasy points for each game
    enhanced_weekly = []
    
    for _, game in weekly_df.iterrows():
        fantasy_points = 0
        position = game.get('position', '')
        
        # DraftKings scoring
        # Passing (1 pt per 25 yards, 4 pts per TD, -1 per INT)
        fantasy_points += game.get('passing_yards', 0) * 0.04
        fantasy_points += game.get('passing_tds', 0) * 4
        fantasy_points -= game.get('interceptions', 0) * 1
        
        # Rushing (1 pt per 10 yards, 6 pts per TD)
        fantasy_points += game.get('rushing_yards', 0) * 0.1
        fantasy_points += game.get('rushing_tds', 0) * 6
        
        # Receiving (1 pt per reception, 1 pt per 10 yards, 6 pts per TD)
        fantasy_points += game.get('receptions', 0) * 1  # PPR
        fantasy_points += game.get('receiving_yards', 0) * 0.1
        fantasy_points += game.get('receiving_tds', 0) * 6
        
        # Bonuses
        if game.get('passing_yards', 0) >= 300:
            fantasy_points += 3
        if game.get('rushing_yards', 0) >= 100:
            fantasy_points += 3
        if game.get('receiving_yards', 0) >= 100:
            fantasy_points += 3
        
        enhanced_weekly.append({
            'position': position,
            'fantasy_points': fantasy_points,
            'season': game.get('season'),
            'week': game.get('week'),
            'player_id': game.get('player_id', ''),
        })
    
    weekly_with_fantasy = pd.DataFrame(enhanced_weekly)
    
    # Calculate thresholds by position
    thresholds = {}
    
    for position in ['QB', 'RB', 'WR', 'TE']:
        pos_data = weekly_with_fantasy[weekly_with_fantasy['position'] == position]
        
        if len(pos_data) > 0:
            threshold = pos_data['fantasy_points'].quantile(quantile)
            thresholds[position] = round(threshold, 2)
            
            print(f"{position}: {threshold:.2f} points ({len(pos_data)} games)")
        else:
            # Default thresholds if no data
            defaults = {'QB': 25.0, 'RB': 20.0, 'WR': 18.0, 'TE': 15.0}
            thresholds[position] = defaults.get(position, 15.0)
            print(f"{position}: {thresholds[position]:.2f} points (default - no data)")
    
    # DST threshold (using mock data)
    thresholds['DST'] = 10.0  # Reasonable DST boom threshold
    print(f"DST: {thresholds['DST']:.2f} points (default)")
    
    return thresholds


def main():
    parser = argparse.ArgumentParser(description='Build boom thresholds from historical fantasy data')
    parser.add_argument('--start', type=int, required=True, help='Start season year (e.g., 2023)')
    parser.add_argument('--end', type=int, required=True, help='End season year (e.g., 2024)')
    parser.add_argument('--out', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--quantile', type=float, default=0.90, help='Quantile for boom threshold (default: 0.90)')
    
    args = parser.parse_args()
    
    if args.quantile < 0.5 or args.quantile > 0.99:
        print("Warning: Quantile should typically be between 0.5 and 0.99")
    
    print(f"Building boom thresholds from {args.start}-{args.end} seasons...")
    print(f"Using {args.quantile*100}% quantile")
    print(f"Output file: {args.out}")
    
    try:
        # Load historical data
        seasons = list(range(args.start, args.end + 1))
        weekly_df = load_weekly_data(seasons)
        
        print(f"Loaded {len(weekly_df)} player-weeks")
        
        # Calculate thresholds
        thresholds = calculate_boom_thresholds(weekly_df, args.quantile)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        
        # Create output data
        output_data = {
            'quantile': args.quantile,
            'seasons': f"{args.start}-{args.end}",
            'generated_games': len(weekly_df),
            'thresholds': thresholds,
            'notes': {
                'scoring': 'DraftKings PPR with 3-point bonuses',
                'definition': f'Boom threshold = {args.quantile*100}% percentile of historical fantasy points',
                'usage': 'Player booms if sim_points >= max(position_threshold, 1.2*site_proj, site_proj+5)'
            }
        }
        
        # Save to JSON
        with open(args.out, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nBoom thresholds saved to {args.out}")
        print("\nThresholds by position:")
        for pos, threshold in thresholds.items():
            print(f"  {pos}: {threshold} points")
        
    except Exception as e:
        print(f"Error building boom thresholds: {e}")
        sys.exit(1)


if __name__ == '__main__':
    import pandas as pd
    main()