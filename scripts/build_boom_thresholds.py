#!/usr/bin/env python3
"""
Script to build boom thresholds from historical fantasy point distributions.

Usage:
    python scripts/build_boom_thresholds.py --start 2023 --end 2024 --out data/baseline/boom_thresholds.json --quantile 0.90
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add src to path  
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from src.metrics.sources import get_global_loader
from src.metrics.prep import prepare_weekly_data
from src.ingest.scoring import calculate_boom_thresholds


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(description='Build boom thresholds from historical data')
    parser.add_argument('--start', type=int, required=True, help='Start season (e.g., 2023)')
    parser.add_argument('--end', type=int, required=True, help='End season (e.g., 2024)')
    parser.add_argument('--out', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--quantile', type=float, default=0.90, help='Quantile for boom threshold (default: 0.90)')
    parser.add_argument('--cache-hours', type=int, default=24, help='Data cache duration in hours')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if args.start > args.end:
        logger.error("Start season must be <= end season")
        sys.exit(1)
    
    if not 0.5 <= args.quantile <= 0.99:
        logger.error("Quantile must be between 0.5 and 0.99")
        sys.exit(1)
    
    seasons = list(range(args.start, args.end + 1))
    output_path = Path(args.out)
    
    logger.info(f"Building boom thresholds for seasons {seasons}")
    logger.info(f"Using {args.quantile:.2f} quantile")
    logger.info(f"Output file: {output_path}")
    
    try:
        # Load weekly data
        logger.info("Loading weekly player data...")
        data_loader = get_global_loader(cache_hours=args.cache_hours)
        weekly_raw = data_loader.load_weekly_data(seasons)
        
        if weekly_raw.empty:
            logger.error("No weekly data found")
            sys.exit(1)
        
        # Clean and prepare weekly data
        logger.info("Preparing weekly data...")
        weekly_clean = prepare_weekly_data(weekly_raw)
        
        # Filter to players with sufficient games
        min_games = 4
        player_game_counts = weekly_clean.groupby(['player_display_name', 'season']).size()
        qualified_players = player_game_counts[player_game_counts >= min_games].index
        
        weekly_qualified = weekly_clean.set_index(['player_display_name', 'season']).loc[qualified_players].reset_index()
        
        logger.info(f"Found {len(weekly_qualified)} player-weeks from {len(qualified_players)} qualified players")
        
        # Prepare data for boom threshold calculation
        threshold_data = []
        
        for _, row in weekly_qualified.iterrows():
            player_name = row.get('player_display_name', '')
            position = row.get('position', '')
            week = row.get('week', 1)
            season = row.get('season', 0)
            
            # Get fantasy points (try multiple column names)
            fantasy_points = (
                row.get('fantasy_points_dk') or 
                row.get('fantasy_points_ppr') or 
                row.get('fantasy_points') or
                0.0
            )
            
            if pd.isna(fantasy_points):
                fantasy_points = 0.0
            
            threshold_data.append({
                'player_name': player_name,
                'position': position,
                'week': week,
                'season': season,
                'fantasy_points': float(fantasy_points)
            })
        
        threshold_df = pd.DataFrame(threshold_data)
        
        if threshold_df.empty:
            logger.error("No fantasy points data found")
            sys.exit(1)
        
        # Calculate boom thresholds
        logger.info(f"Calculating boom thresholds at {args.quantile:.2f} quantile...")
        boom_thresholds = calculate_boom_thresholds(threshold_df, quantile=args.quantile)
        
        logger.info("Boom thresholds calculated:")
        for pos, threshold in boom_thresholds.items():
            n_weeks = len(threshold_df[threshold_df['position'] == pos])
            logger.info(f"  {pos}: {threshold:.1f} points ({n_weeks} player-weeks)")
        
        # Save to JSON
        logger.info(f"Saving boom thresholds to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(boom_thresholds, f, indent=2)
        
        logger.info("Boom thresholds saved successfully!")
        
        # Validation
        logger.info("Validating output...")
        with open(output_path, 'r') as f:
            loaded_thresholds = json.load(f)
        
        logger.info(f"Validation successful. File contains {len(loaded_thresholds)} position thresholds")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to build boom thresholds: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()