#!/usr/bin/env python3
"""
Script to build baseline team and player priors from historical NFL data.

Usage:
    python scripts/build_baseline.py --start 2023 --end 2024 --out data
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.metrics.pipeline import MetricsPipeline
from src.projections.prior_builder import PriorBuilder


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(description='Build baseline priors from historical NFL data')
    parser.add_argument('--start', type=int, required=True, help='Start season (e.g., 2023)')
    parser.add_argument('--end', type=int, required=True, help='End season (e.g., 2024)')  
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--cache-hours', type=int, default=24, help='Data cache duration in hours')
    parser.add_argument('--shrinkage', type=float, default=0.3, help='Shrinkage strength (0.0-1.0)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if args.start > args.end:
        logger.error("Start season must be <= end season")
        sys.exit(1)
    
    if not 0.0 <= args.shrinkage <= 1.0:
        logger.error("Shrinkage must be between 0.0 and 1.0")
        sys.exit(1)
    
    seasons = list(range(args.start, args.end + 1))
    output_dir = Path(args.out)
    
    logger.info(f"Building baseline for seasons {seasons}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Shrinkage strength: {args.shrinkage}")
    
    try:
        # Step 1: Run metrics pipeline
        logger.info("=" * 50)
        logger.info("STEP 1: Running metrics pipeline")
        logger.info("=" * 50)
        
        pipeline = MetricsPipeline(cache_hours=args.cache_hours)
        metrics_results = pipeline.run_full_pipeline(
            seasons=seasons,
            output_dir=output_dir / 'intermediate'
        )
        
        # Step 2: Build priors
        logger.info("=" * 50)
        logger.info("STEP 2: Building priors")
        logger.info("=" * 50)
        
        baseline_dir = output_dir / 'baseline'
        baseline_dir.mkdir(parents=True, exist_ok=True)
        
        prior_builder = PriorBuilder(shrinkage_strength=args.shrinkage)
        
        # Build team priors
        team_metrics = metrics_results.get('team_season')
        if team_metrics is not None and not team_metrics.empty:
            team_priors = prior_builder.build_team_priors(team_metrics)
            logger.info(f"Built priors for {len(team_priors)} teams")
        else:
            logger.warning("No team metrics available")
            team_priors = None
        
        # Build player priors
        player_metrics = metrics_results.get('combined')  
        if player_metrics is not None and not player_metrics.empty:
            player_priors = prior_builder.build_player_priors(player_metrics, team_priors)
            logger.info(f"Built priors for {len(player_priors)} players")
        else:
            logger.warning("No player metrics available")
            player_priors = None
        
        # Step 3: Save priors
        logger.info("=" * 50)
        logger.info("STEP 3: Saving priors")
        logger.info("=" * 50)
        
        prior_builder.save_priors(baseline_dir)
        
        # Step 4: Summary
        logger.info("=" * 50)
        logger.info("STEP 4: Summary")
        logger.info("=" * 50)
        
        summary = prior_builder.get_summary()
        for section, stats in summary.items():
            logger.info(f"{section}: {stats}")
        
        logger.info(f"Baseline building completed successfully!")
        logger.info(f"Files saved to: {baseline_dir}")
        
        # List output files
        for file_path in baseline_dir.glob('*.csv'):
            logger.info(f"  - {file_path.name}: {len(pd.read_csv(file_path))} rows")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()