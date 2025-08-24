"""
Metrics pipeline orchestrator.

This module coordinates the entire metrics calculation process,
from data loading through final prior generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from .sources import NFLDataLoader
from .prep import clean_pbp_data, prepare_weekly_data
from .team_metrics import calculate_team_metrics_by_week, calculate_team_season_aggregates
from .player_metrics import calculate_qb_metrics, calculate_rb_metrics, calculate_wr_te_metrics, combine_player_metrics

logger = logging.getLogger(__name__)


class MetricsPipeline:
    """
    Orchestrates the entire metrics calculation pipeline.
    """
    
    def __init__(self, cache_hours: int = 24):
        self.data_loader = NFLDataLoader(cache_hours=cache_hours)
        self.seasons_processed = []
        
        # Storage for intermediate results
        self.pbp_data = None
        self.weekly_data = None
        self.schedules_data = None
        self.team_metrics = None
        self.player_metrics = None
    
    def run_full_pipeline(self, seasons: List[int], output_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
        """
        Run the complete metrics pipeline for specified seasons.
        
        Args:
            seasons: List of seasons to process (e.g., [2023, 2024])
            output_dir: Optional directory to save intermediate results
            
        Returns:
            Dictionary containing all calculated metrics
        """
        logger.info(f"Starting full metrics pipeline for seasons {seasons}")
        
        try:
            # Step 1: Load raw data
            results = self.load_raw_data(seasons)
            
            # Step 2: Calculate team metrics
            results.update(self.calculate_team_metrics())
            
            # Step 3: Calculate player metrics
            results.update(self.calculate_player_metrics())
            
            # Step 4: Save intermediate results if requested
            if output_dir:
                self.save_intermediate_results(results, output_dir)
            
            self.seasons_processed = seasons
            logger.info("Metrics pipeline completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def load_raw_data(self, seasons: List[int]) -> Dict[str, pd.DataFrame]:
        """
        Load and clean raw NFL data.
        
        Args:
            seasons: List of seasons to load
            
        Returns:
            Dictionary with cleaned data
        """
        logger.info(f"Loading raw data for seasons {seasons}")
        
        # Load play-by-play data
        logger.info("Loading play-by-play data...")
        raw_pbp = self.data_loader.load_play_by_play(seasons)
        self.pbp_data = clean_pbp_data(raw_pbp) if not raw_pbp.empty else pd.DataFrame()
        
        # Load weekly data
        logger.info("Loading weekly player data...")
        raw_weekly = self.data_loader.load_weekly_data(seasons)
        self.weekly_data = prepare_weekly_data(raw_weekly) if not raw_weekly.empty else pd.DataFrame()
        
        # Load schedules
        logger.info("Loading schedule data...")
        self.schedules_data = self.data_loader.load_schedules(seasons)
        
        results = {
            'pbp_cleaned': self.pbp_data,
            'weekly_cleaned': self.weekly_data,
            'schedules': self.schedules_data
        }
        
        # Log data availability
        for key, df in results.items():
            if not df.empty:
                logger.info(f"Loaded {key}: {len(df)} rows, {len(df.columns)} columns")
            else:
                logger.warning(f"No data loaded for {key}")
        
        return results
    
    def calculate_team_metrics(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate team-level metrics.
        
        Returns:
            Dictionary with team metrics
        """
        logger.info("Calculating team metrics")
        
        if self.pbp_data is None or self.pbp_data.empty:
            logger.warning("No play-by-play data available for team metrics")
            return {}
        
        # Team metrics by week
        team_week_metrics = calculate_team_metrics_by_week(self.pbp_data)
        
        # Aggregate to season level
        team_season_metrics = calculate_team_season_aggregates(team_week_metrics)
        
        self.team_metrics = {
            'team_week': team_week_metrics,
            'team_season': team_season_metrics
        }
        
        return self.team_metrics
    
    def calculate_player_metrics(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate player-level metrics.
        
        Returns:
            Dictionary with player metrics
        """
        logger.info("Calculating player metrics")
        
        if self.pbp_data is None or self.pbp_data.empty:
            logger.warning("No play-by-play data available for player metrics")
            return {}
        
        # QB metrics
        qb_metrics = calculate_qb_metrics(self.pbp_data, self.weekly_data)
        
        # RB metrics  
        rb_metrics = calculate_rb_metrics(self.pbp_data, self.weekly_data)
        
        # WR/TE metrics
        wr_te_metrics = calculate_wr_te_metrics(self.pbp_data, self.weekly_data)
        
        # Combine all player metrics
        combined_player_metrics = combine_player_metrics(qb_metrics, rb_metrics, wr_te_metrics)
        
        self.player_metrics = {
            'qb': qb_metrics,
            'rb': rb_metrics,
            'wr_te': wr_te_metrics,
            'combined': combined_player_metrics
        }
        
        return self.player_metrics
    
    def save_intermediate_results(self, results: Dict[str, pd.DataFrame], output_dir: Path):
        """
        Save intermediate results to CSV files.
        
        Args:
            results: Dictionary of DataFrames to save
            output_dir: Directory to save files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving intermediate results to {output_dir}")
        
        for key, df in results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filename = f"{key}.csv"
                filepath = output_dir / filename
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {key} to {filepath} ({len(df)} rows)")
    
    def get_summary_stats(self) -> Dict[str, Dict]:
        """
        Get summary statistics for calculated metrics.
        
        Returns:
            Dictionary with summary stats
        """
        summary = {}
        
        # Team metrics summary
        if self.team_metrics:
            team_season = self.team_metrics.get('team_season', pd.DataFrame())
            if not team_season.empty:
                summary['team_metrics'] = {
                    'n_team_seasons': len(team_season),
                    'seasons': sorted(team_season['season'].unique().tolist()),
                    'teams': len(team_season['team'].unique()),
                    'avg_pace': team_season['avg_pace'].mean(),
                    'avg_pass_rate': team_season['pass_rate'].mean(),
                    'avg_epa_per_play': team_season['epa_per_play'].mean()
                }
        
        # Player metrics summary
        if self.player_metrics:
            combined = self.player_metrics.get('combined', pd.DataFrame())
            if not combined.empty:
                summary['player_metrics'] = {
                    'n_player_seasons': len(combined),
                    'seasons': sorted(combined['season'].unique().tolist()),
                    'players': len(combined['player_name'].unique()),
                    'position_counts': combined['position'].value_counts().to_dict()
                }
        
        return summary
    
    def clear_cache(self):
        """Clear data loader cache."""
        self.data_loader.clear_cache()
        logger.info("Cleared data loader cache")
    
    def get_cache_info(self) -> Dict:
        """Get cache information."""
        return self.data_loader.get_cache_info()


def run_quick_pipeline(seasons: List[int], output_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to run the full pipeline.
    
    Args:
        seasons: List of seasons to process
        output_dir: Optional output directory
        
    Returns:
        Dictionary with all metrics
    """
    pipeline = MetricsPipeline()
    
    output_path = Path(output_dir) if output_dir else None
    results = pipeline.run_full_pipeline(seasons, output_path)
    
    # Add summary stats
    summary = pipeline.get_summary_stats()
    logger.info(f"Pipeline summary: {summary}")
    
    return results


def validate_pipeline_results(results: Dict[str, pd.DataFrame]) -> List[str]:
    """
    Validate pipeline results and return any issues found.
    
    Args:
        results: Results dictionary from pipeline
        
    Returns:
        List of validation issues (empty if all good)
    """
    issues = []
    
    # Check for required DataFrames
    required_keys = ['pbp_cleaned', 'team_season', 'combined']
    for key in required_keys:
        if key not in results or results[key].empty:
            issues.append(f"Missing or empty {key} data")
    
    # Check team metrics
    if 'team_season' in results and not results['team_season'].empty:
        team_df = results['team_season']
        
        # Check for reasonable values
        if team_df['avg_pace'].min() < 40 or team_df['avg_pace'].max() > 100:
            issues.append("Team pace values outside reasonable range (40-100)")
        
        if team_df['pass_rate'].min() < 0.3 or team_df['pass_rate'].max() > 0.9:
            issues.append("Pass rates outside reasonable range (0.3-0.9)")
    
    # Check player metrics
    if 'combined' in results and not results['combined'].empty:
        player_df = results['combined']
        
        # Check position distribution
        pos_counts = player_df['position'].value_counts()
        if len(pos_counts) < 3:
            issues.append("Too few position groups in player metrics")
        
        # Check for missing key metrics by position
        if 'QB' in pos_counts.index:
            qb_df = player_df[player_df['position'] == 'QB']
            if qb_df['epa_per_db'].isna().all():
                issues.append("All QB EPA per dropback values are missing")
    
    return issues


# Global pipeline instance for reuse
_global_pipeline = None


def get_global_pipeline() -> MetricsPipeline:
    """Get global pipeline instance."""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = MetricsPipeline()
    return _global_pipeline