"""
Value metrics calculation for DFS analysis.

This module calculates various value metrics including salary efficiency,
ceiling value, and site comparison metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_value_metrics(projections_df: pd.DataFrame,
                           site_data_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate value metrics for all players.
    
    Args:
        projections_df: DataFrame with player projections
        site_data_df: Optional DataFrame with site salary/ownership data
        
    Returns:
        DataFrame with value metrics added
    """
    logger.info("Calculating value metrics")
    
    result_df = projections_df.copy()
    
    # Basic value metrics if salary is available
    if 'salary' in result_df.columns:
        result_df = _calculate_salary_efficiency(result_df)
    
    # Site comparison metrics if available
    if site_data_df is not None:
        result_df = _calculate_site_comparison(result_df, site_data_df)
    
    # Position-relative value
    result_df = _calculate_position_relative_value(result_df)
    
    # Value ranks
    result_df = _calculate_value_ranks(result_df)
    
    logger.info(f"Calculated value metrics for {len(result_df)} players")
    
    return result_df


def _calculate_salary_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate salary efficiency metrics."""
    
    # Value per $1K
    df['value_per_1k'] = (df['proj_mean'] / df['salary'] * 1000).round(3)
    
    # Ceiling value per $1K  
    if 'ceiling' in df.columns:
        df['ceil_per_1k'] = (df['ceiling'] / df['salary'] * 1000).round(3)
    
    # Floor value per $1K
    if 'floor' in df.columns:
        df['floor_per_1k'] = (df['floor'] / df['salary'] * 1000).round(3)
    
    # Points per $100 (alternative metric)
    df['pts_per_100'] = (df['proj_mean'] / df['salary'] * 100).round(2)
    
    return df


def _calculate_site_comparison(projections_df: pd.DataFrame, 
                             site_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comparison metrics vs site projections."""
    
    # Merge projections with site data
    merged = projections_df.merge(
        site_df[['player', 'fpts', 'salary', 'ownership']], 
        on='player', 
        how='left',
        suffixes=('', '_site')
    )
    
    # Delta vs site
    merged['delta_vs_site'] = (merged['proj_mean'] - merged['fpts']).round(2)
    
    # Percentage delta
    merged['pct_delta'] = (
        merged['delta_vs_site'] / merged['fpts'] * 100
    ).round(1)
    
    # Absolute delta
    merged['abs_delta'] = merged['delta_vs_site'].abs()
    
    # Site value metrics if we have salary
    if 'salary' in merged.columns:
        merged['site_value_per_1k'] = (merged['fpts'] / merged['salary'] * 1000).round(3)
        merged['value_diff'] = merged['value_per_1k'] - merged['site_value_per_1k']
    
    return merged


def _calculate_position_relative_value(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate position-relative value metrics."""
    
    # Calculate position averages
    for col in ['proj_mean', 'value_per_1k', 'salary']:
        if col in df.columns:
            pos_avg_col = f'{col}_pos_avg'
            pos_rel_col = f'{col}_pos_rel'
            
            df[pos_avg_col] = df.groupby('position')[col].transform('mean')
            df[pos_rel_col] = (df[col] - df[pos_avg_col]).round(2)
    
    return df


def _calculate_value_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate value ranking metrics."""
    
    # Overall ranks
    if 'value_per_1k' in df.columns:
        df['value_rank'] = df['value_per_1k'].rank(ascending=False, method='min').astype(int)
    
    if 'proj_mean' in df.columns:
        df['proj_rank'] = df['proj_mean'].rank(ascending=False, method='min').astype(int)
    
    # Position ranks
    for col in ['proj_mean', 'value_per_1k', 'ceil_per_1k']:
        if col in df.columns:
            rank_col = f'{col}_pos_rank'
            df[rank_col] = df.groupby('position')[col].rank(ascending=False, method='min').astype(int)
    
    return df


def identify_value_plays(df: pd.DataFrame, 
                        value_threshold: float = 3.0,
                        projection_threshold: float = None) -> pd.DataFrame:
    """
    Identify value plays based on multiple criteria.
    
    Args:
        df: DataFrame with projections and value metrics
        value_threshold: Minimum value per $1K to be considered a value play
        projection_threshold: Optional minimum projection threshold
        
    Returns:
        DataFrame filtered to value plays
    """
    logger.info(f"Identifying value plays (value_per_1k >= {value_threshold})")
    
    if 'value_per_1k' not in df.columns:
        logger.warning("No value_per_1k column found")
        return df
    
    # Base value filter
    value_plays = df[df['value_per_1k'] >= value_threshold].copy()
    
    # Optional projection filter
    if projection_threshold is not None:
        value_plays = value_plays[value_plays['proj_mean'] >= projection_threshold]
    
    # Sort by value
    value_plays = value_plays.sort_values('value_per_1k', ascending=False)
    
    logger.info(f"Found {len(value_plays)} value plays")
    
    return value_plays


def calculate_salary_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add salary tier classifications.
    
    Args:
        df: DataFrame with salary information
        
    Returns:
        DataFrame with salary tiers added
    """
    if 'salary' not in df.columns or 'position' not in df.columns:
        return df
    
    logger.info("Calculating salary tiers")
    
    def get_salary_tier(row):
        salary = row['salary']
        position = row['position']
        
        # Position-specific salary tiers
        if position == 'QB':
            if salary >= 8000:
                return 'Elite'
            elif salary >= 6500:
                return 'Mid'
            else:
                return 'Value'
        elif position == 'RB':
            if salary >= 8000:
                return 'Elite' 
            elif salary >= 6000:
                return 'Mid'
            else:
                return 'Value'
        elif position == 'WR':
            if salary >= 8000:
                return 'Elite'
            elif salary >= 6000:
                return 'Mid'
            else:
                return 'Value'
        elif position == 'TE':
            if salary >= 6500:
                return 'Elite'
            elif salary >= 4800:
                return 'Mid'
            else:
                return 'Value'
        elif position == 'K':
            if salary >= 5500:
                return 'Elite'
            elif salary >= 4800:
                return 'Mid'
            else:
                return 'Value'
        elif position == 'DST':
            if salary >= 3500:
                return 'Elite'
            elif salary >= 2800:
                return 'Mid'
            else:
                return 'Value'
        else:
            return 'Unknown'
    
    df['salary_tier'] = df.apply(get_salary_tier, axis=1)
    
    return df


def calculate_ceiling_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze ceiling performance metrics.
    
    Args:
        df: DataFrame with ceiling projections
        
    Returns:
        DataFrame with ceiling analysis
    """
    if 'ceiling' not in df.columns:
        return df
    
    logger.info("Calculating ceiling analysis")
    
    # Ceiling upside (ceiling - projection)
    df['ceiling_upside'] = (df['ceiling'] - df['proj_mean']).round(2)
    
    # Ceiling multiplier
    df['ceiling_multiplier'] = (df['ceiling'] / df['proj_mean']).round(2)
    
    # Position ceiling ranks
    df['ceiling_pos_rank'] = df.groupby('position')['ceiling'].rank(ascending=False, method='min').astype(int)
    
    # High-ceiling flags
    if 'position' in df.columns:
        position_ceiling_thresholds = {
            'QB': 30,
            'RB': 25,
            'WR': 22,
            'TE': 18,
            'K': 15,
            'DST': 15
        }
        
        df['high_ceiling'] = df.apply(
            lambda row: row['ceiling'] >= position_ceiling_thresholds.get(row['position'], 20),
            axis=1
        )
    
    return df


def create_value_summary(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Create summary statistics for value analysis.
    
    Args:
        df: DataFrame with value metrics
        
    Returns:
        Dictionary with value summary statistics
    """
    summary = {}
    
    if 'value_per_1k' in df.columns:
        summary['value'] = {
            'avg_value_per_1k': df['value_per_1k'].mean().round(3),
            'max_value_per_1k': df['value_per_1k'].max().round(3),
            'min_value_per_1k': df['value_per_1k'].min().round(3),
            'value_plays_3plus': len(df[df['value_per_1k'] >= 3.0]),
            'value_plays_2_5plus': len(df[df['value_per_1k'] >= 2.5])
        }
    
    if 'position' in df.columns:
        summary['by_position'] = {}
        for pos in df['position'].unique():
            pos_df = df[df['position'] == pos]
            
            pos_summary = {
                'count': len(pos_df),
                'avg_projection': pos_df['proj_mean'].mean().round(2) if 'proj_mean' in pos_df.columns else None,
                'avg_salary': pos_df['salary'].mean().round(0) if 'salary' in pos_df.columns else None,
                'avg_value': pos_df['value_per_1k'].mean().round(3) if 'value_per_1k' in pos_df.columns else None
            }
            
            summary['by_position'][pos] = pos_summary
    
    if 'salary_tier' in df.columns:
        summary['by_salary_tier'] = {}
        for tier in ['Elite', 'Mid', 'Value']:
            tier_df = df[df['salary_tier'] == tier]
            if len(tier_df) > 0:
                summary['by_salary_tier'][tier] = {
                    'count': len(tier_df),
                    'avg_value': tier_df['value_per_1k'].mean().round(3) if 'value_per_1k' in tier_df.columns else None
                }
    
    return summary


def export_value_analysis(df: pd.DataFrame, output_path: str):
    """
    Export value analysis to CSV with optimized column ordering.
    
    Args:
        df: DataFrame with value analysis
        output_path: Output file path
    """
    logger.info(f"Exporting value analysis to {output_path}")
    
    # Define preferred column order
    base_cols = ['player', 'position', 'team', 'salary', 'proj_mean', 'floor', 'ceiling']
    value_cols = ['value_per_1k', 'ceil_per_1k', 'pts_per_100', 'salary_tier']
    rank_cols = ['value_rank', 'proj_rank', 'value_per_1k_pos_rank']
    site_cols = ['fpts', 'delta_vs_site', 'pct_delta', 'ownership']
    
    # Build column list based on what's available
    export_cols = []
    for col_group in [base_cols, value_cols, rank_cols, site_cols]:
        for col in col_group:
            if col in df.columns and col not in export_cols:
                export_cols.append(col)
    
    # Add any remaining columns
    for col in df.columns:
        if col not in export_cols:
            export_cols.append(col)
    
    # Export with selected columns
    export_df = df[export_cols].copy()
    export_df.to_csv(output_path, index=False)
    
    logger.info(f"Exported {len(export_df)} players with {len(export_cols)} columns")