"""
Diagnostics and validation for projection accuracy.

This module provides tools to validate projection accuracy against site
projections and historical performance, flagging major discrepancies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_projection_diagnostics(projections_df: pd.DataFrame,
                                   site_df: Optional[pd.DataFrame] = None,
                                   historical_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive projection diagnostics.
    
    Args:
        projections_df: DataFrame with our projections
        site_df: Optional DataFrame with site projections for comparison
        historical_df: Optional DataFrame with historical actual results
        
    Returns:
        Dictionary with diagnostic metrics
    """
    logger.info("Calculating projection diagnostics")
    
    diagnostics = {
        'summary': _calculate_basic_diagnostics(projections_df),
        'by_position': _calculate_position_diagnostics(projections_df)
    }
    
    # Site comparison diagnostics
    if site_df is not None:
        diagnostics['vs_site'] = _calculate_site_comparison_diagnostics(projections_df, site_df)
    
    # Historical validation diagnostics
    if historical_df is not None:
        diagnostics['vs_historical'] = _calculate_historical_diagnostics(projections_df, historical_df)
    
    # Coverage analysis
    diagnostics['coverage'] = _calculate_coverage_analysis(projections_df, site_df)
    
    logger.info("Projection diagnostics calculated")
    
    return diagnostics


def _calculate_basic_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic projection statistics."""
    
    diagnostics = {
        'n_players': len(df),
        'n_positions': df['position'].nunique() if 'position' in df.columns else 0,
        'total_projected_points': df['proj_mean'].sum() if 'proj_mean' in df.columns else 0
    }
    
    if 'proj_mean' in df.columns:
        diagnostics.update({
            'avg_projection': df['proj_mean'].mean().round(2),
            'median_projection': df['proj_mean'].median().round(2),
            'std_projection': df['proj_mean'].std().round(2),
            'min_projection': df['proj_mean'].min().round(2),
            'max_projection': df['proj_mean'].max().round(2)
        })
    
    # Range analysis (floor to ceiling)
    if all(col in df.columns for col in ['floor', 'ceiling']):
        df_temp = df.dropna(subset=['floor', 'ceiling'])
        if len(df_temp) > 0:
            diagnostics.update({
                'avg_range': (df_temp['ceiling'] - df_temp['floor']).mean().round(2),
                'max_range': (df_temp['ceiling'] - df_temp['floor']).max().round(2),
                'min_range': (df_temp['ceiling'] - df_temp['floor']).min().round(2)
            })
    
    return diagnostics


def _calculate_position_diagnostics(df: pd.DataFrame) -> Dict[str, Dict]:
    """Calculate diagnostics by position."""
    
    if 'position' not in df.columns:
        return {}
    
    position_diagnostics = {}
    
    for pos in df['position'].unique():
        pos_df = df[df['position'] == pos]
        
        pos_diag = {
            'count': len(pos_df),
            'avg_projection': pos_df['proj_mean'].mean().round(2) if 'proj_mean' in pos_df.columns else None,
            'std_projection': pos_df['proj_mean'].std().round(2) if 'proj_mean' in pos_df.columns else None
        }
        
        # Range analysis by position
        if all(col in pos_df.columns for col in ['floor', 'ceiling']):
            pos_temp = pos_df.dropna(subset=['floor', 'ceiling'])
            if len(pos_temp) > 0:
                pos_diag.update({
                    'avg_range': (pos_temp['ceiling'] - pos_temp['floor']).mean().round(2),
                    'volatility': (pos_temp['ceiling'] - pos_temp['floor']).std().round(2)
                })
        
        position_diagnostics[pos] = pos_diag
    
    return position_diagnostics


def _calculate_site_comparison_diagnostics(projections_df: pd.DataFrame, 
                                         site_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate diagnostics comparing our projections to site projections."""
    
    # Merge dataframes
    merged = projections_df.merge(
        site_df[['player', 'fpts']], 
        on='player', 
        how='inner'
    )
    
    if merged.empty or 'proj_mean' not in merged.columns:
        return {'error': 'No matching players or missing projection columns'}
    
    # Calculate differences
    merged['diff'] = merged['proj_mean'] - merged['fpts']
    merged['abs_diff'] = merged['diff'].abs()
    merged['pct_diff'] = (merged['diff'] / merged['fpts'] * 100)
    
    # Overall metrics
    diagnostics = {
        'n_comparisons': len(merged),
        'mae': merged['abs_diff'].mean().round(3),
        'rmse': np.sqrt((merged['diff'] ** 2).mean()).round(3),
        'mean_bias': merged['diff'].mean().round(3),
        'correlation': merged['proj_mean'].corr(merged['fpts']).round(3),
        'avg_pct_diff': merged['pct_diff'].mean().round(1)
    }
    
    # Distribution of differences
    diagnostics['difference_distribution'] = {
        'within_2pts': len(merged[merged['abs_diff'] <= 2]) / len(merged) * 100,
        'within_5pts': len(merged[merged['abs_diff'] <= 5]) / len(merged) * 100,
        'large_discrepancies_20pct': len(merged[merged['pct_diff'].abs() >= 20]) / len(merged) * 100
    }
    
    # By position
    diagnostics['by_position'] = {}
    for pos in merged['position'].unique():
        pos_data = merged[merged['position'] == pos]
        
        diagnostics['by_position'][pos] = {
            'count': len(pos_data),
            'mae': pos_data['abs_diff'].mean().round(3),
            'bias': pos_data['diff'].mean().round(3),
            'correlation': pos_data['proj_mean'].corr(pos_data['fpts']).round(3) if len(pos_data) > 1 else None
        }
    
    return diagnostics


def _calculate_historical_diagnostics(projections_df: pd.DataFrame,
                                    historical_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate diagnostics against historical actual results."""
    
    # This would be implemented if historical actual results are available
    # For now, return placeholder
    return {
        'note': 'Historical validation would require actual game results',
        'available_weeks': len(historical_df) if historical_df is not None else 0
    }


def _calculate_coverage_analysis(projections_df: pd.DataFrame,
                                site_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Calculate how often site projections fall within our projection ranges."""
    
    if site_df is None or not all(col in projections_df.columns for col in ['floor', 'ceiling']):
        return {'note': 'Coverage analysis requires site data and floor/ceiling projections'}
    
    # Merge data
    merged = projections_df.merge(
        site_df[['player', 'fpts']], 
        on='player', 
        how='inner'
    )
    
    if merged.empty:
        return {'error': 'No matching players for coverage analysis'}
    
    # Check coverage
    merged['within_range'] = (
        (merged['fpts'] >= merged['floor']) & 
        (merged['fpts'] <= merged['ceiling'])
    )
    
    coverage_rate = merged['within_range'].mean() * 100
    
    diagnostics = {
        'overall_coverage_rate': coverage_rate.round(1),
        'n_covered': merged['within_range'].sum(),
        'n_total': len(merged)
    }
    
    # Coverage by position
    diagnostics['by_position'] = {}
    for pos in merged['position'].unique():
        pos_data = merged[merged['position'] == pos]
        pos_coverage = pos_data['within_range'].mean() * 100
        
        diagnostics['by_position'][pos] = {
            'coverage_rate': pos_coverage.round(1),
            'n_covered': pos_data['within_range'].sum(),
            'n_total': len(pos_data)
        }
    
    return diagnostics


def identify_projection_flags(projections_df: pd.DataFrame,
                            site_df: Optional[pd.DataFrame] = None,
                            large_diff_threshold: float = 5.0,
                            pct_diff_threshold: float = 25.0) -> pd.DataFrame:
    """
    Identify players with significant projection discrepancies.
    
    Args:
        projections_df: DataFrame with our projections
        site_df: Optional DataFrame with site projections
        large_diff_threshold: Absolute difference threshold for flagging
        pct_diff_threshold: Percentage difference threshold for flagging
        
    Returns:
        DataFrame with flagged players
    """
    logger.info(f"Identifying projection flags (diff >= {large_diff_threshold} or {pct_diff_threshold}%)")
    
    if site_df is None:
        logger.warning("No site data available for flag identification")
        return pd.DataFrame()
    
    # Merge data
    merged = projections_df.merge(
        site_df[['player', 'fpts', 'salary', 'ownership']], 
        on='player', 
        how='inner'
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    # Calculate differences
    merged['diff'] = merged['proj_mean'] - merged['fpts']
    merged['abs_diff'] = merged['diff'].abs()
    merged['pct_diff'] = (merged['diff'] / merged['fpts'] * 100)
    
    # Flag conditions
    large_diff_flag = merged['abs_diff'] >= large_diff_threshold
    pct_diff_flag = merged['pct_diff'].abs() >= pct_diff_threshold
    
    # Combine flags
    flagged = merged[large_diff_flag | pct_diff_flag].copy()
    
    # Add flag reasons
    flagged['flag_reason'] = ''
    flagged.loc[flagged['abs_diff'] >= large_diff_threshold, 'flag_reason'] += 'Large Absolute Diff; '
    flagged.loc[flagged['pct_diff'].abs() >= pct_diff_threshold, 'flag_reason'] += 'Large Percentage Diff; '
    flagged['flag_reason'] = flagged['flag_reason'].str.rstrip('; ')
    
    # Sort by absolute difference (largest discrepancies first)
    flagged = flagged.sort_values('abs_diff', ascending=False)
    
    logger.info(f"Identified {len(flagged)} players with projection flags")
    
    return flagged


def create_diagnostics_summary_table(diagnostics: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a summary table of key diagnostic metrics.
    
    Args:
        diagnostics: Dictionary from calculate_projection_diagnostics
        
    Returns:
        DataFrame with summary metrics
    """
    summary_data = []
    
    # Overall metrics
    if 'summary' in diagnostics:
        summary_data.append({
            'Category': 'Overall',
            'Metric': 'Number of Players',
            'Value': diagnostics['summary'].get('n_players', 0)
        })
        
        summary_data.append({
            'Category': 'Overall',
            'Metric': 'Average Projection',
            'Value': diagnostics['summary'].get('avg_projection', 0)
        })
    
    # Site comparison metrics
    if 'vs_site' in diagnostics:
        vs_site = diagnostics['vs_site']
        
        summary_data.extend([
            {
                'Category': 'vs Site',
                'Metric': 'MAE',
                'Value': vs_site.get('mae', 0)
            },
            {
                'Category': 'vs Site', 
                'Metric': 'RMSE',
                'Value': vs_site.get('rmse', 0)
            },
            {
                'Category': 'vs Site',
                'Metric': 'Correlation',
                'Value': vs_site.get('correlation', 0)
            },
            {
                'Category': 'vs Site',
                'Metric': 'Mean Bias',
                'Value': vs_site.get('mean_bias', 0)
            }
        ])
    
    # Coverage metrics
    if 'coverage' in diagnostics and 'overall_coverage_rate' in diagnostics['coverage']:
        summary_data.append({
            'Category': 'Coverage',
            'Metric': 'Overall Coverage Rate (%)',
            'Value': diagnostics['coverage']['overall_coverage_rate']
        })
    
    return pd.DataFrame(summary_data)


def export_diagnostics_report(diagnostics: Dict[str, Any], 
                            flags_df: pd.DataFrame,
                            output_dir: str):
    """
    Export comprehensive diagnostics report.
    
    Args:
        diagnostics: Dictionary from calculate_projection_diagnostics
        flags_df: DataFrame with flagged players
        output_dir: Output directory path
    """
    from pathlib import Path
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting diagnostics report to {output_path}")
    
    # Export diagnostics JSON
    diagnostics_path = output_path / 'diagnostics_summary.json'
    with open(diagnostics_path, 'w') as f:
        json.dump(diagnostics, f, indent=2, default=str)
    
    # Export summary table
    summary_df = create_diagnostics_summary_table(diagnostics)
    summary_path = output_path / 'diagnostics_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    # Export flags
    if not flags_df.empty:
        flags_path = output_path / 'flags.csv'
        
        flag_cols = [
            'player', 'position', 'team', 'proj_mean', 'fpts', 
            'diff', 'pct_diff', 'salary', 'ownership', 'flag_reason'
        ]
        
        # Keep only available columns
        available_cols = [col for col in flag_cols if col in flags_df.columns]
        flags_df[available_cols].to_csv(flags_path, index=False)
        
        logger.info(f"Exported {len(flags_df)} flagged players")
    
    logger.info("Diagnostics report exported successfully")


def validate_projection_reasonableness(df: pd.DataFrame) -> List[str]:
    """
    Validate that projections are within reasonable ranges.
    
    Args:
        df: DataFrame with projections
        
    Returns:
        List of validation issues found
    """
    issues = []
    
    if 'proj_mean' not in df.columns:
        issues.append("Missing proj_mean column")
        return issues
    
    # Check for negative projections
    negative_count = (df['proj_mean'] < 0).sum()
    if negative_count > 0:
        issues.append(f"{negative_count} players have negative projections")
    
    # Check for extremely high projections by position
    position_limits = {
        'QB': 45,
        'RB': 35,
        'WR': 30,
        'TE': 25,
        'K': 20,
        'DST': 25
    }
    
    if 'position' in df.columns:
        for pos, limit in position_limits.items():
            pos_df = df[df['position'] == pos]
            if len(pos_df) > 0:
                high_count = (pos_df['proj_mean'] > limit).sum()
                if high_count > 0:
                    issues.append(f"{high_count} {pos} players projected above {limit} points")
    
    # Check floor/ceiling relationship
    if all(col in df.columns for col in ['floor', 'ceiling', 'proj_mean']):
        floor_ceiling_issues = (df['floor'] > df['ceiling']).sum()
        if floor_ceiling_issues > 0:
            issues.append(f"{floor_ceiling_issues} players have floor > ceiling")
        
        proj_outside_range = (
            (df['proj_mean'] < df['floor']) | 
            (df['proj_mean'] > df['ceiling'])
        ).sum()
        if proj_outside_range > 0:
            issues.append(f"{proj_outside_range} players have projection outside floor-ceiling range")
    
    return issues