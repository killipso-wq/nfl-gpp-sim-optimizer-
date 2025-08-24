"""Diagnostics and validation for simulation projections."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def calculate_projection_diagnostics(sim_df: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate diagnostic metrics comparing sim projections to site projections.
    
    Args:
        sim_df: DataFrame with simulation projections
        site_df: DataFrame with site projections
        
    Returns:
        DataFrame with diagnostic metrics by position
    """
    # Merge simulation and site data
    merged = sim_df.merge(
        site_df[['player_id', 'site_proj']].dropna(),
        on='player_id',
        how='inner'
    )
    
    if len(merged) == 0:
        print("Warning: No matching players found between sim and site data")
        return pd.DataFrame()
    
    diagnostics = []
    
    # Overall diagnostics
    overall_stats = calculate_error_metrics(merged['proj_mean'], merged['site_proj'])
    overall_stats.update({
        'position': 'ALL',
        'player_count': len(merged),
        'coverage_p10_p90': calculate_coverage(merged, 'site_proj', 'p10', 'p90'),
        'avg_sim_proj': merged['proj_mean'].mean(),
        'avg_site_proj': merged['site_proj'].mean(),
    })
    diagnostics.append(overall_stats)
    
    # By position
    for position in merged['position'].unique():
        pos_data = merged[merged['position'] == position]
        
        if len(pos_data) < 3:  # Skip positions with too few players
            continue
            
        pos_stats = calculate_error_metrics(pos_data['proj_mean'], pos_data['site_proj'])
        pos_stats.update({
            'position': position,
            'player_count': len(pos_data),
            'coverage_p10_p90': calculate_coverage(pos_data, 'site_proj', 'p10', 'p90'),
            'avg_sim_proj': pos_data['proj_mean'].mean(),
            'avg_site_proj': pos_data['site_proj'].mean(),
        })
        
        diagnostics.append(pos_stats)
    
    return pd.DataFrame(diagnostics)


def calculate_error_metrics(sim_proj: pd.Series, site_proj: pd.Series) -> Dict[str, float]:
    """Calculate error metrics between two projection sets.
    
    Args:
        sim_proj: Simulation projections
        site_proj: Site projections
        
    Returns:
        Dictionary with error metrics
    """
    # Remove any NaN values
    mask = ~(sim_proj.isna() | site_proj.isna())
    sim_clean = sim_proj[mask]
    site_clean = site_proj[mask]
    
    if len(sim_clean) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'correlation': np.nan,
            'mean_bias': np.nan,
            'median_bias': np.nan,
        }
    
    # Error calculations
    errors = sim_clean - site_clean
    
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    correlation = sim_clean.corr(site_clean)
    mean_bias = errors.mean()
    median_bias = errors.median()
    
    return {
        'mae': round(mae, 3),
        'rmse': round(rmse, 3), 
        'correlation': round(correlation, 3),
        'mean_bias': round(mean_bias, 3),
        'median_bias': round(median_bias, 3),
    }


def calculate_coverage(df: pd.DataFrame, actual_col: str, lower_col: str, upper_col: str) -> float:
    """Calculate coverage rate (what % of actuals fall within prediction interval).
    
    Args:
        df: DataFrame with projections and actuals
        actual_col: Column name for actual values
        lower_col: Column name for lower bound (e.g., p10)
        upper_col: Column name for upper bound (e.g., p90)
        
    Returns:
        Coverage rate (0-1)
    """
    mask = ~(df[actual_col].isna() | df[lower_col].isna() | df[upper_col].isna())
    
    if mask.sum() == 0:
        return np.nan
    
    within_bounds = (
        (df.loc[mask, actual_col] >= df.loc[mask, lower_col]) &
        (df.loc[mask, actual_col] <= df.loc[mask, upper_col])
    )
    
    return within_bounds.mean()


def identify_projection_flags(sim_df: pd.DataFrame, site_df: pd.DataFrame, 
                            abs_delta_threshold: float = 5.0, 
                            pct_delta_threshold: float = 25.0) -> pd.DataFrame:
    """Identify players with large projection differences for manual review.
    
    Args:
        sim_df: Simulation projections
        site_df: Site projections
        abs_delta_threshold: Absolute point difference threshold
        pct_delta_threshold: Percentage difference threshold
        
    Returns:
        DataFrame with flagged players
    """
    # Merge data
    merged = sim_df.merge(
        site_df[['player_id', 'site_proj', 'name']].dropna(),
        on='player_id',
        how='inner'
    )
    
    if len(merged) == 0:
        return pd.DataFrame()
    
    # Calculate deltas
    merged['abs_delta'] = merged['proj_mean'] - merged['site_proj']
    merged['pct_delta'] = (merged['abs_delta'] / merged['site_proj']) * 100
    
    # Flag criteria
    flags = []
    
    # Large absolute differences
    large_abs = merged[np.abs(merged['abs_delta']) >= abs_delta_threshold].copy()
    if len(large_abs) > 0:
        large_abs['flag_reason'] = 'Large absolute difference (±' + str(abs_delta_threshold) + ' pts)'
        flags.append(large_abs)
    
    # Large percentage differences  
    large_pct = merged[np.abs(merged['pct_delta']) >= pct_delta_threshold].copy()
    if len(large_pct) > 0:
        large_pct['flag_reason'] = 'Large percentage difference (±' + str(pct_delta_threshold) + '%)'
        flags.append(large_pct)
    
    # Very high projections (potential outliers)
    high_proj = merged[merged['proj_mean'] >= 30].copy()
    if len(high_proj) > 0:
        high_proj['flag_reason'] = 'Very high projection (≥30 pts)'
        flags.append(high_proj)
    
    # Very low projections for skill positions
    low_proj = merged[
        (merged['position'].isin(['QB', 'RB', 'WR', 'TE'])) &
        (merged['proj_mean'] <= 5)
    ].copy()
    if len(low_proj) > 0:
        low_proj['flag_reason'] = 'Very low projection for skill position (≤5 pts)'
        flags.append(low_proj)
    
    if not flags:
        return pd.DataFrame()
    
    # Combine all flags
    all_flags = pd.concat(flags, ignore_index=True)
    
    # Remove duplicates (keep first flag reason)
    all_flags = all_flags.drop_duplicates(subset=['player_id'], keep='first')
    
    # Select relevant columns
    flag_cols = [
        'player_id', 'name', 'position', 'team', 'salary', 
        'proj_mean', 'site_proj', 'abs_delta', 'pct_delta', 'flag_reason'
    ]
    
    available_cols = [col for col in flag_cols if col in all_flags.columns]
    flags_df = all_flags[available_cols].copy()
    
    # Sort by absolute delta descending
    flags_df = flags_df.sort_values('abs_delta', key=abs, ascending=False)
    
    return flags_df


def create_diagnostics_summary(diagnostics_df: pd.DataFrame) -> Dict[str, any]:
    """Create summary of diagnostic results.
    
    Args:
        diagnostics_df: DataFrame with diagnostic metrics by position
        
    Returns:
        Dictionary with summary statistics
    """
    if len(diagnostics_df) == 0:
        return {}
    
    overall_row = diagnostics_df[diagnostics_df['position'] == 'ALL']
    
    if len(overall_row) == 0:
        return {}
    
    overall = overall_row.iloc[0]
    
    # Position-specific summaries
    pos_data = diagnostics_df[diagnostics_df['position'] != 'ALL']
    
    summary = {
        'overall_metrics': {
            'players_compared': int(overall['player_count']),
            'mae': overall['mae'],
            'rmse': overall['rmse'],
            'correlation': overall['correlation'],
            'mean_bias': overall['mean_bias'],
            'coverage_p10_p90': overall['coverage_p10_p90'],
        },
        'by_position': {
            'position_count': len(pos_data),
            'best_correlation_pos': pos_data.loc[pos_data['correlation'].idxmax(), 'position'] if len(pos_data) > 0 else None,
            'worst_correlation_pos': pos_data.loc[pos_data['correlation'].idxmin(), 'position'] if len(pos_data) > 0 else None,
            'lowest_mae_pos': pos_data.loc[pos_data['mae'].idxmin(), 'position'] if len(pos_data) > 0 else None,
            'highest_mae_pos': pos_data.loc[pos_data['mae'].idxmax(), 'position'] if len(pos_data) > 0 else None,
        },
    }
    
    # Quality assessment
    if overall['correlation'] >= 0.7:
        quality = 'Excellent'
    elif overall['correlation'] >= 0.5:
        quality = 'Good'  
    elif overall['correlation'] >= 0.3:
        quality = 'Fair'
    else:
        quality = 'Poor'
    
    summary['overall_metrics']['quality_assessment'] = quality
    
    return summary


def validate_simulation_results(sim_df: pd.DataFrame) -> List[str]:
    """Validate simulation results for common issues.
    
    Args:
        sim_df: Simulation results DataFrame
        
    Returns:
        List of validation warnings/errors
    """
    issues = []
    
    # Check for missing critical columns
    required_cols = ['player_id', 'name', 'position', 'salary', 'proj_mean']
    missing_cols = [col for col in required_cols if col not in sim_df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for NaN projections
    if sim_df['proj_mean'].isna().any():
        nan_count = sim_df['proj_mean'].isna().sum()
        issues.append(f"{nan_count} players have NaN projections")
    
    # Check for negative projections
    if (sim_df['proj_mean'] < 0).any():
        neg_count = (sim_df['proj_mean'] < 0).sum()
        issues.append(f"{neg_count} players have negative projections")
    
    # Check for unrealistic high projections
    high_proj = sim_df[sim_df['proj_mean'] > 50]
    if len(high_proj) > 0:
        issues.append(f"{len(high_proj)} players have projections >50 points (check: {high_proj['name'].tolist()})")
    
    # Check quantile ordering
    quantile_cols = ['p10', 'p25', 'p50', 'p75', 'p90', 'p95']
    available_quantiles = [col for col in quantile_cols if col in sim_df.columns]
    
    if len(available_quantiles) >= 2:
        for i in range(len(available_quantiles) - 1):
            col1, col2 = available_quantiles[i], available_quantiles[i+1]
            violations = (sim_df[col1] > sim_df[col2]).sum()
            if violations > 0:
                issues.append(f"{violations} players have {col1} > {col2} (quantile ordering violation)")
    
    # Check for duplicate players
    if sim_df['player_id'].duplicated().any():
        dup_count = sim_df['player_id'].duplicated().sum()
        issues.append(f"{dup_count} duplicate player IDs found")
    
    # Check salary ranges
    if 'salary' in sim_df.columns:
        min_sal = sim_df['salary'].min()
        max_sal = sim_df['salary'].max()
        
        if min_sal < 3000:
            issues.append(f"Unusually low minimum salary: ${min_sal}")
        if max_sal > 15000:
            issues.append(f"Unusually high maximum salary: ${max_sal}")
    
    # Position distribution check
    pos_counts = sim_df['position'].value_counts()
    
    if 'QB' in pos_counts and pos_counts['QB'] > 50:
        issues.append(f"Unusually high QB count: {pos_counts['QB']}")
    
    if len(issues) == 0:
        issues.append("No validation issues found")
    
    return issues