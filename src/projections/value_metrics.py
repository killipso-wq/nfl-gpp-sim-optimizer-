"""Value metrics calculation for fantasy projections."""

import pandas as pd
import numpy as np


def calculate_value_metrics(sim_df: pd.DataFrame, site_df: pd.DataFrame = None) -> pd.DataFrame:
    """Calculate value metrics for simulated projections.
    
    Args:
        sim_df: DataFrame with simulation results (columns: player_id, proj_mean, ceiling, salary, etc.)
        site_df: DataFrame with site data (optional, for comparison projections)
        
    Returns:
        DataFrame with value metrics added
    """
    df = sim_df.copy()
    
    # Basic value per $1k
    df['value_per_1k'] = (df['proj_mean'] / df['salary']) * 1000
    
    # Ceiling value per $1k (using p90)
    df['ceil_per_1k'] = (df['ceiling'] / df['salary']) * 1000
    
    # Floor value per $1k (using p10)
    df['floor_per_1k'] = (df['floor'] / df['salary']) * 1000
    
    # Value rank within position
    for position in df['position'].unique():
        pos_mask = df['position'] == position
        df.loc[pos_mask, 'value_rank'] = df.loc[pos_mask, 'value_per_1k'].rank(ascending=False, method='dense')
        df.loc[pos_mask, 'ceil_value_rank'] = df.loc[pos_mask, 'ceil_per_1k'].rank(ascending=False, method='dense')
    
    # If site data is provided, calculate comparison metrics
    if site_df is not None:
        df = add_site_comparison_metrics(df, site_df)
    
    return df


def add_site_comparison_metrics(sim_df: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:
    """Add metrics comparing simulation projections to site projections.
    
    Args:
        sim_df: DataFrame with simulation results
        site_df: DataFrame with site projections
        
    Returns:
        DataFrame with comparison metrics added
    """
    df = sim_df.copy()
    
    # Merge with site data to get site projections
    site_cols = ['player_id', 'site_proj'] if 'site_proj' in site_df.columns else ['player_id']
    if len(site_cols) > 1:
        df = df.merge(site_df[site_cols], on='player_id', how='left')
        
        # Calculate deltas vs site
        df['delta_vs_site'] = df['proj_mean'] - df['site_proj']
        df['pct_delta_vs_site'] = (df['delta_vs_site'] / df['site_proj']) * 100
        
        # Site value metrics
        df['site_value_per_1k'] = (df['site_proj'] / df['salary']) * 1000
        df['value_diff_vs_site'] = df['value_per_1k'] - df['site_value_per_1k']
        
        # Beat site probability (what % of sims beat site projection)
        # This would need sim-level data, so we'll estimate based on distribution
        df['beat_site_prob'] = estimate_beat_site_probability(df)
    
    return df


def estimate_beat_site_probability(df: pd.DataFrame) -> pd.Series:
    """Estimate probability that simulation beats site projection.
    
    Uses normal approximation based on mean and std from simulations.
    
    Args:
        df: DataFrame with proj_mean, std, and site_proj columns
        
    Returns:
        Series with beat site probabilities
    """
    from scipy.stats import norm
    
    # Handle missing site projections
    mask = df['site_proj'].notna() & (df['std'] > 0)
    
    beat_prob = pd.Series(0.5, index=df.index)  # Default 50% if no comparison possible
    
    if mask.any():
        # Z-score: how many standard deviations is site proj from our mean
        z_scores = (df.loc[mask, 'site_proj'] - df.loc[mask, 'proj_mean']) / df.loc[mask, 'std']
        
        # Probability we exceed site projection
        beat_prob.loc[mask] = 1 - norm.cdf(z_scores)
    
    return beat_prob


def calculate_positional_values(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate position-relative value metrics.
    
    Args:
        df: DataFrame with value metrics
        
    Returns:
        DataFrame with positional value metrics added
    """
    df = df.copy()
    
    for position in df['position'].unique():
        pos_mask = df['position'] == position
        pos_data = df[pos_mask]
        
        if len(pos_data) == 0:
            continue
        
        # Position means for standardization
        pos_value_mean = pos_data['value_per_1k'].mean()
        pos_value_std = pos_data['value_per_1k'].std()
        
        pos_proj_mean = pos_data['proj_mean'].mean() 
        pos_proj_std = pos_data['proj_mean'].std()
        
        # Z-scores (standardized values)
        if pos_value_std > 0:
            df.loc[pos_mask, 'value_zscore'] = (pos_data['value_per_1k'] - pos_value_mean) / pos_value_std
        else:
            df.loc[pos_mask, 'value_zscore'] = 0
            
        if pos_proj_std > 0:
            df.loc[pos_mask, 'proj_zscore'] = (pos_data['proj_mean'] - pos_proj_mean) / pos_proj_std
        else:
            df.loc[pos_mask, 'proj_zscore'] = 0
        
        # Percentile rankings
        df.loc[pos_mask, 'value_percentile'] = pos_data['value_per_1k'].rank(pct=True) * 100
        df.loc[pos_mask, 'proj_percentile'] = pos_data['proj_mean'].rank(pct=True) * 100
    
    return df


def identify_value_plays(df: pd.DataFrame, value_threshold: float = 3.0, 
                        proj_threshold: float = None) -> pd.DataFrame:
    """Identify high-value plays based on value metrics.
    
    Args:
        df: DataFrame with value metrics
        value_threshold: Minimum value per $1k to be considered a value play
        proj_threshold: Minimum projection percentile (optional)
        
    Returns:
        DataFrame with value play flags added
    """
    df = df.copy()
    
    # Value play flags
    df['is_value_play'] = df['value_per_1k'] >= value_threshold
    
    if proj_threshold is not None:
        df['is_value_play'] = df['is_value_play'] & (df['proj_percentile'] >= proj_threshold)
    
    # Premium value plays (top value at each position)
    for position in df['position'].unique():
        pos_mask = df['position'] == position
        pos_data = df[pos_mask]
        
        if len(pos_data) == 0:
            continue
        
        # Top 20% value at position
        value_80th_percentile = pos_data['value_per_1k'].quantile(0.8)
        df.loc[pos_mask, 'is_premium_value'] = pos_data['value_per_1k'] >= value_80th_percentile
    
    return df


def create_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary of value metrics by position.
    
    Args:
        df: DataFrame with value metrics
        
    Returns:
        Summary DataFrame by position
    """
    summary_data = []
    
    for position in df['position'].unique():
        pos_data = df[df['position'] == position]
        
        if len(pos_data) == 0:
            continue
        
        summary = {
            'position': position,
            'player_count': len(pos_data),
            'avg_salary': pos_data['salary'].mean(),
            'avg_projection': pos_data['proj_mean'].mean(),
            'avg_value_per_1k': pos_data['value_per_1k'].mean(),
            'max_value_per_1k': pos_data['value_per_1k'].max(),
            'top_value_player': pos_data.loc[pos_data['value_per_1k'].idxmax(), 'name'],
            'value_plays_count': pos_data['is_value_play'].sum() if 'is_value_play' in pos_data.columns else 0,
        }
        
        # Add site comparison if available
        if 'delta_vs_site' in pos_data.columns:
            summary.update({
                'avg_delta_vs_site': pos_data['delta_vs_site'].mean(),
                'avg_beat_site_prob': pos_data['beat_site_prob'].mean(),
                'positive_delta_count': (pos_data['delta_vs_site'] > 0).sum(),
            })
        
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data)