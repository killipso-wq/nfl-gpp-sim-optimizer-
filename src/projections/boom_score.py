"""Boom probability and boom score calculation."""

import pandas as pd
import numpy as np
import json
from typing import Dict, Optional


def calculate_boom_metrics(sim_df: pd.DataFrame, boom_thresholds: Dict[str, float], 
                          site_df: pd.DataFrame = None) -> pd.DataFrame:
    """Calculate boom probability and boom score for players.
    
    Args:
        sim_df: DataFrame with simulation results 
        boom_thresholds: Dictionary of position -> boom threshold points
        site_df: DataFrame with site data (optional, for ownership and site projections)
        
    Returns:
        DataFrame with boom metrics added
    """
    df = sim_df.copy()
    
    # Calculate boom probability for each player
    df['boom_prob'] = df.apply(
        lambda row: calculate_boom_probability(row, boom_thresholds, site_df), 
        axis=1
    )
    
    # Add ownership data if available
    if site_df is not None and 'ownership' in site_df.columns:
        ownership_cols = ['player_id', 'ownership']
        if 'site_proj' in site_df.columns:
            ownership_cols.append('site_proj')
        
        df = df.merge(site_df[ownership_cols], on='player_id', how='left')
    
    # Calculate boom score (1-100 scale)
    df['boom_score'] = df.apply(calculate_boom_score, axis=1)
    
    # Dart flag (low ownership + high boom score)
    df['dart_flag'] = (
        (df.get('ownership', 0.1) <= 0.05) &  # 5% or lower ownership
        (df['boom_score'] >= 70)  # High boom score
    )
    
    return df


def calculate_boom_probability(player_row: pd.Series, boom_thresholds: Dict[str, float], 
                             site_df: pd.DataFrame = None) -> float:
    """Calculate boom probability for a single player.
    
    Boom occurs if sim_points >= max(positional_threshold, 1.2*site_proj, site_proj+5)
    
    Args:
        player_row: Player data row with simulation results
        boom_thresholds: Position boom thresholds
        site_df: Site data for projection comparison
        
    Returns:
        Boom probability (0-1)
    """
    position = player_row.get('position', '')
    
    # Get position boom threshold
    pos_threshold = boom_thresholds.get(position, 15.0)  # Default 15 points
    
    # Start with positional threshold
    boom_threshold = pos_threshold
    
    # If we have site projection, use the higher of:
    # - Position threshold
    # - 1.2x site projection  
    # - Site projection + 5
    if 'site_proj' in player_row and pd.notna(player_row['site_proj']):
        site_proj = player_row['site_proj']
        
        site_boom_1 = site_proj * 1.2
        site_boom_2 = site_proj + 5
        
        boom_threshold = max(boom_threshold, site_boom_1, site_boom_2)
    
    # Estimate boom probability using normal approximation
    # Based on simulation mean and standard deviation
    proj_mean = player_row.get('proj_mean', 0)
    proj_std = player_row.get('std', 5)  # Default std if missing
    
    if proj_std <= 0:
        proj_std = proj_mean * 0.3  # Rough estimate: 30% coefficient of variation
    
    # Z-score for boom threshold
    if proj_std > 0:
        z_score = (boom_threshold - proj_mean) / proj_std
        
        # Probability of exceeding boom threshold (using normal approximation)
        from scipy.stats import norm
        boom_prob = 1 - norm.cdf(z_score)
        
        # Constrain to reasonable range
        boom_prob = np.clip(boom_prob, 0.01, 0.99)
    else:
        # If no variance, probability is binary
        boom_prob = 1.0 if proj_mean >= boom_threshold else 0.01
    
    return boom_prob


def calculate_boom_score(player_row: pd.Series) -> float:
    """Calculate boom score on 1-100 scale.
    
    Boom score combines:
    - Boom probability (base score)
    - Value per $1k (bonus for positive value)
    - Low ownership boost (bigger boost for lower ownership)
    - Beat site probability bonus
    
    Args:
        player_row: Player data with boom_prob, value metrics, ownership
        
    Returns:
        Boom score (1-100)
    """
    # Base score from boom probability (0-60 points)
    boom_prob = player_row.get('boom_prob', 0)
    base_score = boom_prob * 60
    
    # Value bonus (0-20 points)
    value_per_1k = player_row.get('value_per_1k', 0)
    if value_per_1k > 0:
        # Scaled value bonus: 3.0+ value = full 20 points
        value_bonus = min(value_per_1k / 3.0 * 20, 20)
    else:
        value_bonus = 0
    
    # Low ownership boost (0-15 points)
    ownership = player_row.get('ownership', 0.1)  # Default 10% if missing
    
    if ownership <= 0.01:  # <= 1%
        ownership_boost = 15
    elif ownership <= 0.03:  # <= 3%
        ownership_boost = 12
    elif ownership <= 0.05:  # <= 5%
        ownership_boost = 8
    elif ownership <= 0.10:  # <= 10%
        ownership_boost = 4
    else:
        ownership_boost = 0
    
    # Beat site bonus (0-5 points)
    beat_site_prob = player_row.get('beat_site_prob', 0.5)  # Default 50%
    beat_site_bonus = max((beat_site_prob - 0.5) * 10, 0)  # Bonus for > 50% chance
    beat_site_bonus = min(beat_site_bonus, 5)  # Cap at 5 points
    
    # Combine scores
    boom_score = base_score + value_bonus + ownership_boost + beat_site_bonus
    
    # Scale to 1-100 and round
    boom_score = np.clip(boom_score, 1, 100)
    
    return round(boom_score, 1)


def load_boom_thresholds(filepath: str) -> Dict[str, float]:
    """Load boom thresholds from JSON file.
    
    Args:
        filepath: Path to boom thresholds JSON file
        
    Returns:
        Dictionary of position -> threshold
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return data.get('thresholds', {})
    
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load boom thresholds from {filepath}: {e}")
        
        # Return default thresholds
        return {
            'QB': 25.0,
            'RB': 20.0, 
            'WR': 18.0,
            'TE': 15.0,
            'DST': 10.0,
        }


def create_boom_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary of boom metrics by position.
    
    Args:
        df: DataFrame with boom metrics
        
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
            'avg_boom_prob': pos_data['boom_prob'].mean(),
            'avg_boom_score': pos_data['boom_score'].mean(),
            'high_boom_count': (pos_data['boom_prob'] >= 0.2).sum(),  # >= 20% boom prob
            'dart_flag_count': pos_data['dart_flag'].sum() if 'dart_flag' in pos_data.columns else 0,
            'top_boom_player': pos_data.loc[pos_data['boom_score'].idxmax(), 'name'] if len(pos_data) > 0 else '',
            'max_boom_score': pos_data['boom_score'].max(),
        }
        
        # Add ownership stats if available
        if 'ownership' in pos_data.columns:
            summary.update({
                'avg_ownership': pos_data['ownership'].mean() * 100,  # Convert to percentage
                'low_owned_count': (pos_data['ownership'] <= 0.05).sum(),  # <= 5% owned
            })
        
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data)


def identify_boom_candidates(df: pd.DataFrame, min_boom_prob: float = 0.15,
                           min_boom_score: float = 60, max_ownership: float = 0.15) -> pd.DataFrame:
    """Identify top boom candidates based on criteria.
    
    Args:
        df: DataFrame with boom metrics
        min_boom_prob: Minimum boom probability threshold
        min_boom_score: Minimum boom score threshold  
        max_ownership: Maximum ownership threshold
        
    Returns:
        DataFrame filtered to boom candidates, sorted by boom score
    """
    # Apply filters
    candidates = df[
        (df['boom_prob'] >= min_boom_prob) &
        (df['boom_score'] >= min_boom_score)
    ].copy()
    
    # Add ownership filter if data is available
    if 'ownership' in candidates.columns:
        candidates = candidates[candidates['ownership'] <= max_ownership]
    
    # Sort by boom score descending
    candidates = candidates.sort_values('boom_score', ascending=False)
    
    # Add ranking
    candidates['boom_rank'] = range(1, len(candidates) + 1)
    
    return candidates


def calculate_position_boom_rates(df: pd.DataFrame, boom_thresholds: Dict[str, float]) -> pd.DataFrame:
    """Calculate actual boom rates by position for validation.
    
    This would typically be used on historical data to validate thresholds.
    
    Args:
        df: DataFrame with actual fantasy points by position
        boom_thresholds: Position boom thresholds to test
        
    Returns:
        DataFrame with boom rates by position
    """
    boom_rates = []
    
    for position in df['position'].unique():
        pos_data = df[df['position'] == position]
        threshold = boom_thresholds.get(position, 15.0)
        
        if len(pos_data) > 0:
            boom_count = (pos_data['fantasy_points'] >= threshold).sum()
            boom_rate = boom_count / len(pos_data)
            
            boom_rates.append({
                'position': position,
                'threshold': threshold,
                'games': len(pos_data),
                'boom_count': boom_count,
                'boom_rate': boom_rate,
                'avg_fantasy_points': pos_data['fantasy_points'].mean(),
                'p90_fantasy_points': pos_data['fantasy_points'].quantile(0.9),
            })
    
    return pd.DataFrame(boom_rates)