"""
Boom score calculation for fantasy football projections.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_boom_score(
    boom_prob: float,
    beat_site_prob: Optional[float] = None,
    ownership: Optional[float] = None,
    value_per_1k: Optional[float] = None,
    position: str = ''
) -> float:
    """
    Calculate boom score (1-100) using composite of boom probability and other factors.
    
    Args:
        boom_prob: Probability of boom performance (0-1)
        beat_site_prob: Probability of beating site projection (0-1)
        ownership: Projected ownership percentage (0-100)
        value_per_1k: Value per $1K salary
        position: Player position
        
    Returns:
        Boom score from 1-100
    """
    # Base score from boom probability (0-60 points)
    base_score = boom_prob * 60
    
    # Beat site bonus (0-20 points)
    beat_site_bonus = 0
    if beat_site_prob is not None:
        beat_site_bonus = beat_site_prob * 20
    
    # Ownership discount (0 to -15 points)
    ownership_discount = 0
    if ownership is not None and ownership > 0:
        # Higher ownership = lower boom score
        ownership_factor = min(ownership / 100, 1.0)  # Cap at 100%
        ownership_discount = -15 * ownership_factor
    
    # Value bonus (0-15 points)
    value_bonus = 0
    if value_per_1k is not None:
        # Position-specific value thresholds
        value_thresholds = get_value_thresholds_by_position()
        threshold = value_thresholds.get(position, {}).get('good', 3.5)
        
        if value_per_1k >= threshold:
            value_bonus = min(15, (value_per_1k - threshold) * 5)
    
    # Combine components
    total_score = base_score + beat_site_bonus + ownership_discount + value_bonus
    
    # Clamp to 1-100 range
    return max(1, min(100, round(total_score)))


def calculate_dart_flag(
    boom_score: float,
    ownership: Optional[float] = None,
    boom_prob: float = 0
) -> bool:
    """
    Determine if player should be flagged as a "dart" play.
    
    Args:
        boom_score: Player's boom score
        ownership: Projected ownership percentage
        boom_prob: Boom probability
        
    Returns:
        True if player is a dart play
    """
    # High boom score with low ownership
    if boom_score >= 75 and ownership is not None and ownership <= 8:
        return True
    
    # Very high boom probability regardless of ownership
    if boom_prob >= 0.25:
        return True
    
    # Moderate boom score with very low ownership
    if boom_score >= 60 and ownership is not None and ownership <= 5:
        return True
    
    return False


def add_boom_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boom scores and dart flags to simulation results DataFrame.
    
    Args:
        df: DataFrame with simulation results
        
    Returns:
        DataFrame with boom_score and dart_flag columns
    """
    result = df.copy()
    
    # Calculate boom scores
    boom_scores = []
    dart_flags = []
    
    for idx, row in result.iterrows():
        boom_score = calculate_boom_score(
            boom_prob=row.get('boom_prob', 0),
            beat_site_prob=row.get('beat_site_prob'),
            ownership=row.get('RST%'),
            value_per_1k=row.get('value_per_1k'),
            position=row.get('POS', '')
        )
        
        dart_flag = calculate_dart_flag(
            boom_score=boom_score,
            ownership=row.get('RST%'),
            boom_prob=row.get('boom_prob', 0)
        )
        
        boom_scores.append(boom_score)
        dart_flags.append(dart_flag)
    
    result['boom_score'] = boom_scores
    result['dart_flag'] = dart_flags
    
    return result


def get_boom_score_tiers() -> Dict[str, Dict[str, float]]:
    """
    Get boom score tier definitions.
    
    Returns:
        Dictionary with boom score tiers
    """
    return {
        'elite': {'min': 85, 'description': 'Elite boom potential'},
        'excellent': {'min': 75, 'description': 'Excellent boom potential'},
        'good': {'min': 65, 'description': 'Good boom potential'},
        'average': {'min': 50, 'description': 'Average boom potential'},
        'below_average': {'min': 35, 'description': 'Below average boom potential'},
        'poor': {'min': 0, 'description': 'Poor boom potential'}
    }


def categorize_boom_score(boom_score: float) -> str:
    """
    Categorize boom score into tier.
    
    Args:
        boom_score: Boom score (1-100)
        
    Returns:
        Boom score tier name
    """
    tiers = get_boom_score_tiers()
    
    for tier_name, tier_data in tiers.items():
        if boom_score >= tier_data['min']:
            return tier_name
    
    return 'poor'


def get_value_thresholds_by_position() -> Dict[str, Dict[str, float]]:
    """
    Get value thresholds by position for boom score calculation.
    
    Returns:
        Dictionary with value thresholds by position
    """
    return {
        'QB': {'excellent': 4.5, 'good': 3.5, 'average': 3.0},
        'RB': {'excellent': 4.0, 'good': 3.5, 'average': 3.0},
        'WR': {'excellent': 4.0, 'good': 3.5, 'average': 3.0},
        'TE': {'excellent': 4.5, 'good': 3.5, 'average': 3.0},
        'DST': {'excellent': 3.5, 'good': 3.0, 'average': 2.5},
        'K': {'excellent': 3.5, 'good': 3.0, 'average': 2.5}
    }


def calculate_boom_metrics_summary(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate summary metrics for boom analysis.
    
    Args:
        df: DataFrame with boom scores
        
    Returns:
        Dictionary with boom metrics
    """
    if 'boom_score' not in df.columns:
        return {}
    
    boom_scores = df['boom_score'].dropna()
    
    if len(boom_scores) == 0:
        return {}
    
    metrics = {
        'avg_boom_score': float(boom_scores.mean()),
        'median_boom_score': float(boom_scores.median()),
        'boom_score_std': float(boom_scores.std()),
        'elite_players': int((boom_scores >= 85).sum()),
        'excellent_players': int((boom_scores >= 75).sum()),
        'good_players': int((boom_scores >= 65).sum()),
        'dart_players': int(df.get('dart_flag', pd.Series(dtype=bool)).sum())
    }
    
    return metrics


def get_boom_recommendations(df: pd.DataFrame, min_boom_score: float = 70) -> pd.DataFrame:
    """
    Get recommended boom plays based on boom score threshold.
    
    Args:
        df: DataFrame with boom scores
        min_boom_score: Minimum boom score threshold
        
    Returns:
        DataFrame with recommended boom plays
    """
    if 'boom_score' not in df.columns:
        return pd.DataFrame()
    
    recommendations = df[df['boom_score'] >= min_boom_score].copy()
    
    # Sort by boom score descending
    recommendations = recommendations.sort_values('boom_score', ascending=False)
    
    # Add recommendation reasons
    recommendations['recommendation_reason'] = recommendations.apply(
        lambda row: _generate_recommendation_reason(row),
        axis=1
    )
    
    return recommendations


def _generate_recommendation_reason(row: pd.Series) -> str:
    """Generate recommendation reason text."""
    reasons = []
    
    boom_score = row.get('boom_score', 0)
    if boom_score >= 85:
        reasons.append('Elite boom score')
    elif boom_score >= 75:
        reasons.append('Excellent boom score')
    
    if row.get('dart_flag', False):
        reasons.append('Dart play')
    
    ownership = row.get('RST%')
    if ownership is not None and ownership <= 10:
        reasons.append('Low ownership')
    
    value_per_1k = row.get('value_per_1k')
    if value_per_1k is not None and value_per_1k >= 4.0:
        reasons.append('High value')
    
    boom_prob = row.get('boom_prob', 0)
    if boom_prob >= 0.2:
        reasons.append('High boom probability')
    
    return '; '.join(reasons) if reasons else 'Above boom threshold'


def add_boom_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boom score categories to DataFrame.
    
    Args:
        df: DataFrame with boom_score column
        
    Returns:
        DataFrame with boom_category column
    """
    result = df.copy()
    
    if 'boom_score' in result.columns:
        result['boom_category'] = result['boom_score'].apply(categorize_boom_score)
    
    return result