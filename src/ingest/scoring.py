"""
DraftKings scoring system and percentile calculations.

This module handles fantasy point calculations and provides utilities
for determining boom thresholds based on historical distributions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_dk_points(stats: Dict[str, float], position: str) -> float:
    """
    Calculate DraftKings fantasy points for a player's stat line.
    
    Args:
        stats: Dictionary of player stats (passing_yards, rushing_yards, etc.)
        position: Player position (QB, RB, WR, TE, K, DST)
        
    Returns:
        Total DraftKings fantasy points
    """
    points = 0.0
    
    # Passing stats (mainly QB)
    points += stats.get('passing_yards', 0) * 0.04  # 1 pt per 25 yards
    points += stats.get('passing_tds', 0) * 4
    points += stats.get('interceptions', 0) * -1
    points += stats.get('passing_2pt', 0) * 2
    
    # Rushing stats (QB, RB, WR, TE)  
    points += stats.get('rushing_yards', 0) * 0.1  # 1 pt per 10 yards
    points += stats.get('rushing_tds', 0) * 6
    points += stats.get('rushing_2pt', 0) * 2
    
    # Receiving stats (QB, RB, WR, TE)
    points += stats.get('receptions', 0) * 1  # Full PPR
    points += stats.get('receiving_yards', 0) * 0.1  # 1 pt per 10 yards  
    points += stats.get('receiving_tds', 0) * 6
    points += stats.get('receiving_2pt', 0) * 2
    
    # Fumbles (all offensive positions)
    points += stats.get('fumbles_lost', 0) * -1
    
    # Kicker stats
    if position == 'K':
        points += stats.get('fg_made', 0) * 3
        points += stats.get('fg_missed', 0) * -1  
        points += stats.get('pat_made', 0) * 1
        points += stats.get('pat_missed', 0) * -1
        
        # Bonus for long FGs (50+)
        points += stats.get('fg_made_50_plus', 0) * 2
        
    # DST stats  
    if position == 'DST':
        points += stats.get('dst_points_allowed', 0) * get_dst_points_allowed_multiplier(stats.get('dst_points_allowed', 0))
        points += stats.get('dst_sacks', 0) * 1
        points += stats.get('dst_interceptions', 0) * 2
        points += stats.get('dst_fumbles_recovered', 0) * 2
        points += stats.get('dst_safeties', 0) * 2
        points += stats.get('dst_tds', 0) * 6
        points += stats.get('dst_blocked_kicks', 0) * 2
        points += stats.get('dst_return_tds', 0) * 6
        
        # Yards allowed bonus/penalty
        yards_allowed = stats.get('dst_yards_allowed', 0)
        if yards_allowed < 100:
            points += 5
        elif yards_allowed < 200:
            points += 3  
        elif yards_allowed < 300:
            points += 2
        elif yards_allowed < 350:
            points += 0
        elif yards_allowed < 400:
            points += -1
        elif yards_allowed < 450:
            points += -3
        else:
            points += -5
            
    return round(points, 2)


def get_dst_points_allowed_multiplier(points_allowed: float) -> float:
    """
    Get DST fantasy point multiplier based on points allowed.
    
    Args:
        points_allowed: Points allowed by defense
        
    Returns:
        Fantasy point multiplier
    """
    if points_allowed == 0:
        return 5
    elif points_allowed <= 6:
        return 4
    elif points_allowed <= 13:
        return 3
    elif points_allowed <= 17:
        return 1
    elif points_allowed <= 21:
        return 0
    elif points_allowed <= 27:
        return -1
    elif points_allowed <= 34:
        return -3
    else:
        return -5


def calculate_boom_thresholds(weekly_points: pd.DataFrame, quantile: float = 0.9) -> Dict[str, float]:
    """
    Calculate boom thresholds by position from historical weekly fantasy points.
    
    Args:
        weekly_points: DataFrame with columns [player_name, position, week, fantasy_points]  
        quantile: Quantile to use for boom threshold (default 0.9 = 90th percentile)
        
    Returns:
        Dictionary mapping position to boom threshold
    """
    thresholds = {}
    
    for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
        pos_points = weekly_points[weekly_points['position'] == pos]['fantasy_points']
        if len(pos_points) > 0:
            threshold = pos_points.quantile(quantile)
            thresholds[pos] = round(threshold, 1)
        else:
            # Default fallbacks if no data
            defaults = {'QB': 25.0, 'RB': 20.0, 'WR': 18.0, 'TE': 15.0, 'K': 12.0, 'DST': 12.0}
            thresholds[pos] = defaults.get(pos, 15.0)
    
    return thresholds


def synthesize_percentiles_from_mean_std(mean: float, std: float, 
                                       percentiles: list = [10, 25, 50, 75, 90, 95]) -> Dict[str, float]:
    """
    Synthesize percentiles from mean and standard deviation assuming normal distribution.
    
    Args:
        mean: Mean value
        std: Standard deviation
        percentiles: List of percentiles to calculate
        
    Returns:
        Dictionary mapping percentile names to values
    """
    from scipy import stats
    
    result = {}
    for p in percentiles:
        z_score = stats.norm.ppf(p / 100.0)
        value = mean + (z_score * std)
        result[f'p{p}'] = round(max(0, value), 2)  # Fantasy points can't be negative
    
    result['mean'] = round(mean, 2)
    
    return result


def get_position_salary_tiers(position: str) -> Dict[str, tuple]:
    """
    Get salary tiers for position-based analysis.
    
    Args:
        position: Player position
        
    Returns:
        Dictionary mapping tier names to (min_salary, max_salary) tuples
    """
    # Typical DraftKings salary ranges by position
    tiers = {
        'QB': {
            'elite': (8000, 10000),
            'mid': (6500, 7900), 
            'value': (4800, 6400)
        },
        'RB': {
            'elite': (8000, 10000),
            'mid': (6000, 7900),
            'value': (4500, 5900)
        },
        'WR': {
            'elite': (8000, 10000),
            'mid': (6000, 7900),
            'value': (4500, 5900) 
        },
        'TE': {
            'elite': (6500, 9000),
            'mid': (4800, 6400),
            'value': (3500, 4700)
        },
        'K': {
            'elite': (5500, 6000),
            'mid': (4800, 5400),
            'value': (4200, 4700)
        },
        'DST': {
            'elite': (3500, 4500),
            'mid': (2800, 3400),
            'value': (2200, 2700)
        }
    }
    
    return tiers.get(position, {})


def calculate_value_metrics(projected_points: float, salary: int) -> Dict[str, float]:
    """
    Calculate value metrics for a player.
    
    Args:
        projected_points: Projected fantasy points
        salary: DraftKings salary
        
    Returns:
        Dictionary of value metrics
    """
    value_per_1k = (projected_points / salary) * 1000 if salary > 0 else 0
    
    return {
        'value_per_1k': round(value_per_1k, 3),
        'salary_efficiency': round(projected_points / max(salary / 1000, 1), 2)
    }