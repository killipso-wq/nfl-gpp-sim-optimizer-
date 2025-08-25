"""
DraftKings scoring functions for NFL fantasy football.
"""
from typing import Dict, Any
import pandas as pd


def calculate_dk_points(stats: Dict[str, float], position: str) -> float:
    """
    Calculate DraftKings fantasy points from game statistics.
    
    Args:
        stats: Dictionary with game statistics
        position: Player position (QB, RB, WR, TE, DST)
        
    Returns:
        DraftKings fantasy points
    """
    points = 0.0
    
    if position == 'QB':
        points += calculate_qb_points(stats)
    elif position in ['RB', 'WR', 'TE']:
        points += calculate_skill_points(stats)
    elif position == 'DST':
        points += calculate_dst_points(stats)
    
    return round(points, 2)


def calculate_qb_points(stats: Dict[str, float]) -> float:
    """Calculate QB fantasy points."""
    points = 0.0
    
    # Passing
    points += stats.get('passing_yards', 0) * 0.04  # 1 pt per 25 yards
    points += stats.get('passing_tds', 0) * 4       # 4 pts per TD
    points -= stats.get('interceptions', 0) * 1     # -1 pt per INT
    
    # Rushing
    points += stats.get('rushing_yards', 0) * 0.1   # 1 pt per 10 yards
    points += stats.get('rushing_tds', 0) * 6       # 6 pts per TD
    
    # Bonuses
    if stats.get('passing_yards', 0) >= 300:
        points += 3  # 300+ passing yard bonus
    
    # Fumbles lost
    points -= stats.get('fumbles_lost', 0) * 1      # -1 pt per fumble lost
    
    return points


def calculate_skill_points(stats: Dict[str, float]) -> float:
    """Calculate RB/WR/TE fantasy points."""
    points = 0.0
    
    # Rushing
    points += stats.get('rushing_yards', 0) * 0.1   # 1 pt per 10 yards
    points += stats.get('rushing_tds', 0) * 6       # 6 pts per TD
    
    # Receiving
    points += stats.get('receptions', 0) * 1        # 1 pt per reception
    points += stats.get('receiving_yards', 0) * 0.1 # 1 pt per 10 yards
    points += stats.get('receiving_tds', 0) * 6     # 6 pts per TD
    
    # Bonuses
    total_yards = stats.get('rushing_yards', 0) + stats.get('receiving_yards', 0)
    if total_yards >= 100:
        points += 3  # 100+ yard bonus
    
    # Fumbles lost
    points -= stats.get('fumbles_lost', 0) * 1      # -1 pt per fumble lost
    
    return points


def calculate_dst_points(stats: Dict[str, float]) -> float:
    """Calculate Defense/Special Teams fantasy points."""
    points = 0.0
    
    # Points allowed tiers
    points_allowed = stats.get('points_allowed', 0)
    if points_allowed == 0:
        points += 10
    elif points_allowed <= 6:
        points += 7
    elif points_allowed <= 13:
        points += 4
    elif points_allowed <= 20:
        points += 1
    elif points_allowed <= 27:
        points += 0
    elif points_allowed <= 34:
        points -= 1
    else:
        points -= 4
    
    # Defensive stats
    points += stats.get('sacks', 0) * 1           # 1 pt per sack
    points += stats.get('interceptions', 0) * 2   # 2 pts per INT
    points += stats.get('fumble_recoveries', 0) * 2  # 2 pts per fumble recovery
    points += stats.get('safeties', 0) * 2        # 2 pts per safety
    points += stats.get('def_tds', 0) * 6         # 6 pts per defensive TD
    points += stats.get('special_teams_tds', 0) * 6  # 6 pts per special teams TD
    
    # Blocked kicks
    points += stats.get('blocked_kicks', 0) * 2   # 2 pts per blocked kick
    
    return points


def apply_dk_bonuses_to_samples(samples: pd.Series, position: str) -> pd.Series:
    """
    Apply approximate DraftKings bonuses to simulation samples.
    
    This is a simplified approximation for Monte Carlo simulation.
    
    Args:
        samples: Array of simulated fantasy points
        position: Player position
        
    Returns:
        Samples with bonuses applied
    """
    if position == 'QB':
        # Approximate 300+ passing yard bonus
        # Assume ~18 fantasy points correlates to 300 passing yards
        bonus_mask = samples >= 18
        samples = samples + (bonus_mask * 3)
    
    elif position in ['RB', 'WR', 'TE']:
        # Approximate 100+ total yard bonus
        # Assume ~12 fantasy points correlates to 100 yards
        bonus_mask = samples >= 12
        samples = samples + (bonus_mask * 3)
    
    # DST bonuses are more complex and position-specific
    # For MVP, we'll skip detailed DST bonus approximation
    
    return samples


def get_dk_scoring_config() -> Dict[str, Dict[str, float]]:
    """
    Get DraftKings scoring configuration.
    
    Returns:
        Dictionary with scoring rules by position
    """
    return {
        'QB': {
            'passing_yards': 0.04,
            'passing_tds': 4,
            'interceptions': -1,
            'rushing_yards': 0.1,
            'rushing_tds': 6,
            'fumbles_lost': -1,
            'passing_yards_bonus_threshold': 300,
            'passing_yards_bonus': 3
        },
        'skill': {  # RB, WR, TE
            'rushing_yards': 0.1,
            'rushing_tds': 6,
            'receptions': 1,
            'receiving_yards': 0.1,
            'receiving_tds': 6,
            'fumbles_lost': -1,
            'total_yards_bonus_threshold': 100,
            'total_yards_bonus': 3
        },
        'DST': {
            'points_allowed_tiers': {
                0: 10, 6: 7, 13: 4, 20: 1, 27: 0, 34: -1, 35: -4
            },
            'sacks': 1,
            'interceptions': 2,
            'fumble_recoveries': 2,
            'safeties': 2,
            'def_tds': 6,
            'special_teams_tds': 6,
            'blocked_kicks': 2
        }
    }


def estimate_dk_points_from_stats(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate DraftKings points from weekly statistics DataFrame.
    
    Args:
        weekly_stats: DataFrame with weekly player statistics
        
    Returns:
        DataFrame with added 'fantasy_points_dk' column
    """
    result = weekly_stats.copy()
    result['fantasy_points_dk'] = 0.0
    
    for idx, row in result.iterrows():
        position = row.get('position', '')
        stats = row.to_dict()
        dk_points = calculate_dk_points(stats, position)
        result.at[idx, 'fantasy_points_dk'] = dk_points
    
    return result