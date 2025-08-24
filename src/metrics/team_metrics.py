"""Team-level metrics calculation."""

import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_team_metrics(pbp_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive team metrics from play-by-play data.
    
    Args:
        pbp_df: Play-by-play DataFrame
        schedule_df: Schedule DataFrame
        
    Returns:
        DataFrame with team metrics by season
    """
    team_metrics = []
    
    for (team, season), group in pbp_df.groupby(['posteam', 'season']):
        metrics = {
            'team': team,
            'season': season,
        }
        
        # Basic counts
        metrics.update(_calculate_basic_counts(group))
        
        # EPA metrics
        metrics.update(_calculate_epa_metrics(group))
        
        # Success rate metrics
        metrics.update(_calculate_success_metrics(group))
        
        # Pass rate metrics
        metrics.update(_calculate_pass_rate_metrics(group))
        
        # Pace metrics
        metrics.update(_calculate_pace_metrics(group))
        
        # PROE (Pass Rate Over Expected)
        metrics.update(_calculate_proe_metrics(group))
        
        team_metrics.append(metrics)
    
    return pd.DataFrame(team_metrics)


def _calculate_basic_counts(group_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic play counts."""
    return {
        'total_plays': len(group_df),
        'pass_plays': (group_df['play_type'] == 'pass').sum(),
        'run_plays': (group_df['play_type'] == 'run').sum(),
        'games': group_df['game_id'].nunique(),
    }


def _calculate_epa_metrics(group_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate EPA (Expected Points Added) metrics."""
    if 'epa' not in group_df.columns:
        return {
            'epa_per_play': 0,
            'pass_epa_per_play': 0,
            'run_epa_per_play': 0,
        }
    
    # Overall EPA
    epa_per_play = group_df['epa'].mean()
    
    # EPA by play type
    pass_plays = group_df[group_df['play_type'] == 'pass']
    run_plays = group_df[group_df['play_type'] == 'run']
    
    pass_epa = pass_plays['epa'].mean() if len(pass_plays) > 0 else 0
    run_epa = run_plays['epa'].mean() if len(run_plays) > 0 else 0
    
    return {
        'epa_per_play': epa_per_play,
        'pass_epa_per_play': pass_epa,
        'run_epa_per_play': run_epa,
    }


def _calculate_success_metrics(group_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate success rate metrics."""
    if 'success' not in group_df.columns:
        return {
            'success_rate': 0,
            'pass_success_rate': 0,
            'run_success_rate': 0,
        }
    
    # Overall success rate
    success_rate = group_df['success'].mean()
    
    # Success rate by play type
    pass_plays = group_df[group_df['play_type'] == 'pass']
    run_plays = group_df[group_df['play_type'] == 'run']
    
    pass_success = pass_plays['success'].mean() if len(pass_plays) > 0 else 0
    run_success = run_plays['success'].mean() if len(run_plays) > 0 else 0
    
    return {
        'success_rate': success_rate,
        'pass_success_rate': pass_success,
        'run_success_rate': run_success,
    }


def _calculate_pass_rate_metrics(group_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate pass rate metrics."""
    total_plays = len(group_df)
    pass_plays = (group_df['play_type'] == 'pass').sum()
    
    pass_rate = pass_plays / total_plays if total_plays > 0 else 0
    
    # Neutral situation pass rate (if neutral filtering was applied)
    neutral_plays = group_df
    if 'early_down' in group_df.columns:
        neutral_plays = group_df[group_df['early_down'] == 1]
    
    neutral_total = len(neutral_plays)
    neutral_passes = (neutral_plays['play_type'] == 'pass').sum()
    neutral_pass_rate = neutral_passes / neutral_total if neutral_total > 0 else 0
    
    return {
        'pass_rate': pass_rate,
        'neutral_pass_rate': neutral_pass_rate,
    }


def _calculate_pace_metrics(group_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate pace metrics."""
    games = group_df['game_id'].nunique()
    total_plays = len(group_df)
    
    plays_per_game = total_plays / games if games > 0 else 0
    
    return {
        'plays_per_game': plays_per_game,
        'pace_rank': 0,  # Will be filled in later with relative ranking
    }


def _calculate_proe_metrics(group_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate PROE (Pass Rate Over Expected) metrics.
    
    This is a simplified version - in reality, PROE uses expected pass rate
    based on down, distance, score, time, etc.
    """
    # For now, use a simple expected pass rate based on league average
    league_avg_pass_rate = 0.60  # Approximate NFL average
    
    actual_pass_rate = _calculate_pass_rate_metrics(group_df)['pass_rate']
    
    proe = actual_pass_rate - league_avg_pass_rate
    
    # Neutral situation PROE
    neutral_pass_rate = _calculate_pass_rate_metrics(group_df)['neutral_pass_rate']
    neutral_expected = 0.58  # Slightly lower in neutral situations
    
    proe_neutral = neutral_pass_rate - neutral_expected
    
    return {
        'proe': proe,
        'proe_neutral': proe_neutral,
        'neutral_xpass': neutral_expected,  # Expected neutral pass rate
    }


def add_relative_rankings(team_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Add relative rankings within each season."""
    df = team_metrics_df.copy()
    
    # Rank teams within each season for key metrics
    ranking_cols = ['epa_per_play', 'success_rate', 'plays_per_game', 'proe', 'proe_neutral']
    
    for col in ranking_cols:
        if col in df.columns:
            df[f'{col}_rank'] = df.groupby('season')[col].rank(ascending=False, method='dense')
    
    return df


def calculate_defensive_metrics(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate defensive metrics (team as defense).
    
    Args:
        pbp_df: Play-by-play DataFrame
        
    Returns:
        DataFrame with defensive metrics by team/season
    """
    def_metrics = []
    
    for (def_team, season), group in pbp_df.groupby(['defteam', 'season']):
        metrics = {
            'team': def_team,
            'season': season,
            'def_plays_faced': len(group),
            'def_epa_allowed_per_play': group['epa'].mean() if 'epa' in group.columns else 0,
            'def_success_rate_allowed': group['success'].mean() if 'success' in group.columns else 0,
        }
        
        # Sack rate and turnover proxies for DST scoring
        pass_plays = group[group['play_type'] == 'pass']
        if len(pass_plays) > 0:
            # Mock sack rate - in real implementation would come from play-by-play
            metrics['sack_rate'] = np.random.uniform(0.05, 0.15)  
            metrics['int_rate'] = np.random.uniform(0.02, 0.08)
        else:
            metrics['sack_rate'] = 0.08
            metrics['int_rate'] = 0.03
            
        # Mock fumble recovery rate
        metrics['fumble_recovery_rate'] = np.random.uniform(0.01, 0.05)
        
        def_metrics.append(metrics)
    
    return pd.DataFrame(def_metrics)