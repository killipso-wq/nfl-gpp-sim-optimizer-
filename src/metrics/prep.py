"""Data preparation and filtering for NFL play-by-play data."""

import pandas as pd
import numpy as np
from typing import Optional


def clean_pbp_data(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare play-by-play data.
    
    Args:
        pbp_df: Raw play-by-play data
        
    Returns:
        Cleaned DataFrame
    """
    df = pbp_df.copy()
    
    # Basic data cleaning
    df = df.dropna(subset=['posteam', 'defteam', 'play_type'])
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['epa', 'win_prob', 'score_differential', 'yardline_100', 'air_yards', 'yards_after_catch']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def filter_neutral_situations(pbp_df: pd.DataFrame, 
                            wp_min: float = 0.2, 
                            wp_max: float = 0.8,
                            score_diff_max: int = 7,
                            exclude_fourth_down: bool = True) -> pd.DataFrame:
    """Filter play-by-play data to neutral game situations.
    
    Neutral situations are defined as:
    - Win probability between 20-80%
    - Score differential within ±7 points  
    - Exclude 4th down plays (typically)
    - Regular play types only
    
    Args:
        pbp_df: Play-by-play DataFrame
        wp_min: Minimum win probability
        wp_max: Maximum win probability
        score_diff_max: Maximum absolute score differential
        exclude_fourth_down: Whether to exclude 4th down plays
        
    Returns:
        Filtered DataFrame
    """
    df = pbp_df.copy()
    
    # Win probability filter
    df = df[(df['win_prob'] >= wp_min) & (df['win_prob'] <= wp_max)]
    
    # Score differential filter  
    df = df[abs(df['score_differential']) <= score_diff_max]
    
    # Down filter
    if exclude_fourth_down and 'down' in df.columns:
        df = df[df['down'].isin([1, 2, 3])]
    
    # Play type filter - only regular offensive plays
    if 'play_type' in df.columns:
        df = df[df['play_type'].isin(['pass', 'run'])]
    
    return df


def add_derived_features(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to play-by-play data.
    
    Args:
        pbp_df: Play-by-play DataFrame
        
    Returns:
        DataFrame with additional features
    """
    df = pbp_df.copy()
    
    # Pass rate features
    df['is_pass'] = (df['play_type'] == 'pass').astype(int)
    df['is_run'] = (df['play_type'] == 'run').astype(int)
    
    # Red zone indicators
    if 'yardline_100' in df.columns:
        df['red_zone'] = (df['yardline_100'] <= 20).astype(int)
        df['goal_line'] = (df['yardline_100'] <= 5).astype(int)
        df['inside_10'] = (df['yardline_100'] <= 10).astype(int)
    
    # Down and distance categories
    if 'down' in df.columns and 'ydstogo' in df.columns:
        df['early_down'] = (df['down'].isin([1, 2])).astype(int)
        df['third_down'] = (df['down'] == 3).astype(int)
        df['long_distance'] = (df['ydstogo'] >= 7).astype(int)
        df['short_distance'] = (df['ydstogo'] <= 3).astype(int)
    
    # Game flow indicators
    if 'score_differential' in df.columns:
        df['ahead'] = (df['score_differential'] > 0).astype(int)
        df['behind'] = (df['score_differential'] < 0).astype(int)
        df['tied'] = (df['score_differential'] == 0).astype(int)
    
    # Air yards categories for passes
    if 'air_yards' in df.columns:
        df.loc[df['play_type'] == 'pass', 'deep_pass'] = (df['air_yards'] >= 15).astype(int)
        df.loc[df['play_type'] == 'pass', 'short_pass'] = (df['air_yards'] <= 5).astype(int)
    
    return df


def calculate_team_pace(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate team pace metrics from play-by-play data.
    
    Args:
        pbp_df: Play-by-play DataFrame
        
    Returns:
        DataFrame with team pace metrics by season
    """
    # Group by team and season
    pace_data = []
    
    for (team, season), group in pbp_df.groupby(['posteam', 'season']):
        # Count total plays
        total_plays = len(group)
        
        # Count games (approximate)
        games = group['game_id'].nunique()
        
        if games > 0:
            pace_data.append({
                'team': team,
                'season': season,
                'total_plays': total_plays,
                'games': games,
                'plays_per_game': total_plays / games,
            })
    
    return pd.DataFrame(pace_data)


def calculate_neutral_pass_rate(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate neutral situation pass rates by team.
    
    Args:
        pbp_df: Play-by-play DataFrame (should be pre-filtered for neutral situations)
        
    Returns:
        DataFrame with team neutral pass rates
    """
    pass_rate_data = []
    
    for (team, season), group in pbp_df.groupby(['posteam', 'season']):
        if len(group) > 0:
            pass_plays = (group['play_type'] == 'pass').sum()
            total_plays = len(group)
            
            pass_rate_data.append({
                'team': team,
                'season': season,
                'neutral_passes': pass_plays,
                'neutral_total_plays': total_plays,
                'neutral_pass_rate': pass_plays / total_plays if total_plays > 0 else 0,
            })
    
    return pd.DataFrame(pass_rate_data)