"""
Data preparation and feature engineering for NFL play-by-play data.

This module cleans raw nfl_data_py data and derives key flags and metrics
used throughout the simulation system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def clean_pbp_data(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare play-by-play data.
    
    Args:
        pbp_df: Raw play-by-play DataFrame from nfl_data_py
        
    Returns:
        Cleaned DataFrame with derived flags
    """
    df = pbp_df.copy()
    
    logger.info(f"Cleaning {len(df)} play-by-play rows")
    
    # Basic data type fixes
    df = _fix_data_types(df)
    
    # Derive key flags
    df = _add_play_type_flags(df)
    df = _add_situation_flags(df)
    df = _add_success_flags(df)
    df = _add_timing_metrics(df)
    
    # Filter to meaningful plays
    df = _filter_meaningful_plays(df)
    
    logger.info(f"Cleaned data contains {len(df)} meaningful plays")
    
    return df


def _fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Fix common data type issues in play-by-play data."""
    
    # Ensure numeric columns are numeric
    numeric_cols = [
        'game_seconds_remaining', 'down', 'ydstogo', 'yardline_100',
        'yards_gained', 'epa', 'wpa', 'wp', 'cp', 'cpoe',
        'air_yards', 'yards_after_catch', 'xyac_epa', 'xyac_mean_yardage'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fix season/week
    if 'season' in df.columns:
        df['season'] = pd.to_numeric(df['season'], errors='coerce')
    if 'week' in df.columns:
        df['week'] = pd.to_numeric(df['week'], errors='coerce')
    
    return df


def _add_play_type_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add flags for different play types."""
    
    # Dropback plays (pass attempts + sacks)
    df['is_dropback'] = (
        (df.get('pass_attempt', 0) == 1) | 
        (df.get('sack', 0) == 1)
    ).astype(int)
    
    # Rush attempts (excluding QB kneels and spikes)
    df['is_rush'] = (
        (df.get('rush_attempt', 0) == 1) & 
        (df.get('qb_kneel', 0) != 1) & 
        (df.get('qb_spike', 0) != 1)
    ).astype(int)
    
    # Red zone plays
    df['is_red_zone'] = (df.get('yardline_100', 100) <= 20).astype(int)
    
    # Goal line plays  
    df['is_goal_line'] = (df.get('yardline_100', 100) <= 5).astype(int)
    
    # Two minute warning situations
    df['is_two_minute'] = (
        (df.get('game_seconds_remaining', 0) <= 120) |
        ((df.get('game_seconds_remaining', 0) <= 1920) & (df.get('game_seconds_remaining', 0) > 1800))
    ).astype(int)
    
    return df


def _add_situation_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add situational context flags."""
    
    # Score differential - positive means offense is ahead
    df['score_diff'] = df.get('posteam_score', 0) - df.get('defteam_score', 0)
    
    # Game state categories
    df['game_state'] = 'neutral'
    df.loc[df['score_diff'] >= 17, 'game_state'] = 'blowout_ahead'
    df.loc[df['score_diff'] <= -17, 'game_state'] = 'blowout_behind'
    df.loc[df['score_diff'].between(8, 16), 'game_state'] = 'ahead'
    df.loc[df['score_diff'].between(-16, -8), 'game_state'] = 'behind'
    
    # Neutral game script (close games, middle quarters)
    df['is_neutral_script'] = (
        (abs(df['score_diff']) <= 7) &
        (df.get('qtr', 1).isin([2, 3])) &
        (df.get('game_seconds_remaining', 0) > 120)
    ).astype(int)
    
    # Down and distance categories
    df['down_distance'] = 'other'
    df.loc[df.get('down', 1) == 1, 'down_distance'] = 'first_down'
    df.loc[
        (df.get('down', 1) == 2) & (df.get('ydstogo', 10) <= 7), 
        'down_distance'
    ] = 'second_short'
    df.loc[
        (df.get('down', 1) == 2) & (df.get('ydstogo', 10) > 7), 
        'down_distance'
    ] = 'second_long'
    df.loc[
        (df.get('down', 1).isin([3, 4])) & (df.get('ydstogo', 10) <= 3), 
        'down_distance'
    ] = 'third_fourth_short'
    df.loc[
        (df.get('down', 1).isin([3, 4])) & (df.get('ydstogo', 10) > 3), 
        'down_distance'
    ] = 'third_fourth_long'
    
    return df


def _add_success_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add success rate flags based on EPA and down/distance."""
    
    # EPA success
    df['epa_success'] = (df.get('epa', 0) > 0).astype(int)
    
    # Traditional success rate (1st down or TD)
    # First down: gain 45% of yards needed
    # Second down: gain 60% of yards needed  
    # Third/Fourth down: convert or TD
    df['traditional_success'] = 0
    
    # First down success
    first_down_mask = (df.get('down', 1) == 1)
    df.loc[
        first_down_mask & (df.get('yards_gained', 0) >= df.get('ydstogo', 10) * 0.45),
        'traditional_success'
    ] = 1
    
    # Second down success
    second_down_mask = (df.get('down', 1) == 2)
    df.loc[
        second_down_mask & (df.get('yards_gained', 0) >= df.get('ydstogo', 10) * 0.6),
        'traditional_success'
    ] = 1
    
    # Third/fourth down success (convert or TD)
    third_fourth_mask = df.get('down', 1).isin([3, 4])
    df.loc[
        third_fourth_mask & (
            (df.get('yards_gained', 0) >= df.get('ydstogo', 10)) |
            (df.get('touchdown', 0) == 1)
        ),
        'traditional_success'
    ] = 1
    
    # Always success on TD
    df.loc[df.get('touchdown', 0) == 1, 'traditional_success'] = 1
    
    return df


def _add_timing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add timing and pace metrics."""
    
    # Seconds per play (approximate)
    # NFL average is around 27-30 seconds between plays
    if 'play_clock' in df.columns:
        df['seconds_per_play'] = df['play_clock']
    else:
        # Estimate based on play type and situation
        df['seconds_per_play'] = 28  # Default
        
        # Hurry up situations take less time
        df.loc[df['is_two_minute'] == 1, 'seconds_per_play'] = 20
        
        # Running plays typically faster
        df.loc[df['is_rush'] == 1, 'seconds_per_play'] = 25
        
        # Incomplete passes stop clock
        df.loc[
            (df.get('incomplete_pass', 0) == 1) | (df.get('interception', 0) == 1),
            'seconds_per_play'
        ] = 22
    
    # Early down tendency
    df['is_early_down'] = df.get('down', 1).isin([1, 2]).astype(int)
    
    return df


def _filter_meaningful_plays(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to meaningful offensive plays."""
    
    # Remove special teams, penalties, timeouts, etc.
    meaningful_mask = (
        (df.get('play_type', '').isin(['pass', 'run'])) |
        (df['is_dropback'] == 1) |
        (df['is_rush'] == 1)
    )
    
    # Remove plays with missing critical data
    meaningful_mask = meaningful_mask & (
        df.get('posteam', '').notna() &
        df.get('defteam', '').notna() &
        df.get('down', 0).notna() &
        (df.get('down', 0) > 0)
    )
    
    # Remove end-of-game kneels
    kneel_mask = (
        (df.get('qb_kneel', 0) == 1) | 
        (df.get('qb_spike', 0) == 1)
    )
    meaningful_mask = meaningful_mask & (~kneel_mask)
    
    return df[meaningful_mask].copy()


def add_target_share_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add target share and air yard share calculations."""
    
    # Only for pass plays
    pass_plays = df[df['is_dropback'] == 1].copy()
    
    if pass_plays.empty:
        return df
    
    # Calculate team totals by game
    team_game_totals = pass_plays.groupby(['game_id', 'posteam']).agg({
        'pass_attempt': 'sum',
        'air_yards': 'sum',
        'yards_gained': 'sum'
    }).reset_index()
    
    team_game_totals.columns = ['game_id', 'posteam', 'team_pass_attempts', 'team_air_yards', 'team_pass_yards']
    
    # Merge back to main dataframe
    df = df.merge(
        team_game_totals,
        on=['game_id', 'posteam'],
        how='left'
    )
    
    return df


def calculate_drive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate drive-level metrics."""
    
    if 'drive' not in df.columns:
        return df
    
    # Drive outcomes
    drive_outcomes = df.groupby(['game_id', 'posteam', 'drive']).agg({
        'epa': 'sum',
        'yards_gained': 'sum',
        'is_dropback': 'sum',
        'is_rush': 'sum',
        'touchdown': 'max',
        'field_goal_result': lambda x: (x == 'made').astype(int).max(),
        'turnover': 'max'
    }).reset_index()
    
    drive_outcomes['total_plays'] = drive_outcomes['is_dropback'] + drive_outcomes['is_rush']
    drive_outcomes['drive_success'] = (
        (drive_outcomes['touchdown'] == 1) |
        (drive_outcomes['field_goal_result'] == 1)
    ).astype(int)
    
    # Merge back
    df = df.merge(
        drive_outcomes[['game_id', 'posteam', 'drive', 'total_plays', 'drive_success']],
        on=['game_id', 'posteam', 'drive'],
        how='left'
    )
    
    return df


def prepare_weekly_data(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare weekly player stats data.
    
    Args:
        weekly_df: Raw weekly data from nfl_data_py
        
    Returns:
        Cleaned weekly stats DataFrame
    """
    df = weekly_df.copy()
    
    logger.info(f"Preparing {len(df)} weekly stat rows")
    
    # Fix data types
    numeric_cols = [
        'passing_yards', 'passing_tds', 'interceptions', 'passing_attempts',
        'rushing_yards', 'rushing_tds', 'rushing_attempts',
        'receiving_yards', 'receiving_tds', 'receptions', 'targets',
        'fantasy_points', 'fantasy_points_ppr'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Add position groups
    df['pos_group'] = 'OTHER'
    df.loc[df.get('position', '') == 'QB', 'pos_group'] = 'QB'
    df.loc[df.get('position', '').isin(['RB', 'FB']), 'pos_group'] = 'RB'
    df.loc[df.get('position', '').isin(['WR']), 'pos_group'] = 'WR'
    df.loc[df.get('position', '').isin(['TE']), 'pos_group'] = 'TE'
    df.loc[df.get('position', '').isin(['K']), 'pos_group'] = 'K'
    
    # Calculate DraftKings fantasy points if not present
    if 'fantasy_points_dk' not in df.columns:
        df['fantasy_points_dk'] = _calculate_dk_points_from_weekly(df)
    
    logger.info(f"Prepared weekly data for {df['player_display_name'].nunique()} unique players")
    
    return df


def _calculate_dk_points_from_weekly(df: pd.DataFrame) -> pd.Series:
    """Calculate DraftKings points from weekly stat columns."""
    
    points = pd.Series(0.0, index=df.index)
    
    # Passing (4 pts/TD, -1 INT, 1 pt/25 yards)
    points += df.get('passing_yards', 0) * 0.04
    points += df.get('passing_tds', 0) * 4
    points += df.get('interceptions', 0) * -1
    
    # Rushing (6 pts/TD, 1 pt/10 yards)
    points += df.get('rushing_yards', 0) * 0.1
    points += df.get('rushing_tds', 0) * 6
    
    # Receiving (1 pt/reception, 6 pts/TD, 1 pt/10 yards)
    points += df.get('receptions', 0) * 1
    points += df.get('receiving_yards', 0) * 0.1
    points += df.get('receiving_tds', 0) * 6
    
    # Fumbles lost
    points += df.get('fumbles_lost', 0) * -1
    
    return points.round(2)