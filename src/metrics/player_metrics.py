"""Player-level metrics calculation."""

import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_player_metrics(weekly_df: pd.DataFrame, pbp_df: pd.DataFrame = None) -> pd.DataFrame:
    """Calculate comprehensive player metrics from weekly data.
    
    Args:
        weekly_df: Weekly player stats DataFrame
        pbp_df: Play-by-play DataFrame (optional, for advanced metrics)
        
    Returns:
        DataFrame with player metrics aggregated across seasons
    """
    player_metrics = []
    
    # Group by player and calculate metrics
    for player_id, group in weekly_df.groupby('player_id'):
        if len(group) == 0:
            continue
            
        # Get player info
        player_info = group.iloc[0]
        position = player_info['position']
        
        metrics = {
            'player_id': player_id,
            'player_name': player_info['player_name'],
            'position': position,
            'team': player_info['team'],  # Most recent team
            'seasons_played': group['season'].nunique(),
            'games_played': len(group),
        }
        
        # Position-specific metrics
        if position == 'QB':
            metrics.update(_calculate_qb_metrics(group))
        elif position in ['RB']:
            metrics.update(_calculate_rb_metrics(group))
        elif position in ['WR', 'TE']:
            metrics.update(_calculate_receiver_metrics(group))
        
        # Common efficiency metrics for all positions
        metrics.update(_calculate_fantasy_metrics(group))
        
        player_metrics.append(metrics)
    
    return pd.DataFrame(player_metrics)


def _calculate_qb_metrics(group_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate QB-specific metrics."""
    total_games = len(group_df)
    
    # Passing metrics
    total_attempts = group_df['attempts'].sum()
    total_completions = group_df['completions'].sum()
    total_pass_yards = group_df['passing_yards'].sum()
    total_pass_tds = group_df['passing_tds'].sum()
    total_ints = group_df['interceptions'].sum()
    
    # Rushing metrics
    total_carries = group_df['carries'].sum()
    total_rush_yards = group_df['rushing_yards'].sum()
    total_rush_tds = group_df['rushing_tds'].sum()
    
    # Per-game averages
    attempts_per_game = total_attempts / total_games if total_games > 0 else 0
    completions_per_game = total_completions / total_games if total_games > 0 else 0
    
    return {
        'pass_attempts_per_game': attempts_per_game,
        'completions_per_game': completions_per_game,
        'completion_pct': total_completions / total_attempts if total_attempts > 0 else 0,
        'pass_yards_per_game': total_pass_yards / total_games if total_games > 0 else 0,
        'pass_yards_per_attempt': total_pass_yards / total_attempts if total_attempts > 0 else 0,
        'pass_tds_per_game': total_pass_tds / total_games if total_games > 0 else 0,
        'int_rate': total_ints / total_attempts if total_attempts > 0 else 0,
        'rush_attempts_per_game': total_carries / total_games if total_games > 0 else 0,
        'rush_yards_per_game': total_rush_yards / total_games if total_games > 0 else 0,
        'rush_yards_per_carry': total_rush_yards / total_carries if total_carries > 0 else 0,
        'rush_tds_per_game': total_rush_tds / total_games if total_games > 0 else 0,
        # Mock CPOE (Completion Percentage Over Expected) and EPA per dropback
        'cpoe': np.random.uniform(-0.05, 0.08),  # In real implementation, calculated from pbp
        'epa_per_dropback': np.random.uniform(-0.1, 0.3),
        'sack_rate': np.random.uniform(0.05, 0.12),
    }


def _calculate_rb_metrics(group_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate RB-specific metrics."""
    total_games = len(group_df)
    
    # Rushing metrics
    total_carries = group_df['carries'].sum()
    total_rush_yards = group_df['rushing_yards'].sum()
    total_rush_tds = group_df['rushing_tds'].sum()
    
    # Receiving metrics
    total_targets = group_df['targets'].sum()
    total_receptions = group_df['receptions'].sum()
    total_rec_yards = group_df['receiving_yards'].sum()
    total_rec_tds = group_df['receiving_tds'].sum()
    
    # Usage shares (would be calculated relative to team in real implementation)
    carry_share = np.random.uniform(0.1, 0.8)  # Mock team carry share
    target_share = np.random.uniform(0.05, 0.25)  # Mock team target share
    
    return {
        'carries_per_game': total_carries / total_games if total_games > 0 else 0,
        'rush_yards_per_game': total_rush_yards / total_games if total_games > 0 else 0,
        'rush_yards_per_carry': total_rush_yards / total_carries if total_carries > 0 else 0,
        'rush_tds_per_game': total_rush_tds / total_games if total_games > 0 else 0,
        'rush_td_rate': total_rush_tds / total_carries if total_carries > 0 else 0,
        'targets_per_game': total_targets / total_games if total_games > 0 else 0,
        'receptions_per_game': total_receptions / total_games if total_games > 0 else 0,
        'rec_yards_per_game': total_rec_yards / total_games if total_games > 0 else 0,
        'rec_tds_per_game': total_rec_tds / total_games if total_games > 0 else 0,
        'catch_rate': total_receptions / total_targets if total_targets > 0 else 0,
        'yards_per_target': total_rec_yards / total_targets if total_targets > 0 else 0,
        'carry_share': carry_share,
        'target_share': target_share,
        # High-value touch metrics
        'inside_10_carries': np.random.uniform(0.1, 0.4),  # Share of team inside-10 carries
        'goal_line_carries': np.random.uniform(0.15, 0.6),  # Share of team goal-line carries
    }


def _calculate_receiver_metrics(group_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate WR/TE-specific metrics."""
    total_games = len(group_df)
    
    # Receiving metrics
    total_targets = group_df['targets'].sum()
    total_receptions = group_df['receptions'].sum()
    total_rec_yards = group_df['receiving_yards'].sum()
    total_rec_tds = group_df['receiving_tds'].sum()
    
    # Usage shares (mock - would be calculated relative to team)
    target_share = np.random.uniform(0.08, 0.35)
    air_yards_share = np.random.uniform(0.08, 0.35)
    
    return {
        'targets_per_game': total_targets / total_games if total_games > 0 else 0,
        'receptions_per_game': total_receptions / total_games if total_games > 0 else 0,
        'rec_yards_per_game': total_rec_yards / total_games if total_games > 0 else 0,
        'rec_tds_per_game': total_rec_tds / total_games if total_games > 0 else 0,
        'catch_rate': total_receptions / total_targets if total_targets > 0 else 0,
        'yards_per_target': total_rec_yards / total_targets if total_targets > 0 else 0,
        'yards_per_reception': total_rec_yards / total_receptions if total_receptions > 0 else 0,
        'rec_td_rate': total_rec_tds / total_targets if total_targets > 0 else 0,
        'target_share': target_share,
        'air_yards_share': air_yards_share,
        # Advanced metrics (mock values)
        'wopr': target_share * 1.5 + air_yards_share * 0.7,  # Weighted Opportunity Rating
        'racr': np.random.uniform(0.6, 1.4),  # Receiver Air Conversion Ratio
        'adot': np.random.uniform(6, 18),  # Average Depth of Target
        'xyac_per_reception': np.random.uniform(2, 8),  # Expected YAC
        'yac_over_expected': np.random.uniform(-2, 3),  # YAC over expected
        # Target quality
        'red_zone_target_share': np.random.uniform(0.05, 0.4),
        'inside_10_target_share': np.random.uniform(0.05, 0.35),
    }


def _calculate_fantasy_metrics(group_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate DraftKings fantasy points and related metrics."""
    total_games = len(group_df)
    
    # Calculate DK fantasy points for each game
    fantasy_points = []
    for _, game in group_df.iterrows():
        points = 0
        
        # Passing (1 pt per 25 yards, 4 pts per TD, -1 per INT)
        points += game.get('passing_yards', 0) * 0.04  # 1/25
        points += game.get('passing_tds', 0) * 4
        points -= game.get('interceptions', 0) * 1
        
        # Rushing (1 pt per 10 yards, 6 pts per TD)
        points += game.get('rushing_yards', 0) * 0.1  # 1/10
        points += game.get('rushing_tds', 0) * 6
        
        # Receiving (1 pt per reception, 1 pt per 10 yards, 6 pts per TD)
        points += game.get('receptions', 0) * 1  # PPR
        points += game.get('receiving_yards', 0) * 0.1  # 1/10
        points += game.get('receiving_tds', 0) * 6
        
        # Bonuses (mock implementation)
        if game.get('passing_yards', 0) >= 300:
            points += 3
        if game.get('rushing_yards', 0) >= 100:
            points += 3
        if game.get('receiving_yards', 0) >= 100:
            points += 3
        
        fantasy_points.append(points)
    
    fantasy_series = pd.Series(fantasy_points)
    
    return {
        'fantasy_ppg': fantasy_series.mean(),
        'fantasy_total': fantasy_series.sum(),
        'fantasy_floor_p10': fantasy_series.quantile(0.1),
        'fantasy_floor_p25': fantasy_series.quantile(0.25),
        'fantasy_ceil_p75': fantasy_series.quantile(0.75),
        'fantasy_ceil_p90': fantasy_series.quantile(0.9),
        'fantasy_std': fantasy_series.std(),
        'fantasy_cv': fantasy_series.std() / fantasy_series.mean() if fantasy_series.mean() > 0 else 0,
        'boom_rate_15': (fantasy_series >= 15).mean(),  # Position-agnostic boom threshold
        'boom_rate_20': (fantasy_series >= 20).mean(),
        'bust_rate_5': (fantasy_series <= 5).mean(),   # Bust threshold
    }


def calculate_dst_metrics(weekly_df: pd.DataFrame, team_def_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Defense/Special Teams fantasy metrics.
    
    Args:
        weekly_df: Weekly stats (not used for DST but kept for consistency)
        team_def_df: Team defensive metrics from team_metrics.py
        
    Returns:
        DataFrame with DST fantasy metrics
    """
    dst_metrics = []
    
    for _, row in team_def_df.iterrows():
        team = row['team']
        season = row['season']
        
        # Mock DST fantasy scoring based on defensive metrics
        sacks_per_game = np.random.uniform(1.5, 4.0)
        ints_per_game = np.random.uniform(0.5, 1.5)
        fumbles_per_game = np.random.uniform(0.3, 1.2)
        tds_per_game = np.random.uniform(0.05, 0.4)
        
        # Points allowed (affects DST scoring)
        points_allowed_per_game = np.random.uniform(16, 28)
        
        # Calculate DST fantasy points per game
        fantasy_ppg = (
            sacks_per_game * 1 +  # 1 pt per sack
            ints_per_game * 2 +   # 2 pts per interception
            fumbles_per_game * 2 + # 2 pts per fumble recovery
            tds_per_game * 6      # 6 pts per TD
        )
        
        # Points allowed penalty
        if points_allowed_per_game <= 6:
            fantasy_ppg += 10
        elif points_allowed_per_game <= 13:
            fantasy_ppg += 7
        elif points_allowed_per_game <= 20:
            fantasy_ppg += 4
        elif points_allowed_per_game <= 27:
            fantasy_ppg += 1
        elif points_allowed_per_game <= 34:
            fantasy_ppg += 0
        else:
            fantasy_ppg -= 1
        
        dst_metrics.append({
            'player_id': f'{team}_DST',
            'player_name': f'{team} DST',
            'position': 'DST',
            'team': team,
            'season': season,
            'sacks_per_game': sacks_per_game,
            'ints_per_game': ints_per_game,
            'fumbles_per_game': fumbles_per_game,
            'tds_per_game': tds_per_game,
            'points_allowed_per_game': points_allowed_per_game,
            'fantasy_ppg': fantasy_ppg,
            'opp_sack_rate': row.get('sack_rate', 0.08),
            'opp_int_rate': row.get('int_rate', 0.03),
            'opp_fumble_rate': row.get('fumble_recovery_rate', 0.02),
        })
    
    return pd.DataFrame(dst_metrics)