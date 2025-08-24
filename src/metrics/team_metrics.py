"""
Team-level metrics calculation from play-by-play data.

This module calculates team pace, pass rate tendencies, EPA efficiency,
and other metrics used to build team priors for simulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_team_metrics_by_week(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team-level metrics by season/week from play-by-play data.
    
    Args:
        pbp_df: Cleaned play-by-play DataFrame
        
    Returns:
        DataFrame with team-week level metrics
    """
    logger.info("Calculating team metrics by week")
    
    # Filter to meaningful plays
    meaningful_plays = pbp_df[
        (pbp_df.get('is_dropback', 0) == 1) | 
        (pbp_df.get('is_rush', 0) == 1)
    ].copy()
    
    if meaningful_plays.empty:
        logger.warning("No meaningful plays found for team metrics")
        return pd.DataFrame()
    
    # Group by team-week
    team_weeks = []
    
    for (season, week, team), group in meaningful_plays.groupby(['season', 'week', 'posteam']):
        if pd.isna(team) or team == '':
            continue
            
        metrics = _calculate_single_team_week_metrics(group)
        metrics.update({
            'season': season,
            'week': week,
            'team': team,
            'n_plays': len(group)
        })
        
        team_weeks.append(metrics)
    
    if not team_weeks:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(team_weeks)
    logger.info(f"Calculated metrics for {len(result_df)} team-weeks")
    
    return result_df


def _calculate_single_team_week_metrics(plays_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate metrics for a single team-week."""
    
    metrics = {}
    
    # Basic play counts
    n_dropbacks = (plays_df.get('is_dropback', 0) == 1).sum()
    n_rushes = (plays_df.get('is_rush', 0) == 1).sum() 
    n_total = n_dropbacks + n_rushes
    
    if n_total == 0:
        return {k: np.nan for k in [
            'pace', 'pass_rate', 'neutral_pass_rate', 'epa_per_play', 
            'success_rate', 'red_zone_td_rate', 'proe'
        ]}
    
    # Pace (plays per minute of possession time - estimated)
    # Use play count as proxy - typical team runs ~65-70 plays per game
    metrics['pace'] = n_total  # Will be adjusted for game time later
    
    # Pass rate (overall)
    metrics['pass_rate'] = n_dropbacks / n_total if n_total > 0 else 0
    
    # Neutral script pass rate (close games, middle quarters)
    neutral_plays = plays_df[plays_df.get('is_neutral_script', 0) == 1]
    if len(neutral_plays) > 0:
        n_neutral_dropbacks = (neutral_plays.get('is_dropback', 0) == 1).sum()
        n_neutral_total = len(neutral_plays) 
        metrics['neutral_pass_rate'] = n_neutral_dropbacks / n_neutral_total
    else:
        metrics['neutral_pass_rate'] = metrics['pass_rate']
    
    # EPA efficiency
    epa_values = plays_df.get('epa', []).dropna()
    metrics['epa_per_play'] = epa_values.mean() if len(epa_values) > 0 else 0
    
    # Success rate
    success_plays = (plays_df.get('epa_success', 0) == 1).sum()
    metrics['success_rate'] = success_plays / n_total if n_total > 0 else 0
    
    # Red zone touchdown rate
    rz_plays = plays_df[plays_df.get('is_red_zone', 0) == 1]
    if len(rz_plays) > 0:
        rz_tds = (rz_plays.get('touchdown', 0) == 1).sum()
        metrics['red_zone_td_rate'] = rz_tds / len(rz_plays)
    else:
        metrics['red_zone_td_rate'] = 0
    
    # Pass Rate Over Expected (PROE) - proxy
    # Compare actual pass rate to expected based on down/distance/score/time
    expected_pass_rate = _calculate_expected_pass_rate(plays_df)
    metrics['proe'] = metrics['pass_rate'] - expected_pass_rate
    
    return metrics


def _calculate_expected_pass_rate(plays_df: pd.DataFrame) -> float:
    """
    Calculate expected pass rate based on situational factors.
    
    This is a simplified model - in reality you'd use more sophisticated
    situational factors and historical data.
    """
    # Base rates by down
    down_pass_rates = {1: 0.55, 2: 0.60, 3: 0.75, 4: 0.70}
    
    expected_rates = []
    
    for _, play in plays_df.iterrows():
        down = play.get('down', 1)
        ydstogo = play.get('ydstogo', 10)
        score_diff = play.get('score_diff', 0)
        seconds_remaining = play.get('game_seconds_remaining', 1800)
        
        # Base rate by down
        base_rate = down_pass_rates.get(down, 0.6)
        
        # Adjust for yards to go
        if ydstogo >= 10:
            base_rate += 0.1
        elif ydstogo <= 3:
            base_rate -= 0.1
            
        # Adjust for score differential
        if score_diff <= -14:  # Behind by 2+ TDs
            base_rate += 0.2
        elif score_diff >= 14:  # Ahead by 2+ TDs  
            base_rate -= 0.2
        elif abs(score_diff) <= 3:  # Close game
            base_rate += 0.0  # No adjustment
        
        # Adjust for time remaining (late game situations)
        if seconds_remaining < 120:  # Under 2 minutes
            if score_diff < 0:  # Behind
                base_rate += 0.3
            elif score_diff > 7:  # Ahead by more than 1 TD
                base_rate -= 0.3
        
        expected_rates.append(max(0, min(1, base_rate)))
    
    return np.mean(expected_rates) if expected_rates else 0.6


def calculate_team_season_aggregates(team_week_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate team-week metrics to season-level averages.
    
    Args:
        team_week_df: DataFrame with team-week metrics
        
    Returns:
        DataFrame with team-season aggregates
    """
    if team_week_df.empty:
        return pd.DataFrame()
    
    logger.info("Aggregating team metrics to season level")
    
    # Group by team-season
    agg_metrics = []
    
    for (season, team), group in team_week_df.groupby(['season', 'team']):
        if len(group) < 4:  # Need at least 4 weeks of data
            continue
            
        # Weight by number of plays when averaging rates
        weights = group['n_plays']
        total_plays = weights.sum()
        
        if total_plays == 0:
            continue
        
        # Weighted averages for rate stats
        metrics = {
            'season': season,
            'team': team,
            'weeks_played': len(group),
            'total_plays': total_plays,
            'avg_pace': np.average(group['pace'], weights=weights),
            'pass_rate': np.average(group['pass_rate'], weights=weights),
            'neutral_pass_rate': np.average(group['neutral_pass_rate'], weights=weights),
            'epa_per_play': np.average(group['epa_per_play'], weights=weights),
            'success_rate': np.average(group['success_rate'], weights=weights),
            'red_zone_td_rate': np.average(group['red_zone_td_rate'], weights=weights),
            'proe': np.average(group['proe'], weights=weights)
        }
        
        # Add consistency metrics (standard deviations)
        for col in ['pace', 'pass_rate', 'epa_per_play']:
            if col in group.columns:
                metrics[f'{col}_std'] = group[col].std()
        
        agg_metrics.append(metrics)
    
    if not agg_metrics:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(agg_metrics)
    logger.info(f"Aggregated metrics for {len(result_df)} team-seasons")
    
    return result_df


def calculate_defensive_metrics(pbp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate defensive metrics from the perspective of the defense.
    
    Args:
        pbp_df: Play-by-play DataFrame
        
    Returns:
        DataFrame with defensive metrics by team-season
    """
    logger.info("Calculating defensive metrics")
    
    # Get plays from defensive team perspective
    def_plays = pbp_df[
        (pbp_df.get('is_dropback', 0) == 1) | 
        (pbp_df.get('is_rush', 0) == 1)
    ].copy()
    
    if def_plays.empty:
        return pd.DataFrame()
    
    # Group by defensive team
    def_metrics = []
    
    for (season, def_team), group in def_plays.groupby(['season', 'defteam']):
        if pd.isna(def_team) or def_team == '' or len(group) < 50:
            continue
        
        n_plays = len(group)
        epa_allowed = group.get('epa', []).mean()
        success_rate_allowed = (group.get('epa_success', 0) == 1).mean()
        
        # Pass defense
        pass_plays = group[group.get('is_dropback', 0) == 1]
        if len(pass_plays) > 0:
            pass_epa_allowed = pass_plays.get('epa', []).mean()
            sacks = (pass_plays.get('sack', 0) == 1).sum()
            sack_rate = sacks / len(pass_plays)
        else:
            pass_epa_allowed = 0
            sack_rate = 0
        
        # Run defense  
        rush_plays = group[group.get('is_rush', 0) == 1]
        if len(rush_plays) > 0:
            rush_epa_allowed = rush_plays.get('epa', []).mean()
            stuff_rate = (rush_plays.get('yards_gained', 0) <= 0).mean()
        else:
            rush_epa_allowed = 0
            stuff_rate = 0
        
        def_metrics.append({
            'season': season,
            'def_team': def_team,
            'def_plays': n_plays,
            'def_epa_allowed': epa_allowed,
            'def_success_rate_allowed': success_rate_allowed,
            'def_pass_epa_allowed': pass_epa_allowed,
            'def_sack_rate': sack_rate,
            'def_rush_epa_allowed': rush_epa_allowed,
            'def_stuff_rate': stuff_rate
        })
    
    if not def_metrics:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(def_metrics)
    logger.info(f"Calculated defensive metrics for {len(result_df)} team-seasons")
    
    return result_df


def calculate_game_environment_metrics(pbp_df: pd.DataFrame, schedules_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate game environment metrics (pace, total plays, etc.) by game.
    
    Args:
        pbp_df: Play-by-play DataFrame
        schedules_df: Optional schedules DataFrame for additional game info
        
    Returns:
        DataFrame with game-level environment metrics
    """
    logger.info("Calculating game environment metrics")
    
    meaningful_plays = pbp_df[
        (pbp_df.get('is_dropback', 0) == 1) | 
        (pbp_df.get('is_rush', 0) == 1)
    ].copy()
    
    if meaningful_plays.empty:
        return pd.DataFrame()
    
    # Group by game
    game_metrics = []
    
    for game_id, group in meaningful_plays.groupby('game_id'):
        if pd.isna(game_id) or len(group) < 20:  # Need reasonable sample
            continue
        
        # Basic game info
        season = group['season'].iloc[0] if 'season' in group.columns else np.nan
        week = group['week'].iloc[0] if 'week' in group.columns else np.nan
        
        # Teams in the game
        teams = group['posteam'].unique()
        teams = [t for t in teams if pd.notna(t) and t != '']
        
        if len(teams) != 2:
            continue
        
        total_plays = len(group)
        
        # Pass/run split
        n_dropbacks = (group.get('is_dropback', 0) == 1).sum()
        pass_rate = n_dropbacks / total_plays if total_plays > 0 else 0
        
        # Scoring environment
        avg_epa = group.get('epa', []).mean()
        high_scoring = (group.get('touchdown', 0) == 1).sum()
        
        # Pace proxy - total plays is main indicator
        pace_proxy = total_plays
        
        game_metrics.append({
            'game_id': game_id,
            'season': season,
            'week': week,
            'home_team': teams[0],  # Simplified - would need actual home/away
            'away_team': teams[1],
            'total_plays': total_plays,
            'game_pass_rate': pass_rate,
            'game_epa_per_play': avg_epa,
            'game_tds': high_scoring,
            'pace_proxy': pace_proxy
        })
    
    if not game_metrics:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(game_metrics)
    logger.info(f"Calculated game environment metrics for {len(result_df)} games")
    
    return result_df