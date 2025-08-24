"""
Player-level metrics calculation from play-by-play and weekly data.

This module calculates advanced player metrics like WOPR, RACR, XYAC,
High-Value Touches, and efficiency metrics for building player priors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_qb_metrics(pbp_df: pd.DataFrame, weekly_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate QB-specific metrics from play-by-play data.
    
    Args:
        pbp_df: Play-by-play DataFrame
        weekly_df: Optional weekly stats DataFrame for validation
        
    Returns:
        DataFrame with QB metrics by player-season
    """
    logger.info("Calculating QB metrics")
    
    # Filter to QB dropbacks
    qb_plays = pbp_df[
        (pbp_df.get('is_dropback', 0) == 1) & 
        pbp_df['passer_player_name'].notna()
    ].copy()
    
    if qb_plays.empty:
        logger.warning("No QB plays found")
        return pd.DataFrame()
    
    qb_metrics = []
    
    # Group by QB-season
    for (season, qb_name), group in qb_plays.groupby(['season', 'passer_player_name']):
        if len(group) < 50:  # Minimum sample size
            continue
            
        team = group['posteam'].mode().iloc[0] if len(group['posteam'].mode()) > 0 else ''
        n_dropbacks = len(group)
        
        # Basic passing metrics
        completions = (group.get('complete_pass', 0) == 1).sum()
        attempts = (group.get('pass_attempt', 0) == 1).sum()
        comp_pct = completions / attempts if attempts > 0 else 0
        
        # EPA metrics
        epa_per_db = group.get('epa', []).mean()
        success_rate = (group.get('epa_success', 0) == 1).mean()
        
        # CPOE (Completion Percentage Over Expected)
        cpoe_values = group.get('cpoe', []).dropna()
        avg_cpoe = cpoe_values.mean() if len(cpoe_values) > 0 else 0
        
        # Air yards and YAC
        air_yards = group.get('air_yards', []).dropna()
        avg_air_yards = air_yards.mean() if len(air_yards) > 0 else 0
        
        yac_values = group.get('yards_after_catch', []).dropna()
        avg_yac = yac_values.mean() if len(yac_values) > 0 else 0
        
        # Deep ball metrics (20+ air yards)
        deep_throws = group[group.get('air_yards', 0) >= 20]
        deep_attempts = len(deep_throws)
        deep_comp_rate = (deep_throws.get('complete_pass', 0) == 1).mean() if deep_attempts > 0 else 0
        
        # Pressure metrics
        pressures = (group.get('qb_hit', 0) == 1) | (group.get('sack', 0) == 1)
        pressure_rate = pressures.mean()
        
        under_pressure = group[pressures]
        pressure_epa = under_pressure.get('epa', []).mean() if len(under_pressure) > 0 else 0
        
        # Red zone efficiency
        rz_plays = group[group.get('is_red_zone', 0) == 1]
        rz_tds = (rz_plays.get('touchdown', 0) == 1).sum()
        rz_attempts = (rz_plays.get('pass_attempt', 0) == 1).sum()
        rz_td_rate = rz_tds / rz_attempts if rz_attempts > 0 else 0
        
        # Scrambling (QB runs)
        qb_rushes = pbp_df[
            (pbp_df.get('is_rush', 0) == 1) & 
            (pbp_df['rusher_player_name'] == qb_name) &
            (pbp_df['season'] == season)
        ]
        scramble_rate = len(qb_rushes) / n_dropbacks if n_dropbacks > 0 else 0
        scramble_epa = qb_rushes.get('epa', []).mean() if len(qb_rushes) > 0 else 0
        
        qb_metrics.append({
            'season': season,
            'player_name': qb_name,
            'position': 'QB',
            'team': team,
            'dropbacks': n_dropbacks,
            'attempts': attempts,
            'completions': completions,
            'comp_pct': comp_pct,
            'epa_per_db': epa_per_db,
            'success_rate': success_rate,
            'cpoe': avg_cpoe,
            'avg_air_yards': avg_air_yards,
            'avg_yac': avg_yac,
            'deep_attempts': deep_attempts,
            'deep_comp_rate': deep_comp_rate,
            'pressure_rate': pressure_rate,
            'pressure_epa': pressure_epa,
            'rz_td_rate': rz_td_rate,
            'scramble_rate': scramble_rate,
            'scramble_epa': scramble_epa
        })
    
    if not qb_metrics:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(qb_metrics)
    logger.info(f"Calculated QB metrics for {len(result_df)} player-seasons")
    
    return result_df


def calculate_rb_metrics(pbp_df: pd.DataFrame, weekly_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate RB-specific metrics including High-Value Touches.
    
    Args:
        pbp_df: Play-by-play DataFrame
        weekly_df: Optional weekly stats DataFrame
        
    Returns:
        DataFrame with RB metrics by player-season
    """
    logger.info("Calculating RB metrics")
    
    # Get RB carries
    rb_rushes = pbp_df[
        (pbp_df.get('is_rush', 0) == 1) & 
        pbp_df['rusher_player_name'].notna()
    ].copy()
    
    # Get RB targets  
    rb_targets = pbp_df[
        (pbp_df.get('is_dropback', 0) == 1) &
        pbp_df['receiver_player_name'].notna()
    ].copy()
    
    # Combine and identify RBs (those with significant rushing)
    all_rb_plays = pd.concat([rb_rushes, rb_targets], ignore_index=True)
    
    if all_rb_plays.empty:
        return pd.DataFrame()
    
    rb_metrics = []
    
    # First pass: identify RBs by rushing volume
    rb_names = rb_rushes.groupby(['season', 'rusher_player_name']).size()
    rb_names = rb_names[rb_names >= 30].index.tolist()  # 30+ carries to be considered RB
    
    for season, player_name in rb_names:
        # Get all touches for this RB
        player_rushes = rb_rushes[
            (rb_rushes['season'] == season) & 
            (rb_rushes['rusher_player_name'] == player_name)
        ]
        
        player_targets = rb_targets[
            (rb_targets['season'] == season) & 
            (rb_targets['receiver_player_name'] == player_name)
        ]
        
        team = player_rushes['posteam'].mode().iloc[0] if len(player_rushes) > 0 else ''
        
        # Rushing metrics
        n_carries = len(player_rushes)
        rush_yards = player_rushes.get('yards_gained', []).sum()
        rush_tds = (player_rushes.get('touchdown', 0) == 1).sum()
        rush_epa = player_rushes.get('epa', []).mean()
        
        # Success rates
        rush_success = (player_rushes.get('epa_success', 0) == 1).mean()
        
        # High-Value Touches (HVT) - carries inside 10, 3rd/4th down short, 2-minute drill
        hvt_carries = player_rushes[
            (player_rushes.get('yardline_100', 100) <= 10) |  # Goal line
            (player_rushes.get('is_goal_line', 0) == 1) |
            ((player_rushes.get('down', 1).isin([3, 4])) & (player_rushes.get('ydstogo', 10) <= 2)) |  # 3rd/4th and short
            (player_rushes.get('is_two_minute', 0) == 1)  # 2-minute drill
        ]
        hvt_carry_count = len(hvt_carries)
        hvt_carry_rate = hvt_carry_count / n_carries if n_carries > 0 else 0
        
        # Receiving metrics
        n_targets = len(player_targets)
        catches = (player_targets.get('complete_pass', 0) == 1).sum()
        catch_rate = catches / n_targets if n_targets > 0 else 0
        
        rec_yards = player_targets[player_targets.get('complete_pass', 0) == 1].get('yards_gained', []).sum()
        rec_tds = (player_targets.get('touchdown', 0) == 1).sum()
        rec_epa = player_targets.get('epa', []).mean()
        
        # High-Value Targets (similar concept)
        hvt_targets = player_targets[
            (player_targets.get('yardline_100', 100) <= 20) |  # Red zone
            (player_targets.get('down', 1).isin([3, 4])) |  # 3rd/4th down
            (player_targets.get('is_two_minute', 0) == 1)  # 2-minute drill  
        ]
        hvt_target_count = len(hvt_targets)
        
        # Calculate team context metrics
        team_rushes = rb_rushes[
            (rb_rushes['season'] == season) & 
            (rb_rushes['posteam'] == team)
        ]
        team_targets = rb_targets[
            (rb_targets['season'] == season) & 
            (rb_targets['posteam'] == team)
        ]
        
        # Market share
        carry_share = n_carries / len(team_rushes) if len(team_rushes) > 0 else 0
        target_share = n_targets / len(team_targets) if len(team_targets) > 0 else 0
        
        # Opportunity share (carries + targets)
        opportunities = n_carries + n_targets
        team_opportunities = len(team_rushes) + len(team_targets)
        opportunity_share = opportunities / team_opportunities if team_opportunities > 0 else 0
        
        rb_metrics.append({
            'season': season,
            'player_name': player_name,
            'position': 'RB',
            'team': team,
            'carries': n_carries,
            'rush_yards': rush_yards,
            'rush_tds': rush_tds,
            'rush_epa': rush_epa,
            'rush_success_rate': rush_success,
            'targets': n_targets,
            'catches': catches,
            'catch_rate': catch_rate,
            'rec_yards': rec_yards,
            'rec_tds': rec_tds,
            'rec_epa': rec_epa,
            'opportunities': opportunities,
            'hvt_carries': hvt_carry_count,
            'hvt_carry_rate': hvt_carry_rate,
            'hvt_targets': hvt_target_count,
            'carry_share': carry_share,
            'target_share': target_share,
            'opportunity_share': opportunity_share
        })
    
    if not rb_metrics:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(rb_metrics)
    logger.info(f"Calculated RB metrics for {len(result_df)} player-seasons")
    
    return result_df


def calculate_wr_te_metrics(pbp_df: pd.DataFrame, weekly_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate WR/TE metrics including WOPR, RACR, and XYAC.
    
    Args:
        pbp_df: Play-by-play DataFrame  
        weekly_df: Optional weekly stats DataFrame
        
    Returns:
        DataFrame with WR/TE metrics by player-season
    """
    logger.info("Calculating WR/TE metrics")
    
    # Get all passing plays with receivers
    pass_plays = pbp_df[
        (pbp_df.get('is_dropback', 0) == 1) &
        pbp_df['receiver_player_name'].notna()
    ].copy()
    
    if pass_plays.empty:
        return pd.DataFrame()
    
    # Use weekly data to identify WR/TE if available
    wr_te_names = set()
    if weekly_df is not None:
        wr_te_weekly = weekly_df[weekly_df.get('position', '').isin(['WR', 'TE'])]
        for _, player in wr_te_weekly.iterrows():
            name = player.get('player_display_name') or player.get('player_name', '')
            season = player.get('season')
            if name and season:
                wr_te_names.add((season, name))
    
    # If no weekly data, identify by target volume (50+ targets likely WR/TE)  
    if not wr_te_names:
        target_counts = pass_plays.groupby(['season', 'receiver_player_name']).size()
        wr_te_names = set(target_counts[target_counts >= 20].index.tolist())
    
    wr_te_metrics = []
    
    for season, player_name in wr_te_names:
        player_targets = pass_plays[
            (pass_plays['season'] == season) &
            (pass_plays['receiver_player_name'] == player_name)
        ]
        
        if len(player_targets) < 10:  # Minimum sample
            continue
        
        team = player_targets['posteam'].mode().iloc[0] if len(player_targets) > 0 else ''
        
        # Basic receiving metrics
        n_targets = len(player_targets)
        catches = (player_targets.get('complete_pass', 0) == 1).sum()
        catch_rate = catches / n_targets if n_targets > 0 else 0
        
        rec_yards = player_targets[player_targets.get('complete_pass', 0) == 1].get('yards_gained', []).sum()
        rec_tds = (player_targets.get('touchdown', 0) == 1).sum()
        
        # EPA metrics
        target_epa = player_targets.get('epa', []).mean()
        rec_success_rate = (player_targets.get('epa_success', 0) == 1).mean()
        
        # Air yards metrics
        air_yards = player_targets.get('air_yards', []).dropna()
        total_air_yards = air_yards.sum() if len(air_yards) > 0 else 0
        avg_air_yards = air_yards.mean() if len(air_yards) > 0 else 0
        
        # YAC metrics
        completed_passes = player_targets[player_targets.get('complete_pass', 0) == 1]
        yac_values = completed_passes.get('yards_after_catch', []).dropna()
        total_yac = yac_values.sum() if len(yac_values) > 0 else 0
        avg_yac = yac_values.mean() if len(yac_values) > 0 else 0
        
        # XYAC (expected YAC) - if available
        xyac_values = completed_passes.get('xyac_mean_yardage', []).dropna()
        if len(xyac_values) > 0:
            expected_yac = xyac_values.sum()
            yac_over_expected = total_yac - expected_yac
        else:
            yac_over_expected = 0
        
        # Team context for market share calculations
        team_pass_plays = pass_plays[
            (pass_plays['season'] == season) &
            (pass_plays['posteam'] == team)
        ]
        
        team_targets = len(team_pass_plays)
        team_air_yards = team_pass_plays.get('air_yards', []).sum()
        team_rec_yards = team_pass_plays[team_pass_plays.get('complete_pass', 0) == 1].get('yards_gained', []).sum()
        
        # WOPR (Weighted Opportunity Rating) = 1.5 * Target Share + 0.7 * Air Yard Share  
        target_share = n_targets / team_targets if team_targets > 0 else 0
        air_yard_share = total_air_yards / team_air_yards if team_air_yards > 0 else 0
        wopr = (1.5 * target_share) + (0.7 * air_yard_share)
        
        # RACR (Receiver Air Conversion Ratio) = Receiving Yards / Air Yards
        racr = rec_yards / total_air_yards if total_air_yards > 0 else 0
        
        # Target quality metrics
        rz_targets = player_targets[player_targets.get('is_red_zone', 0) == 1]
        rz_target_share = len(rz_targets) / n_targets if n_targets > 0 else 0
        
        deep_targets = player_targets[player_targets.get('air_yards', 0) >= 20]
        deep_target_rate = len(deep_targets) / n_targets if n_targets > 0 else 0
        
        # Situational usage
        third_down_targets = player_targets[player_targets.get('down', 1) == 3]
        third_down_target_rate = len(third_down_targets) / n_targets if n_targets > 0 else 0
        
        # Determine position (rough heuristic)
        position = 'WR'  # Default
        if avg_air_yards < 8 and target_share > 0.15:  # Likely TE usage pattern
            position = 'TE'
        
        wr_te_metrics.append({
            'season': season,
            'player_name': player_name,
            'position': position,
            'team': team,
            'targets': n_targets,
            'catches': catches,
            'catch_rate': catch_rate,
            'rec_yards': rec_yards,
            'rec_tds': rec_tds,
            'target_epa': target_epa,
            'rec_success_rate': rec_success_rate,
            'total_air_yards': total_air_yards,
            'avg_air_yards': avg_air_yards,
            'total_yac': total_yac,
            'avg_yac': avg_yac,
            'yac_over_expected': yac_over_expected,
            'target_share': target_share,
            'air_yard_share': air_yard_share,
            'wopr': wopr,
            'racr': racr,
            'rz_target_share': rz_target_share,
            'deep_target_rate': deep_target_rate,
            'third_down_target_rate': third_down_target_rate
        })
    
    if not wr_te_metrics:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(wr_te_metrics)
    logger.info(f"Calculated WR/TE metrics for {len(result_df)} player-seasons")
    
    return result_df


def combine_player_metrics(qb_df: pd.DataFrame, rb_df: pd.DataFrame, 
                          wr_te_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all player metrics into single DataFrame.
    
    Args:
        qb_df: QB metrics DataFrame
        rb_df: RB metrics DataFrame  
        wr_te_df: WR/TE metrics DataFrame
        
    Returns:
        Combined player metrics DataFrame
    """
    logger.info("Combining player metrics")
    
    dfs = [df for df in [qb_df, rb_df, wr_te_df] if not df.empty]
    
    if not dfs:
        return pd.DataFrame()
    
    # Add common columns and fill missing with NaN
    common_cols = ['season', 'player_name', 'position', 'team']
    
    for df in dfs:
        for col in common_cols:
            if col not in df.columns:
                df[col] = np.nan
    
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    
    logger.info(f"Combined metrics for {len(combined)} player-seasons")
    
    return combined