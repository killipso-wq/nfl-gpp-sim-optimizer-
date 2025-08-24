"""Main pipeline for processing NFL data and computing metrics."""

import pandas as pd
from typing import List, Tuple
import warnings

from . import sources, prep, team_metrics, player_metrics


def run_baseline_pipeline(start_year: int, end_year: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete baseline data pipeline.
    
    Args:
        start_year: Starting season year
        end_year: Ending season year
        
    Returns:
        Tuple of (team_priors_df, player_priors_df, dst_priors_df)
    """
    print(f"Loading NFL data for seasons {start_year}-{end_year}...")
    
    # Load raw data
    seasons = list(range(start_year, end_year + 1))
    pbp_df = sources.load_pbp_data(seasons)
    weekly_df = sources.load_weekly_data(seasons)
    schedule_df = sources.load_schedule_data(seasons)
    
    print(f"Loaded {len(pbp_df)} plays, {len(weekly_df)} player-weeks, {len(schedule_df)} games")
    
    # Process play-by-play data
    print("Processing play-by-play data...")
    pbp_clean = prep.clean_pbp_data(pbp_df)
    pbp_with_features = prep.add_derived_features(pbp_clean)
    
    # Create neutral situation dataset
    pbp_neutral = prep.filter_neutral_situations(pbp_with_features)
    
    print(f"Filtered to {len(pbp_neutral)} neutral situation plays")
    
    # Calculate team metrics
    print("Calculating team metrics...")
    team_metrics_df = team_metrics.calculate_team_metrics(pbp_with_features, schedule_df)
    team_metrics_ranked = team_metrics.add_relative_rankings(team_metrics_df)
    
    # Calculate defensive metrics
    def_metrics_df = team_metrics.calculate_defensive_metrics(pbp_with_features)
    
    # Calculate player metrics
    print("Calculating player metrics...")
    player_metrics_df = player_metrics.calculate_player_metrics(weekly_df, pbp_with_features)
    
    # Calculate DST metrics
    dst_metrics_df = player_metrics.calculate_dst_metrics(weekly_df, def_metrics_df)
    
    print("Pipeline complete!")
    
    return team_metrics_ranked, player_metrics_df, dst_metrics_df


def create_team_priors(team_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Create team prior parameters for simulation.
    
    Args:
        team_metrics_df: Team metrics DataFrame
        
    Returns:
        DataFrame with team priors suitable for simulation
    """
    # Average metrics across seasons with some empirical Bayes shrinkage
    team_priors = []
    
    for team in team_metrics_df['team'].unique():
        team_data = team_metrics_df[team_metrics_df['team'] == team]
        
        if len(team_data) == 0:
            continue
            
        # Simple averaging with some shrinkage toward league mean
        shrinkage = 0.3  # 30% shrinkage toward league mean
        
        priors = {
            'team': team,
            'seasons_data': len(team_data),
        }
        
        # Key metrics for simulation
        metrics_to_average = [
            'plays_per_game', 'pass_rate', 'neutral_pass_rate', 
            'proe_neutral', 'epa_per_play', 'success_rate'
        ]
        
        for metric in metrics_to_average:
            if metric in team_data.columns:
                team_mean = team_data[metric].mean()
                league_mean = team_metrics_df[metric].mean()
                
                # Apply shrinkage
                prior_value = shrinkage * league_mean + (1 - shrinkage) * team_mean
                priors[f'{metric}_prior'] = prior_value
                priors[f'{metric}_std'] = team_data[metric].std()
        
        team_priors.append(priors)
    
    return pd.DataFrame(team_priors)


def create_player_priors(player_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Create player prior parameters for simulation.
    
    Args:
        player_metrics_df: Player metrics DataFrame
        
    Returns:
        DataFrame with player priors suitable for simulation
    """
    player_priors = []
    
    for _, player in player_metrics_df.iterrows():
        position = player['position']
        
        priors = {
            'player_id': player['player_id'],
            'player_name': player['player_name'],
            'position': position,
            'team': player['team'],
            'games_played': player['games_played'],
        }
        
        # Position-specific priors
        if position == 'QB':
            priors.update({
                'pass_attempts_prior': player.get('pass_attempts_per_game', 0),
                'completion_pct_prior': player.get('completion_pct', 0.65),
                'pass_ypa_prior': player.get('pass_yards_per_attempt', 7.0),
                'pass_td_rate_prior': player.get('pass_tds_per_game', 1.5) / max(player.get('pass_attempts_per_game', 30), 1),
                'int_rate_prior': player.get('int_rate', 0.03),
                'rush_attempts_prior': player.get('rush_attempts_per_game', 4),
                'rush_ypc_prior': player.get('rush_yards_per_carry', 4.5),
                'rush_td_rate_prior': player.get('rush_tds_per_game', 0.3) / max(player.get('rush_attempts_per_game', 4), 1),
            })
            
        elif position == 'RB':
            priors.update({
                'carry_share_prior': player.get('carry_share', 0.3),
                'rush_ypc_prior': player.get('rush_yards_per_carry', 4.2),
                'rush_td_rate_prior': player.get('rush_td_rate', 0.05),
                'target_share_prior': player.get('target_share', 0.08),
                'catch_rate_prior': player.get('catch_rate', 0.75),
                'yards_per_target_prior': player.get('yards_per_target', 6.0),
                'inside_10_carry_share_prior': player.get('inside_10_carries', 0.2),
            })
            
        elif position in ['WR', 'TE']:
            priors.update({
                'target_share_prior': player.get('target_share', 0.15),
                'air_yards_share_prior': player.get('air_yards_share', 0.15),
                'catch_rate_prior': player.get('catch_rate', 0.65),
                'yards_per_target_prior': player.get('yards_per_target', 8.0),
                'rec_td_rate_prior': player.get('rec_td_rate', 0.06),
                'wopr_prior': player.get('wopr', 0.25),
                'adot_prior': player.get('adot', 10.0),
                'red_zone_target_share_prior': player.get('red_zone_target_share', 0.15),
            })
            
        elif position == 'DST':
            priors.update({
                'sacks_per_game_prior': player.get('sacks_per_game', 2.5),
                'ints_per_game_prior': player.get('ints_per_game', 1.0),
                'fumbles_per_game_prior': player.get('fumbles_per_game', 0.7),
                'tds_per_game_prior': player.get('tds_per_game', 0.2),
                'points_allowed_prior': player.get('points_allowed_per_game', 22),
                'opp_sack_rate_prior': player.get('opp_sack_rate', 0.08),
                'opp_int_rate_prior': player.get('opp_int_rate', 0.03),
            })
        
        # Fantasy metrics
        priors.update({
            'fantasy_ppg_prior': player.get('fantasy_ppg', 8.0),
            'fantasy_floor_prior': player.get('fantasy_floor_p10', 2.0),
            'fantasy_ceil_prior': player.get('fantasy_ceil_p90', 15.0),
            'fantasy_std_prior': player.get('fantasy_std', 5.0),
        })
        
        player_priors.append(priors)
    
    return pd.DataFrame(player_priors)


def save_baseline_data(team_priors_df: pd.DataFrame, 
                      player_priors_df: pd.DataFrame,
                      output_dir: str) -> None:
    """Save baseline data to CSV files.
    
    Args:
        team_priors_df: Team priors DataFrame
        player_priors_df: Player priors DataFrame  
        output_dir: Output directory path
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    team_file = os.path.join(output_dir, 'team_priors.csv')
    player_file = os.path.join(output_dir, 'player_priors.csv')
    
    team_priors_df.to_csv(team_file, index=False)
    player_priors_df.to_csv(player_file, index=False)
    
    print(f"Saved team priors to {team_file}")
    print(f"Saved player priors to {player_file}")


def load_baseline_data(baseline_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load baseline data from CSV files.
    
    Args:
        baseline_dir: Directory containing baseline CSV files
        
    Returns:
        Tuple of (team_priors_df, player_priors_df)
    """
    import os
    
    team_file = os.path.join(baseline_dir, 'team_priors.csv')
    player_file = os.path.join(baseline_dir, 'player_priors.csv')
    
    if not os.path.exists(team_file) or not os.path.exists(player_file):
        raise FileNotFoundError(f"Baseline files not found in {baseline_dir}")
    
    team_priors_df = pd.read_csv(team_file)
    player_priors_df = pd.read_csv(player_file)
    
    return team_priors_df, player_priors_df