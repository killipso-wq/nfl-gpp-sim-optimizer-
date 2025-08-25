"""
Build baseline priors from historical NFL data (2023-2024 seasons).

This script uses nfl_data_py to fetch historical weekly statistics and
creates team and player priors for simulation input.
"""
import argparse
import os
from typing import Dict, Tuple
import warnings

import pandas as pd
import numpy as np

# NOTE: nfl_data_py import would go here when package is available
# For MVP, we'll create mock data structure and note the dependency
# import nfl_data_py as nfl


def fetch_historical_data(start_year: int, end_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch historical weekly data from nfl_data_py.
    
    Args:
        start_year: Starting season year
        end_year: Ending season year
        
    Returns:
        Tuple of (weekly_stats_df, team_stats_df)
    """
    # NOTE: This would use nfl_data_py when available
    # For MVP, we'll create a mock structure showing the expected data format
    
    print("NOTE: This script requires nfl_data_py package for historical data.")
    print("Creating mock data structure for demonstration...")
    
    # Mock weekly player data structure
    weekly_data = []
    teams = ['BUF', 'MIA', 'NWE', 'NYJ', 'BAL', 'CIN', 'CLE', 'PIT', 'HOU', 'IND', 
             'JAC', 'TEN', 'DEN', 'KC', 'LV', 'LAC', 'DAL', 'NYG', 'PHI', 'WSH',
             'CHI', 'DET', 'GB', 'MIN', 'ATL', 'CAR', 'NOR', 'TAM', 'ARI', 'LAR', 'SF', 'SEA']
    
    # This would be replaced with actual nfl_data_py calls:
    # years = list(range(start_year, end_year + 1))
    # weekly_stats = nfl.import_weekly_data(years)
    # team_stats = nfl.import_team_stats(years)
    
    # Mock structure showing expected columns
    mock_weekly = pd.DataFrame({
        'player_id': ['BUF_QB_JOSHALLEN', 'BUF_RB_JAMESCOOK', 'BUF_WR_STEFANDIGGS'],
        'player_name': ['Josh Allen', 'James Cook', 'Stefon Diggs'],
        'position': ['QB', 'RB', 'WR'],
        'team': ['BUF', 'BUF', 'BUF'],
        'season': [2023, 2023, 2023],
        'week': [1, 1, 1],
        'fantasy_points': [24.5, 12.8, 18.2],
        'passing_yards': [283, 0, 0],
        'passing_tds': [2, 0, 0],
        'rushing_yards': [54, 78, 0],
        'rushing_tds': [1, 0, 0],
        'receiving_yards': [0, 15, 95],
        'receiving_tds': [0, 0, 1],
        'receptions': [0, 2, 8]
    })
    
    mock_team = pd.DataFrame({
        'team': teams[:5],
        'season': [2023] * 5,
        'avg_game_total': [45.2, 43.8, 41.5, 44.1, 46.8],
        'avg_points_scored': [24.1, 22.3, 18.5, 22.8, 25.2],
        'avg_points_allowed': [21.1, 21.5, 23.0, 21.3, 21.6]
    })
    
    return mock_weekly, mock_team


def calculate_player_priors(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate player priors from weekly statistics.
    
    Args:
        weekly_stats: Weekly player statistics
        
    Returns:
        DataFrame with player priors
    """
    # Group by player and calculate statistics
    player_priors = weekly_stats.groupby(['player_id', 'player_name', 'position', 'team']).agg({
        'fantasy_points': ['mean', 'std', 'count', 'min', 'max'],
        'passing_yards': 'mean',
        'passing_tds': 'mean',
        'rushing_yards': 'mean',
        'rushing_tds': 'mean',
        'receiving_yards': 'mean',
        'receiving_tds': 'mean',
        'receptions': 'mean'
    }).reset_index()
    
    # Flatten column names
    player_priors.columns = ['player_id', 'player_name', 'position', 'team',
                            'mean_fantasy_points', 'std_fantasy_points', 'games_played',
                            'min_fantasy_points', 'max_fantasy_points',
                            'avg_passing_yards', 'avg_passing_tds', 'avg_rushing_yards',
                            'avg_rushing_tds', 'avg_receiving_yards', 'avg_receiving_tds',
                            'avg_receptions']
    
    # Calculate variance for simulation
    player_priors['variance_fantasy_points'] = player_priors['std_fantasy_points'] ** 2
    
    # Fill NaN values for std/variance with position defaults
    position_defaults = {
        'QB': 6.0, 'RB': 5.5, 'WR': 6.5, 'TE': 4.5, 'DST': 7.0, 'K': 3.5
    }
    
    for position, default_std in position_defaults.items():
        mask = (player_priors['position'] == position) & player_priors['std_fantasy_points'].isna()
        player_priors.loc[mask, 'std_fantasy_points'] = default_std
        player_priors.loc[mask, 'variance_fantasy_points'] = default_std ** 2
    
    # Filter out players with insufficient data
    min_games = 4
    player_priors = player_priors[player_priors['games_played'] >= min_games]
    
    return player_priors


def calculate_team_priors(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate team priors from team statistics.
    
    Args:
        team_stats: Team statistics
        
    Returns:
        DataFrame with team priors
    """
    # For mock data, team_stats is already aggregated
    # In real implementation, this would aggregate from game-level data
    
    team_priors = team_stats.copy()
    
    # Add additional calculated fields
    team_priors['pace_factor'] = 1.0  # Placeholder for pace calculations
    team_priors['pass_rate'] = 0.6    # Placeholder for pass/run mix
    
    return team_priors


def save_priors(
    player_priors: pd.DataFrame,
    team_priors: pd.DataFrame,
    output_dir: str
) -> Dict[str, str]:
    """
    Save priors to CSV files.
    
    Args:
        player_priors: Player priors DataFrame
        team_priors: Team priors DataFrame
        output_dir: Output directory
        
    Returns:
        Dictionary of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create baseline subdirectory
    baseline_dir = os.path.join(output_dir, 'baseline')
    os.makedirs(baseline_dir, exist_ok=True)
    
    # Save files
    player_path = os.path.join(baseline_dir, 'player_priors.csv')
    team_path = os.path.join(baseline_dir, 'team_priors.csv')
    
    player_priors.to_csv(player_path, index=False)
    team_priors.to_csv(team_path, index=False)
    
    return {
        'player_priors': player_path,
        'team_priors': team_path
    }


def validate_priors(player_priors: pd.DataFrame, team_priors: pd.DataFrame) -> list:
    """
    Validate the generated priors and return any warnings.
    
    Args:
        player_priors: Player priors DataFrame
        team_priors: Team priors DataFrame
        
    Returns:
        List of warning messages
    """
    warnings_list = []
    
    # Check player priors
    if len(player_priors) == 0:
        warnings_list.append("No player priors generated")
    else:
        # Check for missing variance
        missing_variance = player_priors['variance_fantasy_points'].isna().sum()
        if missing_variance > 0:
            warnings_list.append(f"{missing_variance} players have missing variance")
        
        # Check for extreme values
        max_points = player_priors['mean_fantasy_points'].max()
        if max_points > 30:
            warnings_list.append(f"Unusually high mean fantasy points detected: {max_points:.1f}")
        
        min_points = player_priors['mean_fantasy_points'].min()
        if min_points < 0:
            warnings_list.append(f"Negative mean fantasy points detected: {min_points:.1f}")
    
    # Check team priors
    if len(team_priors) == 0:
        warnings_list.append("No team priors generated")
    else:
        # Check game totals
        if 'avg_game_total' in team_priors.columns:
            max_total = team_priors['avg_game_total'].max()
            min_total = team_priors['avg_game_total'].min()
            
            if max_total > 60 or min_total < 30:
                warnings_list.append(f"Unusual game totals: {min_total:.1f} - {max_total:.1f}")
    
    return warnings_list


def print_summary(player_priors: pd.DataFrame, team_priors: pd.DataFrame):
    """Print summary statistics."""
    print(f"\nSummary:")
    print(f"Player priors: {len(player_priors)} players")
    
    if len(player_priors) > 0:
        by_position = player_priors['position'].value_counts()
        for pos, count in by_position.items():
            avg_points = player_priors[player_priors['position'] == pos]['mean_fantasy_points'].mean()
            print(f"  {pos}: {count} players (avg {avg_points:.1f} pts)")
    
    print(f"Team priors: {len(team_priors)} teams")
    
    if len(team_priors) > 0 and 'avg_game_total' in team_priors.columns:
        avg_total = team_priors['avg_game_total'].mean()
        print(f"  Average game total: {avg_total:.1f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Build baseline priors from historical data')
    parser.add_argument('--start', type=int, required=True, help='Start year (e.g., 2023)')
    parser.add_argument('--end', type=int, required=True, help='End year (e.g., 2024)')
    parser.add_argument('--out', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Building priors for {args.start}-{args.end} seasons...")
    
    # Fetch historical data
    try:
        weekly_stats, team_stats = fetch_historical_data(args.start, args.end)
        print(f"Fetched data: {len(weekly_stats)} weekly records, {len(team_stats)} team records")
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return 1
    
    # Calculate priors
    try:
        print("Calculating player priors...")
        player_priors = calculate_player_priors(weekly_stats)
        
        print("Calculating team priors...")
        team_priors = calculate_team_priors(team_stats)
        
    except Exception as e:
        print(f"Error calculating priors: {e}")
        return 1
    
    # Validate priors
    warnings_list = validate_priors(player_priors, team_priors)
    if warnings_list:
        print("Warnings:")
        for warning in warnings_list:
            print(f"  - {warning}")
    
    # Save priors
    try:
        file_paths = save_priors(player_priors, team_priors, args.out)
        print(f"\nSaved priors:")
        for file_type, path in file_paths.items():
            print(f"  {file_type}: {path}")
    except Exception as e:
        print(f"Error saving priors: {e}")
        return 1
    
    # Print summary
    print_summary(player_priors, team_priors)
    
    print("\nNOTE: This script requires 'nfl_data_py' package for production use.")
    print("Install with: pip install nfl_data_py")
    print("Mock data was generated for demonstration purposes.")
    
    return 0


if __name__ == '__main__':
    exit(main())