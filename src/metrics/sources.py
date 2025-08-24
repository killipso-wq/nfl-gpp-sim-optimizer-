"""Data sources for NFL data retrieval."""

import pandas as pd
import numpy as np
from typing import List, Optional
import warnings

# Mock implementation for now - will be replaced with actual nfl_data_py calls
# when the dependency installation issue is resolved

def load_pbp_data(seasons: List[int]) -> pd.DataFrame:
    """Load play-by-play data for specified seasons.
    
    Args:
        seasons: List of seasons to load (e.g., [2023, 2024])
        
    Returns:
        DataFrame with play-by-play data
    """
    warnings.warn("Using mock data - nfl_data_py not available", UserWarning)
    
    # Create mock play-by-play data structure
    np.random.seed(42)  # For reproducible mock data
    
    mock_data = {
        'game_id': [f'2023_01_{i:02d}_{["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE"][i%8]}_{["DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC"][i%8]}'
                   for i in range(1000)],
        'season': np.random.choice(seasons, 1000),
        'week': np.random.randint(1, 19, 1000),
        'posteam': np.random.choice(['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE'], 1000),
        'defteam': np.random.choice(['DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC'], 1000),
        'down': np.random.randint(1, 5, 1000),
        'ydstogo': np.random.randint(1, 21, 1000),
        'yardline_100': np.random.randint(1, 101, 1000),
        'score_differential': np.random.randint(-21, 22, 1000),
        'win_prob': np.random.uniform(0.1, 0.9, 1000),
        'play_type': np.random.choice(['pass', 'run', 'punt', 'field_goal'], 1000, p=[0.6, 0.3, 0.05, 0.05]),
        'epa': np.random.normal(0, 2.5, 1000),
        'success': np.random.choice([0, 1], 1000, p=[0.6, 0.4]),
        'passer_player_id': np.random.choice(['QB1', 'QB2', 'QB3'], 1000),
        'receiver_player_id': np.random.choice(['WR1', 'WR2', 'WR3', 'TE1'], 1000),
        'rusher_player_id': np.random.choice(['RB1', 'RB2', 'QB1'], 1000),
        'air_yards': np.random.exponential(8, 1000),
        'yards_after_catch': np.random.exponential(4, 1000),
        'complete_pass': np.random.choice([0, 1], 1000, p=[0.35, 0.65]),
    }
    
    return pd.DataFrame(mock_data)


def load_weekly_data(seasons: List[int]) -> pd.DataFrame:
    """Load weekly player stats for specified seasons.
    
    Args:
        seasons: List of seasons to load
        
    Returns:
        DataFrame with weekly player statistics
    """
    warnings.warn("Using mock data - nfl_data_py not available", UserWarning)
    
    np.random.seed(43)
    
    players = ['Josh Allen', 'Lamar Jackson', 'Dak Prescott', 'Tyreek Hill', 'Stefon Diggs', 
               'Derrick Henry', 'Christian McCaffrey', 'Travis Kelce', 'Mark Andrews']
    positions = ['QB', 'QB', 'QB', 'WR', 'WR', 'RB', 'RB', 'TE', 'TE']
    teams = ['BUF', 'BAL', 'DAL', 'MIA', 'BUF', 'BAL', 'SF', 'KC', 'BAL']
    
    mock_data = []
    for season in seasons:
        for week in range(1, 19):
            for i, player in enumerate(players):
                mock_data.append({
                    'season': season,
                    'week': week,
                    'player_id': f'{teams[i]}_{positions[i]}_{player.replace(" ", "_")}',
                    'player_name': player,
                    'position': positions[i],
                    'team': teams[i],
                    'targets': np.random.poisson(8) if positions[i] in ['WR', 'TE'] else 0,
                    'receptions': np.random.poisson(5) if positions[i] in ['WR', 'TE'] else 0,
                    'receiving_yards': np.random.exponential(60) if positions[i] in ['WR', 'TE'] else 0,
                    'receiving_tds': np.random.poisson(0.5) if positions[i] in ['WR', 'TE'] else 0,
                    'carries': np.random.poisson(15) if positions[i] == 'RB' else (np.random.poisson(4) if positions[i] == 'QB' else 0),
                    'rushing_yards': np.random.exponential(70) if positions[i] == 'RB' else (np.random.exponential(20) if positions[i] == 'QB' else 0),
                    'rushing_tds': np.random.poisson(0.6) if positions[i] == 'RB' else (np.random.poisson(0.3) if positions[i] == 'QB' else 0),
                    'completions': np.random.poisson(20) if positions[i] == 'QB' else 0,
                    'attempts': np.random.poisson(30) if positions[i] == 'QB' else 0,
                    'passing_yards': np.random.exponential(250) if positions[i] == 'QB' else 0,
                    'passing_tds': np.random.poisson(1.5) if positions[i] == 'QB' else 0,
                    'interceptions': np.random.poisson(0.8) if positions[i] == 'QB' else 0,
                })
    
    return pd.DataFrame(mock_data)


def load_schedule_data(seasons: List[int]) -> pd.DataFrame:
    """Load schedule data for specified seasons.
    
    Args:
        seasons: List of seasons to load
        
    Returns:
        DataFrame with game schedule information
    """
    warnings.warn("Using mock data - nfl_data_py not available", UserWarning)
    
    teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
             'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
             'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
             'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS']
    
    mock_data = []
    for season in seasons:
        for week in range(1, 19):
            # Create some games for each week
            shuffled_teams = teams.copy()
            np.random.shuffle(shuffled_teams)
            
            for i in range(0, len(shuffled_teams)-1, 2):
                home_team = shuffled_teams[i]
                away_team = shuffled_teams[i+1]
                
                mock_data.append({
                    'game_id': f'{season}_{week:02d}_{home_team}_{away_team}',
                    'season': season,
                    'week': week,
                    'home_team': home_team,
                    'away_team': away_team,
                    'total_line': np.random.uniform(37, 58),  # Over/under
                    'spread_line': np.random.uniform(-14, 14),  # Point spread
                    'home_score': np.random.randint(10, 35),
                    'away_score': np.random.randint(10, 35),
                })
    
    return pd.DataFrame(mock_data)