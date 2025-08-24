"""Site player data ingestion and normalization."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import re


def load_site_players(file_path: str) -> pd.DataFrame:
    """Load and normalize site players CSV file.
    
    Expected columns (case insensitive):
    - PLAYER/Name: Player name
    - POS/Position: Position 
    - TEAM: Team abbreviation
    - OPP: Opponent team
    - O/U/Total: Over/under line
    - SPRD/Spread: Point spread  
    - SAL/Salary: DraftKings salary
    - RST%/Own%/Ownership: Projected ownership
    - FPTS/PROJ/POINTS: Site projection (optional)
    
    Args:
        file_path: Path to site players CSV
        
    Returns:
        Normalized DataFrame with standard column names
    """
    # Load the CSV
    df = pd.read_csv(file_path)
    
    # Normalize column names
    df = normalize_column_headers(df)
    
    # Validate required columns
    required_cols = ['name', 'position', 'team', 'opponent', 'total', 'spread', 'salary', 'ownership']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean and normalize data
    df = clean_player_data(df)
    
    # Add derived fields
    df = add_derived_fields(df)
    
    return df


def normalize_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column headers to standard names."""
    df = df.copy()
    
    # Column mapping (original -> standardized)
    column_mapping = {
        # Player name
        'PLAYER': 'name',
        'Name': 'name', 
        'player': 'name',
        'player_name': 'name',
        
        # Position
        'POS': 'position',
        'Position': 'position',
        'pos': 'position',
        
        # Team  
        'TEAM': 'team',
        'Team': 'team',
        'tm': 'team',
        
        # Opponent
        'OPP': 'opponent', 
        'Opponent': 'opponent',
        'opp': 'opponent',
        'vs': 'opponent',
        
        # Total (over/under)
        'O/U': 'total',
        'Total': 'total',
        'total_line': 'total',
        'ou': 'total',
        
        # Spread
        'SPRD': 'spread',
        'Spread': 'spread', 
        'spread_line': 'spread',
        'line': 'spread',
        
        # Salary
        'SAL': 'salary',
        'Salary': 'salary',
        'sal': 'salary',
        'dk_salary': 'salary',
        
        # Ownership
        'RST%': 'ownership',
        'Own%': 'ownership',
        'Ownership': 'ownership',
        'ownership_proj': 'ownership',
        'own': 'ownership',
        
        # Projections (optional)
        'FPTS': 'site_proj',
        'PROJ': 'site_proj',
        'POINTS': 'site_proj',
        'projection': 'site_proj',
        'proj': 'site_proj',
        'fppg': 'site_proj',
    }
    
    # Rename columns
    df.columns = [column_mapping.get(col, col.lower()) for col in df.columns]
    
    return df


def clean_player_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize player data."""
    df = df.copy()
    
    # Clean player names
    df['name'] = df['name'].apply(normalize_name)
    
    # Standardize positions
    df['position'] = df['position'].apply(normalize_position)
    
    # Standardize team abbreviations
    df['team'] = df['team'].apply(normalize_team)
    df['opponent'] = df['opponent'].apply(normalize_team)
    
    # Clean numeric columns
    numeric_cols = ['total', 'spread', 'salary', 'ownership']
    if 'site_proj' in df.columns:
        numeric_cols.append('site_proj')
        
    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    # Convert ownership percentage
    if df['ownership'].max() > 1:
        df['ownership'] = df['ownership'] / 100  # Convert percentage to decimal
    
    return df


def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived fields to the DataFrame."""
    df = df.copy()
    
    # Create unique player ID
    df['player_id'] = df.apply(lambda row: build_player_id(
        row['team'], row['position'], row['name']
    ), axis=1)
    
    # Create game ID
    df['game_id'] = df.apply(lambda row: f"{row['team']}_{row['opponent']}", axis=1)
    
    # Add favorite/underdog indicator based on spread
    df['is_favorite'] = df['spread'] < 0
    df['is_underdog'] = df['spread'] > 0
    
    # Add implied total points 
    df['implied_points'] = (df['total'] - df['spread']) / 2
    df['opp_implied_points'] = (df['total'] + df['spread']) / 2
    
    return df


def normalize_name(name: str) -> str:
    """Normalize player name."""
    if pd.isna(name):
        return ""
    
    name = str(name).strip()
    
    # Remove common suffixes
    name = re.sub(r'\s+(Jr\.?|Sr\.?|III|IV|V)$', '', name, flags=re.IGNORECASE)
    
    # Handle common abbreviations
    name = name.replace('.', '')
    
    # Standardize spacing
    name = re.sub(r'\s+', ' ', name)
    
    return name.title()


def normalize_position(position: str) -> str:
    """Normalize position abbreviation."""
    if pd.isna(position):
        return ""
    
    pos = str(position).upper().strip()
    
    # Handle common variations
    position_map = {
        'QUARTERBACK': 'QB',
        'RUNNINGBACK': 'RB', 
        'RUNNING BACK': 'RB',
        'WIDERECIVER': 'WR',
        'WIDE RECEIVER': 'WR',
        'TIGHTEND': 'TE',
        'TIGHT END': 'TE',
        'DEFENSE': 'DST',
        'DEFENSE/ST': 'DST',
        'DEF': 'DST',
        'D/ST': 'DST',
    }
    
    return position_map.get(pos, pos)


def normalize_team(team: str) -> str:
    """Normalize team abbreviation."""
    if pd.isna(team):
        return ""
    
    team = str(team).upper().strip()
    
    # Handle common variations
    team_map = {
        'ARZ': 'ARI',
        'JAC': 'JAX', 
        'LV': 'LVR',
        'RAIDERS': 'LVR',
        'RAMS': 'LAR',
        'CHARGERS': 'LAC',
    }
    
    return team_map.get(team, team)


def build_player_id(team: str, position: str, name: str) -> str:
    """Build standardized player ID."""
    # For DST, use team_DST format
    if position == 'DST':
        return f"{team}_DST"
    
    # For players, use team_pos_normalizedname format
    normalized_name = name.replace(' ', '_').replace('.', '').replace("'", '')
    return f"{team}_{position}_{normalized_name}"


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Clean numeric column by removing non-numeric characters."""
    def clean_value(val):
        if pd.isna(val):
            return np.nan
        
        # Convert to string and clean
        val_str = str(val).strip()
        
        # Remove common formatting
        val_str = val_str.replace('$', '').replace(',', '').replace('%', '')
        
        # Handle negative values in parentheses
        if val_str.startswith('(') and val_str.endswith(')'):
            val_str = '-' + val_str[1:-1]
        
        try:
            return float(val_str)
        except (ValueError, TypeError):
            return np.nan
    
    return series.apply(clean_value)


def validate_site_data(df: pd.DataFrame) -> List[str]:
    """Validate site data and return list of issues."""
    issues = []
    
    # Check for missing required data
    required_numeric = ['salary', 'total', 'spread']
    for col in required_numeric:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                issues.append(f"{null_count} missing values in {col}")
    
    # Check salary ranges
    if 'salary' in df.columns:
        min_sal = df['salary'].min()
        max_sal = df['salary'].max()
        if min_sal < 3000 or max_sal > 15000:
            issues.append(f"Unusual salary range: ${min_sal}-${max_sal}")
    
    # Check for duplicate players
    if len(df) != df['player_id'].nunique():
        issues.append("Duplicate players found")
    
    # Check position distribution
    pos_counts = df['position'].value_counts()
    if 'QB' in pos_counts and pos_counts['QB'] > 50:
        issues.append(f"Unusually high QB count: {pos_counts['QB']}")
    
    return issues