"""
Robust loader for site players.csv files with column mapping and validation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

from .name_normalizer import build_player_id, normalize_name, normalize_position, normalize_team


def load_site_players(filepath: str, validate: bool = True) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Load and normalize site players CSV file.
    
    Args:
        filepath: Path to players CSV file
        validate: Whether to perform validation checks
        
    Returns:
        Tuple of (normalized_df, column_mapping, warnings_list)
    """
    # Read the raw CSV
    raw_df = pd.read_csv(filepath)
    
    # Create column mapping
    column_mapping = create_column_mapping(raw_df.columns.tolist())
    
    # Apply column mapping
    df = apply_column_mapping(raw_df, column_mapping)
    
    # Normalize data
    df = normalize_site_data(df)
    
    # Validate data
    warnings_list = []
    if validate:
        warnings_list = validate_site_data(df)
    
    return df, column_mapping, warnings_list


def create_column_mapping(columns: List[str]) -> Dict[str, str]:
    """
    Create mapping from site column names to canonical names.
    
    Args:
        columns: List of column names from site CSV
        
    Returns:
        Dictionary mapping site columns to canonical columns
    """
    canonical_columns = {
        'PLAYER': ['PLAYER', 'Player', 'Name', 'PLAYER NAME', 'PlayerName'],
        'POS': ['POS', 'Position', 'Pos', 'POSITION'],
        'TEAM': ['TEAM', 'Team', 'Tm', 'TEAM NAME', 'TeamName'],
        'OPP': ['OPP', 'Opponent', 'Opp', 'OPPONENT', 'VS'],
        'SAL': ['SAL', 'Salary', 'Sal', 'DK Salary', 'DKSalary', 'SALARY'],
        'FPTS': ['FPTS', 'Projection', 'Proj', 'DK Points', 'DKPoints', 'PROJECTION', 'POINTS'],
        'RST%': ['RST%', 'Ownership', 'Own', 'Roster%', 'OWNERSHIP', 'ROSTER%'],
        'O/U': ['O/U', 'Over/Under', 'Total', 'TOTAL', 'OverUnder'],
        'SPRD': ['SPRD', 'Spread', 'Line', 'SPREAD', 'SPRD'],
        'ML': ['ML', 'Moneyline', 'MoneyLine', 'MONEYLINE'],
        'TM/P': ['TM/P', 'Team Total', 'TeamTotal', 'TEAM TOTAL', 'IMPLIED'],
        'VAL': ['VAL', 'Value', 'VALUE', 'Pts/$', 'Points per Dollar']
    }
    
    mapping = {}
    
    for canonical, variations in canonical_columns.items():
        for col in columns:
            if col in variations:
                mapping[col] = canonical
                break
    
    return mapping


def apply_column_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Apply column mapping to rename columns to canonical names.
    
    Args:
        df: Original DataFrame
        mapping: Column mapping dictionary
        
    Returns:
        DataFrame with renamed columns
    """
    result = df.copy()
    
    # Rename mapped columns
    result = result.rename(columns=mapping)
    
    # Ensure required columns exist (fill with NaN if missing)
    required_columns = ['PLAYER', 'POS', 'TEAM', 'OPP', 'SAL', 'FPTS']
    
    for col in required_columns:
        if col not in result.columns:
            result[col] = np.nan
    
    return result


def normalize_site_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data types and values in site players DataFrame.
    
    Args:
        df: Site players DataFrame
        
    Returns:
        Normalized DataFrame
    """
    result = df.copy()
    
    # Normalize player names
    if 'PLAYER' in result.columns:
        result['PLAYER'] = result['PLAYER'].apply(lambda x: normalize_name(str(x)) if pd.notna(x) else '')
    
    # Normalize positions
    if 'POS' in result.columns:
        result['POS'] = result['POS'].apply(lambda x: normalize_position(str(x)) if pd.notna(x) else '')
    
    # Normalize teams
    if 'TEAM' in result.columns:
        result['TEAM'] = result['TEAM'].apply(lambda x: normalize_team(str(x)) if pd.notna(x) else '')
    
    if 'OPP' in result.columns:
        result['OPP'] = result['OPP'].apply(lambda x: normalize_team(str(x)) if pd.notna(x) else '')
    
    # Convert numeric columns
    numeric_columns = ['SAL', 'FPTS', 'O/U', 'SPRD', 'ML', 'TM/P', 'VAL']
    for col in numeric_columns:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce')
    
    # Normalize ownership percentage
    if 'RST%' in result.columns:
        result['RST%'] = normalize_ownership(result['RST%'])
    
    # Create player_id
    if all(col in result.columns for col in ['PLAYER', 'TEAM', 'POS']):
        result['player_id'] = result.apply(
            lambda row: build_player_id(row['PLAYER'], row['TEAM'], row['POS']), 
            axis=1
        )
    
    # Create game_id
    if all(col in result.columns for col in ['TEAM', 'OPP']):
        result['game_id'] = result.apply(
            lambda row: create_game_id(row['TEAM'], row['OPP']),
            axis=1
        )
    
    return result


def normalize_ownership(ownership_series: pd.Series) -> pd.Series:
    """
    Normalize ownership percentage values.
    
    If values are <= 1, treat as fraction and multiply by 100.
    Otherwise, treat as percentage.
    
    Args:
        ownership_series: Series with ownership values
        
    Returns:
        Normalized ownership percentages
    """
    result = pd.to_numeric(ownership_series, errors='coerce')
    
    # If most values are <= 1, assume they're fractions
    valid_values = result.dropna()
    if len(valid_values) > 0 and (valid_values <= 1).mean() > 0.8:
        result = result * 100
    
    return result


def create_game_id(team: str, opponent: str) -> str:
    """
    Create consistent game identifier.
    
    Args:
        team: Team abbreviation
        opponent: Opponent abbreviation
        
    Returns:
        Game identifier
    """
    if pd.isna(team) or pd.isna(opponent):
        return ''
    
    # Sort teams alphabetically for consistency
    teams = sorted([str(team), str(opponent)])
    return f"{teams[0]}_vs_{teams[1]}"


def validate_site_data(df: pd.DataFrame) -> List[str]:
    """
    Validate site players data and return list of warnings.
    
    Args:
        df: Site players DataFrame
        
    Returns:
        List of warning messages
    """
    warnings_list = []
    
    # Check for required columns
    required_columns = ['PLAYER', 'POS', 'TEAM']
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        warnings_list.append(f"Missing required columns: {missing_required}")
    
    # Check for empty player names
    if 'PLAYER' in df.columns:
        empty_names = df['PLAYER'].isna().sum() + (df['PLAYER'] == '').sum()
        if empty_names > 0:
            warnings_list.append(f"{empty_names} players have empty names")
    
    # Check for unknown positions
    if 'POS' in df.columns:
        valid_positions = {'QB', 'RB', 'WR', 'TE', 'DST', 'K'}
        unknown_positions = set(df['POS'].dropna()) - valid_positions
        if unknown_positions:
            warnings_list.append(f"Unknown positions found: {unknown_positions}")
    
    # Check salary range
    if 'SAL' in df.columns:
        valid_salaries = df['SAL'].dropna()
        if len(valid_salaries) > 0:
            if valid_salaries.min() < 3000 or valid_salaries.max() > 15000:
                warnings_list.append("Salary values outside expected range (3000-15000)")
    
    # Check projection range
    if 'FPTS' in df.columns:
        valid_projections = df['FPTS'].dropna()
        if len(valid_projections) > 0:
            if valid_projections.min() < 0 or valid_projections.max() > 50:
                warnings_list.append("Projection values outside expected range (0-50)")
    
    # Check ownership range
    if 'RST%' in df.columns:
        valid_ownership = df['RST%'].dropna()
        if len(valid_ownership) > 0:
            if valid_ownership.min() < 0 or valid_ownership.max() > 100:
                warnings_list.append("Ownership values outside expected range (0-100%)")
    
    # Check for duplicate player_ids
    if 'player_id' in df.columns:
        duplicates = df['player_id'].duplicated().sum()
        if duplicates > 0:
            warnings_list.append(f"{duplicates} duplicate player IDs found")
    
    return warnings_list


def get_column_mapping_display(mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Create a display-friendly DataFrame showing column mappings.
    
    Args:
        mapping: Column mapping dictionary
        
    Returns:
        DataFrame for display in UI
    """
    if not mapping:
        return pd.DataFrame(columns=['Site Column', 'Mapped To', 'Status'])
    
    data = []
    for site_col, canonical_col in mapping.items():
        data.append({
            'Site Column': site_col,
            'Mapped To': canonical_col,
            'Status': '✓ Mapped'
        })
    
    return pd.DataFrame(data)


def get_required_columns() -> List[str]:
    """
    Get list of required columns for site players data.
    
    Returns:
        List of required column names
    """
    return ['PLAYER', 'POS', 'TEAM', 'OPP', 'SAL', 'FPTS']


def get_optional_columns() -> List[str]:
    """
    Get list of optional but recommended columns.
    
    Returns:
        List of optional column names
    """
    return ['RST%', 'O/U', 'SPRD', 'ML', 'TM/P', 'VAL']