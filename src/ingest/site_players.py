"""
Robust loader for site player CSV files with flexible header mapping.

This module handles the complexities of parsing player CSV files with various
header formats and constructs consistent game IDs from team/opponent information.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Optional, Tuple, List
from .name_normalizer import normalize_player_name, normalize_position


class SitePlayerLoader:
    """
    Loads and normalizes player CSV files from DFS sites.
    
    Handles flexible column mapping and data type inference.
    """
    
    # Standard column mappings - handles various header formats
    COLUMN_MAPPINGS = {
        'player': ['PLAYER', 'Player', 'NAME', 'Name', 'player_name', 'PlayerName'],
        'position': ['POS', 'Position', 'Pos', 'position'],  
        'team': ['TEAM', 'Team', 'Tm', 'team'],
        'opponent': ['OPP', 'Opponent', 'Opp', 'VS', 'vs', 'opponent'],
        'over_under': ['O/U', 'OU', 'Over/Under', 'Total', 'total', 'over_under'],
        'spread': ['SPRD', 'Spread', 'Line', 'line', 'spread'],
        'salary': ['SAL', 'Salary', 'salary', 'SALARY', 'Price'],
        'fpts': ['FPTS', 'Proj', 'Projection', 'Points', 'fpts', 'projected_points'],
        'ownership': ['RST%', 'Own%', 'Ownership', 'ownership', 'rst_pct', 'own_pct'],
        'money_line': ['ML', 'MoneyLine', 'money_line'],
        'team_total': ['TM/P', 'Team Total', 'TT', 'team_total', 'team_implied'],
        'value': ['VAL', 'Value', 'value']
    }
    
    def __init__(self):
        self.column_map = {}
        self.df = None
        
    def load_csv(self, file_path_or_buffer, **kwargs) -> pd.DataFrame:
        """
        Load player CSV with automatic column detection and normalization.
        
        Args:
            file_path_or_buffer: Path to CSV file or file-like buffer
            **kwargs: Additional arguments passed to pd.read_csv
            
        Returns:
            Normalized DataFrame with standardized columns
        """
        # Read raw CSV
        self.df = pd.read_csv(file_path_or_buffer, **kwargs)
        
        # Map columns to standard names
        self._map_columns()
        
        # Clean and normalize data
        self._clean_data()
        
        # Generate game IDs
        self._generate_game_ids()
        
        return self.df
    
    def _map_columns(self):
        """Map CSV columns to standardized names."""
        original_cols = list(self.df.columns)
        
        for standard_col, possible_names in self.COLUMN_MAPPINGS.items():
            for col in original_cols:
                if col in possible_names:
                    self.column_map[standard_col] = col
                    break
        
        # Rename columns in DataFrame
        rename_dict = {v: k for k, v in self.column_map.items()}
        self.df = self.df.rename(columns=rename_dict)
        
        # Add missing standard columns with defaults
        required_cols = ['player', 'position', 'team']
        for col in required_cols:
            if col not in self.df.columns:
                self.df[col] = ''
                
        optional_cols = ['opponent', 'over_under', 'spread', 'salary', 'fpts', 
                        'ownership', 'money_line', 'team_total', 'value']
        for col in optional_cols:
            if col not in self.df.columns:
                self.df[col] = np.nan
    
    def _clean_data(self):
        """Clean and normalize player data."""
        # Normalize player names
        self.df['player_normalized'] = self.df['player'].apply(normalize_player_name)
        
        # Normalize positions (handle D/DEF -> DST mapping)
        self.df['position'] = self.df['position'].apply(normalize_position)
        
        # Clean team names (uppercase, remove spaces)
        self.df['team'] = self.df['team'].astype(str).str.upper().str.strip()
        
        # Parse numeric columns
        self._parse_numeric_columns()
        
        # Clean opponent formatting
        self._clean_opponent()
    
    def _parse_numeric_columns(self):
        """Parse numeric columns handling various formats."""
        numeric_cols = ['over_under', 'spread', 'salary', 'fpts', 'money_line', 'team_total']
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = self._parse_numeric_column(self.df[col])
        
        # Handle ownership percentage (could be "5%" or "5" or "0.05")
        if 'ownership' in self.df.columns:
            self.df['ownership'] = self._parse_ownership_column(self.df['ownership'])
    
    def _parse_numeric_column(self, series: pd.Series) -> pd.Series:
        """Parse a numeric column handling various string formats."""
        def parse_value(val):
            if pd.isna(val) or val == '':
                return np.nan
            
            # Convert to string and clean
            val_str = str(val).strip()
            
            # Remove currency symbols, commas, parentheses
            val_str = re.sub(r'[\$,\(\)]', '', val_str)
            
            # Handle negative values in parentheses like (3.5)
            if val_str.startswith('(') and val_str.endswith(')'):
                val_str = '-' + val_str[1:-1]
            
            try:
                return float(val_str)
            except (ValueError, TypeError):
                return np.nan
        
        return series.apply(parse_value)
    
    def _parse_ownership_column(self, series: pd.Series) -> pd.Series:
        """Parse ownership column handling percentage formats."""
        def parse_ownership(val):
            if pd.isna(val) or val == '':
                return np.nan
                
            val_str = str(val).strip()
            
            # Remove % symbol
            val_str = val_str.replace('%', '')
            
            try:
                pct = float(val_str)
                # If value > 1, assume it's already percentage (e.g., 15 = 15%)
                # If value <= 1, assume it's decimal (e.g., 0.15 = 15%)
                if pct <= 1.0 and pct >= 0:
                    return pct * 100
                elif pct > 1.0 and pct <= 100:
                    return pct
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
        
        return series.apply(parse_ownership)
    
    def _clean_opponent(self):
        """Clean opponent column and extract home/away information."""
        if 'opponent' not in self.df.columns:
            return
            
        def parse_opponent(opp_str):
            if pd.isna(opp_str) or opp_str == '':
                return '', ''
                
            opp_str = str(opp_str).strip().upper()
            
            # Handle @GB (away), vs GB (home), or just GB
            if opp_str.startswith('@'):
                return opp_str[1:].strip(), 'away'
            elif opp_str.startswith('VS'):
                return opp_str[2:].strip(), 'home'
            elif opp_str.startswith('V'):
                return opp_str[1:].strip(), 'home'
            else:
                # Just opponent name, can't determine home/away definitively
                return opp_str, 'unknown'
        
        opp_info = self.df['opponent'].apply(parse_opponent)
        self.df['opponent_clean'] = [info[0] for info in opp_info]
        self.df['home_away'] = [info[1] for info in opp_info]
    
    def _generate_game_ids(self):
        """Generate consistent game IDs from team/opponent information."""
        game_ids = []
        
        for _, row in self.df.iterrows():
            team = row.get('team', '')
            opp = row.get('opponent_clean', '')
            home_away = row.get('home_away', 'unknown')
            
            if not team or not opp:
                game_ids.append('')
                continue
            
            # Create consistent AWAY-HOME format
            if home_away == 'away':
                game_id = f"{team}-{opp}"
            elif home_away == 'home':  
                game_id = f"{opp}-{team}"
            else:
                # Unknown home/away, use alphabetical order for consistency
                teams_sorted = sorted([team, opp])
                game_id = f"{teams_sorted[0]}-{teams_sorted[1]}"
            
            game_ids.append(game_id)
        
        self.df['game_id'] = game_ids
    
    def get_column_mapping_summary(self) -> Dict[str, str]:
        """Get summary of how CSV columns were mapped."""
        return {k: v for k, v in self.column_map.items() if v}
    
    def validate_required_data(self) -> List[str]:
        """
        Validate that required data is present.
        
        Returns:
            List of validation errors/warnings
        """
        errors = []
        
        # Check for required columns
        if self.df['player'].isna().any():
            errors.append("Some rows missing player names")
            
        if self.df['position'].isna().any() or (self.df['position'] == '').any():
            errors.append("Some rows missing positions")
            
        if self.df['team'].isna().any() or (self.df['team'] == '').any():
            errors.append("Some rows missing team information")
        
        # Check for reasonable data ranges
        if 'salary' in self.df.columns:
            sal_col = self.df['salary'].dropna()
            if len(sal_col) > 0:
                if sal_col.max() > 20000 or sal_col.min() < 3000:
                    errors.append("Salary values outside expected range (3000-20000)")
        
        if 'ownership' in self.df.columns:
            own_col = self.df['ownership'].dropna()  
            if len(own_col) > 0:
                if own_col.max() > 100 or own_col.min() < 0:
                    errors.append("Ownership values outside expected range (0-100%)")
        
        return errors


def load_site_players_csv(file_path_or_buffer, **kwargs) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Convenience function to load site player CSV.
    
    Args:
        file_path_or_buffer: Path to CSV or file buffer
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        Tuple of (DataFrame, column_mapping, validation_errors)
    """
    loader = SitePlayerLoader()
    df = loader.load_csv(file_path_or_buffer, **kwargs)
    mapping = loader.get_column_mapping_summary()
    errors = loader.validate_required_data()
    
    return df, mapping, errors