"""
Name normalization utilities for consistent player identification.

This module handles the messy reality of player names across different data sources,
creating stable player IDs that can be matched across nfl_data_py and site uploads.
"""

import re
from typing import Dict, Optional
from slugify import slugify


def normalize_player_name(name: str) -> str:
    """
    Normalize player name to a consistent format.
    
    Args:
        name: Raw player name (e.g., "Josh Allen", "Josh Allen Jr.", "D'Andre Swift")
        
    Returns:
        Normalized name suitable for matching
    """
    if not name or pd.isna(name):
        return ""
    
    # Clean up common issues
    name = str(name).strip()
    
    # Remove suffixes like Jr., Sr., II, III, IV
    name = re.sub(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', '', name, flags=re.IGNORECASE)
    
    # Handle periods in names (like "D.K. Metcalf" -> "DK Metcalf")
    name = re.sub(r'\.', '', name)
    
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def create_player_id(name: str, team: str = "", pos: str = "") -> str:
    """
    Create a stable player ID from name, team, and position.
    
    Args:
        name: Player name
        team: Team abbreviation (optional)
        pos: Position (optional)
        
    Returns:
        Stable player ID string
    """
    normalized_name = normalize_player_name(name)
    
    # Use slugify to create a URL-safe ID
    base_id = slugify(normalized_name, separator='_')
    
    if team:
        base_id += f"_{team.upper()}"
    
    if pos and pos not in ['', 'NA']:
        base_id += f"_{pos.upper()}"
    
    return base_id


def fuzzy_match_names(name1: str, name2: str, threshold: float = 0.8) -> bool:
    """
    Check if two player names likely refer to the same player.
    
    Args:
        name1: First name to compare
        name2: Second name to compare  
        threshold: Similarity threshold (0-1)
        
    Returns:
        True if names likely match the same player
    """
    norm1 = normalize_player_name(name1).lower()
    norm2 = normalize_player_name(name2).lower()
    
    if norm1 == norm2:
        return True
    
    # Simple fuzzy matching - check if one name contains the other
    # or if they share most words
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    
    if words1 and words2:
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap >= threshold
    
    return False


class PlayerNameMapper:
    """
    Maintains mappings between different name formats and canonical player IDs.
    """
    
    def __init__(self):
        self.name_to_id: Dict[str, str] = {}
        self.id_to_canonical: Dict[str, Dict[str, str]] = {}
    
    def add_player(self, name: str, team: str = "", pos: str = "", 
                   canonical_name: Optional[str] = None) -> str:
        """
        Add a player to the mapper and return their stable ID.
        
        Args:
            name: Player name as it appears in data
            team: Team abbreviation
            pos: Position
            canonical_name: Preferred canonical name (defaults to normalized input name)
            
        Returns:
            Stable player ID
        """
        player_id = create_player_id(name, team, pos)
        canonical = canonical_name or normalize_player_name(name)
        
        self.name_to_id[name.lower()] = player_id
        
        if player_id not in self.id_to_canonical:
            self.id_to_canonical[player_id] = {
                'name': canonical,
                'team': team.upper() if team else '',
                'pos': pos.upper() if pos else ''
            }
        
        return player_id
    
    def get_player_id(self, name: str) -> Optional[str]:
        """
        Get the stable player ID for a given name.
        
        Args:
            name: Player name to lookup
            
        Returns:
            Player ID if found, None otherwise
        """
        return self.name_to_id.get(name.lower())
    
    def get_canonical_info(self, player_id: str) -> Optional[Dict[str, str]]:
        """
        Get canonical player information by ID.
        
        Args:
            player_id: Stable player ID
            
        Returns:
            Dict with canonical name, team, pos if found
        """
        return self.id_to_canonical.get(player_id)


# Global mapper instance
_global_mapper = PlayerNameMapper()


def get_global_mapper() -> PlayerNameMapper:
    """Get the global player name mapper instance."""
    return _global_mapper


# Common position mappings
POSITION_MAPPINGS = {
    'D': 'DST',
    'DEF': 'DST',
    'DEFENSE': 'DST',
    'K': 'K',
    'PK': 'K',
    'KICKER': 'K',
}


def normalize_position(pos: str) -> str:
    """
    Normalize position to standard format.
    
    Args:
        pos: Raw position string
        
    Returns:
        Normalized position
    """
    if not pos:
        return ""
    
    pos = str(pos).strip().upper()
    return POSITION_MAPPINGS.get(pos, pos)


# Need to import pandas for isna check
import pandas as pd