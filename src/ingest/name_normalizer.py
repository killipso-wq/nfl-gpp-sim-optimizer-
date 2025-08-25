"""
Name normalization utilities for NFL player data.
"""
import re
from typing import Dict


def normalize_name(name: str) -> str:
    """
    Normalize player name to standard format.
    
    Args:
        name: Raw player name
        
    Returns:
        Normalized name string
    """
    if not isinstance(name, str):
        return ""
    
    # Remove common suffixes
    name = re.sub(r'\s+(Jr\.?|Sr\.?|III?|IV)$', '', name, flags=re.IGNORECASE)
    
    # Remove periods and extra whitespace
    name = re.sub(r'\.', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Convert to title case
    name = name.title()
    
    # Handle common name patterns
    name = name.replace("'", "")  # Remove apostrophes
    
    return name


def normalize_position(pos: str) -> str:
    """
    Normalize position to standard format.
    
    Args:
        pos: Raw position string
        
    Returns:
        Normalized position (QB, RB, WR, TE, DST)
    """
    if not isinstance(pos, str):
        return ""
    
    pos = pos.upper().strip()
    
    # Handle defense variations
    if pos in ['D', 'DEF', 'DST', 'D/ST']:
        return 'DST'
    
    # Handle standard positions
    position_map = {
        'QB': 'QB',
        'RB': 'RB', 
        'WR': 'WR',
        'TE': 'TE',
        'K': 'K'
    }
    
    return position_map.get(pos, pos)


def normalize_team(team: str) -> str:
    """
    Normalize team abbreviation.
    
    Args:
        team: Raw team string
        
    Returns:
        Normalized 3-letter team abbreviation
    """
    if not isinstance(team, str):
        return ""
    
    team = team.upper().strip()
    
    # Handle common variations
    team_map = {
        'LV': 'LAS',  # Las Vegas Raiders
        'LA': 'LAR',  # Los Angeles Rams (default to Rams, could be Chargers)
        'JAX': 'JAC',  # Jacksonville Jaguars
        'NE': 'NWE',  # New England Patriots
        'NO': 'NOR',  # New Orleans Saints
        'SF': 'SFO',  # San Francisco 49ers
        'TB': 'TAM',  # Tampa Bay Buccaneers
        'WAS': 'WSH', # Washington
    }
    
    return team_map.get(team, team)


def build_player_id(player_name: str, team: str, position: str) -> str:
    """
    Build stable player_id in format TEAM_POS_NORMALIZEDNAME.
    
    Args:
        player_name: Player name
        team: Team abbreviation
        position: Position
        
    Returns:
        Stable player identifier
    """
    norm_name = normalize_name(player_name)
    norm_team = normalize_team(team)
    norm_pos = normalize_position(position)
    
    # Convert name to slug format
    name_slug = re.sub(r'[^a-zA-Z0-9]+', '', norm_name).upper()
    
    return f"{norm_team}_{norm_pos}_{name_slug}"


def create_name_mappings() -> Dict[str, str]:
    """
    Create common name variation mappings.
    
    Returns:
        Dictionary mapping variations to canonical names
    """
    return {
        # Common nickname variations
        'JOSH ALLEN': 'JOSHUA ALLEN',
        'MIKE EVANS': 'MICHAEL EVANS', 
        'DJ MOORE': 'D.J. MOORE',
        'AJ BROWN': 'A.J. BROWN',
        'CJ STROUD': 'C.J. STROUD',
        'TJ HOCKENSON': 'T.J. HOCKENSON',
        # Add more as needed
    }


def resolve_name_synonyms(name: str, mappings: Dict[str, str] = None) -> str:
    """
    Resolve name synonyms using mapping dictionary.
    
    Args:
        name: Input player name
        mappings: Optional custom mapping dictionary
        
    Returns:
        Canonical player name
    """
    if mappings is None:
        mappings = create_name_mappings()
    
    norm_name = normalize_name(name).upper()
    return mappings.get(norm_name, name)