"""Player name normalization utilities."""

import re
from typing import Dict


def normalize_name(name: str) -> str:
    """Normalize player name to standard format.
    
    Args:
        name: Raw player name
        
    Returns:
        Normalized name
    """
    if not name or not isinstance(name, str):
        return ""
    
    # Basic cleaning
    name = name.strip()
    
    # Remove suffixes
    name = remove_suffixes(name)
    
    # Handle apostrophes and hyphens consistently
    name = name.replace("'", "").replace("-", " ")
    
    # Remove periods  
    name = name.replace(".", "")
    
    # Standardize spacing
    name = re.sub(r'\s+', ' ', name)
    
    # Title case
    name = name.title()
    
    # Handle known exceptions
    name = apply_name_exceptions(name)
    
    return name


def remove_suffixes(name: str) -> str:
    """Remove common name suffixes."""
    suffixes = [
        r'\s+Jr\.?$', 
        r'\s+Sr\.?$',
        r'\s+III$',
        r'\s+IV$',
        r'\s+V$',
        r'\s+II$',
    ]
    
    for suffix in suffixes:
        name = re.sub(suffix, '', name, flags=re.IGNORECASE)
    
    return name


def apply_name_exceptions(name: str) -> str:
    """Apply known name standardizations."""
    
    # Common NFL name variations
    name_exceptions = {
        'Dk Metcalf': 'DK Metcalf',
        'Cj Stroud': 'CJ Stroud', 
        'Aj Brown': 'AJ Brown',
        'Dj Moore': 'DJ Moore',
        'Tj Hockenson': 'TJ Hockenson',
        'Jj Mccarthy': 'JJ McCarthy',
        'Ej Manuel': 'EJ Manuel',
        # Add more as needed
    }
    
    return name_exceptions.get(name, name)


def build_player_id(team: str, position: str, name: str) -> str:
    """Build standardized player ID.
    
    Format: TEAM_POS_NORMALIZEDNAME
    Special case: DST players use TEAM_DST
    
    Args:
        team: Team abbreviation
        position: Position abbreviation  
        name: Player name
        
    Returns:
        Standardized player ID
    """
    if position == 'DST':
        return f"{team}_DST"
    
    # Normalize name for ID
    normalized_name = normalize_name(name)
    normalized_name = normalized_name.replace(' ', '_')
    normalized_name = re.sub(r'[^a-zA-Z0-9_]', '', normalized_name)
    
    return f"{team}_{position}_{normalized_name}"


def create_name_mapping_table() -> Dict[str, str]:
    """Create mapping table for common name variations.
    
    This would typically be built from historical data matching.
    For now, returns a basic set of common mappings.
    
    Returns:
        Dictionary mapping variations to canonical names
    """
    
    mappings = {
        # Nicknames to full names
        'Josh Gordon': 'Joshua Gordon',
        'Chris Jones': 'Christopher Jones', 
        'Mike Evans': 'Michael Evans',
        'Rob Gronkowski': 'Robert Gronkowski',
        
        # Common variations
        'T.J. Watt': 'TJ Watt',
        'A.J. Brown': 'AJ Brown', 
        'D.K. Metcalf': 'DK Metcalf',
        'C.J. Stroud': 'CJ Stroud',
        'D.J. Moore': 'DJ Moore',
        
        # Team DST variations
        'Kansas City Defense': 'Kansas City DST',
        'KC Defense': 'KC DST', 
        'Chiefs D/ST': 'Kansas City DST',
        'Buffalo Defense': 'Buffalo DST',
        'Bills D/ST': 'Buffalo DST',
    }
    
    return mappings


def match_player_names(site_names: list, baseline_names: list, threshold: float = 0.8) -> Dict[str, str]:
    """Match site player names to baseline names using fuzzy matching.
    
    Args:
        site_names: List of names from site data
        baseline_names: List of names from baseline data  
        threshold: Minimum similarity score (0-1)
        
    Returns:
        Dictionary mapping site names to baseline names
    """
    try:
        from difflib import SequenceMatcher
    except ImportError:
        # Fallback to exact matching if difflib not available
        exact_matches = {}
        baseline_set = set(baseline_names)
        for name in site_names:
            if name in baseline_set:
                exact_matches[name] = name
        return exact_matches
    
    matches = {}
    
    for site_name in site_names:
        best_match = None
        best_score = 0
        
        normalized_site = normalize_name(site_name)
        
        for baseline_name in baseline_names:
            normalized_baseline = normalize_name(baseline_name)
            
            # Calculate similarity
            similarity = SequenceMatcher(None, normalized_site, normalized_baseline).ratio()
            
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = baseline_name
        
        if best_match:
            matches[site_name] = best_match
    
    return matches


def standardize_team_abbreviations(team: str) -> str:
    """Standardize team abbreviations.
    
    Args:
        team: Team abbreviation or name
        
    Returns:
        Standardized abbreviation
    """
    if not team or not isinstance(team, str):
        return ""
    
    team = team.upper().strip()
    
    # Handle full team names
    team_name_map = {
        'ARIZONA CARDINALS': 'ARI',
        'ATLANTA FALCONS': 'ATL', 
        'BALTIMORE RAVENS': 'BAL',
        'BUFFALO BILLS': 'BUF',
        'CAROLINA PANTHERS': 'CAR',
        'CHICAGO BEARS': 'CHI',
        'CINCINNATI BENGALS': 'CIN',
        'CLEVELAND BROWNS': 'CLE',
        'DALLAS COWBOYS': 'DAL',
        'DENVER BRONCOS': 'DEN',
        'DETROIT LIONS': 'DET',
        'GREEN BAY PACKERS': 'GB',
        'HOUSTON TEXANS': 'HOU',
        'INDIANAPOLIS COLTS': 'IND',
        'JACKSONVILLE JAGUARS': 'JAX',
        'KANSAS CITY CHIEFS': 'KC',
        'LAS VEGAS RAIDERS': 'LV',
        'LOS ANGELES CHARGERS': 'LAC',
        'LOS ANGELES RAMS': 'LAR',
        'MIAMI DOLPHINS': 'MIA',
        'MINNESOTA VIKINGS': 'MIN',
        'NEW ENGLAND PATRIOTS': 'NE',
        'NEW ORLEANS SAINTS': 'NO',
        'NEW YORK GIANTS': 'NYG',
        'NEW YORK JETS': 'NYJ',
        'PHILADELPHIA EAGLES': 'PHI',
        'PITTSBURGH STEELERS': 'PIT',
        'SEATTLE SEAHAWKS': 'SEA',
        'SAN FRANCISCO 49ERS': 'SF',
        'TAMPA BAY BUCCANEERS': 'TB',
        'TENNESSEE TITANS': 'TEN',
        'WASHINGTON COMMANDERS': 'WAS',
    }
    
    if team in team_name_map:
        return team_name_map[team]
    
    # Handle common abbreviation variations
    abbrev_map = {
        'ARZ': 'ARI',
        'JAC': 'JAX',
        'LVR': 'LV',
        'RAIDERS': 'LV',
        'RAMS': 'LAR', 
        'CHARGERS': 'LAC',
        'WSH': 'WAS',
        'WFT': 'WAS',
    }
    
    return abbrev_map.get(team, team)