"""
Data source loaders for nfl_data_py with API version tolerance.

This module provides stable interfaces to nfl_data_py data while handling
potential API changes and providing caching for efficiency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class NFLDataLoader:
    """
    Wrapper around nfl_data_py with error handling and caching.
    """
    
    def __init__(self, cache_hours: int = 24):
        self.cache_hours = cache_hours
        self._cache = {}
        self._cache_timestamps = {}
        
        # Try to import nfl_data_py
        try:
            import nfl_data_py as nfl
            self.nfl = nfl
        except ImportError:
            raise ImportError("nfl_data_py is required. Install with: pip install nfl_data_py")
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache_timestamps:
            return False
        
        cache_time = self._cache_timestamps[key]
        expiry_time = cache_time + timedelta(hours=self.cache_hours)
        return datetime.now() < expiry_time
    
    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Get data from cache if valid."""
        if self._is_cache_valid(key):
            return self._cache.get(key)
        return None
    
    def _store_in_cache(self, key: str, data: pd.DataFrame):
        """Store data in cache."""
        self._cache[key] = data.copy()
        self._cache_timestamps[key] = datetime.now()
    
    def load_play_by_play(self, seasons: List[int], cache: bool = True) -> pd.DataFrame:
        """
        Load play-by-play data for specified seasons.
        
        Args:
            seasons: List of seasons (e.g., [2023, 2024])
            cache: Whether to use/store cache
            
        Returns:
            Combined play-by-play DataFrame
        """
        cache_key = f"pbp_{'-'.join(map(str, seasons))}"
        
        if cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached play-by-play data for seasons {seasons}")
                return cached_data
        
        logger.info(f"Loading play-by-play data for seasons {seasons}")
        
        dfs = []
        for season in seasons:
            try:
                df = self.nfl.import_pbp_data([season])
                if df is not None and not df.empty:
                    dfs.append(df)
                    logger.info(f"Loaded {len(df)} plays for season {season}")
                else:
                    logger.warning(f"No play-by-play data found for season {season}")
            except Exception as e:
                logger.error(f"Failed to load play-by-play data for season {season}: {e}")
                continue
        
        if not dfs:
            logger.warning("No play-by-play data loaded for any season")
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        if cache:
            self._store_in_cache(cache_key, combined_df)
        
        return combined_df
    
    def load_weekly_data(self, seasons: List[int], cache: bool = True) -> pd.DataFrame:
        """
        Load weekly player stats for specified seasons.
        
        Args:
            seasons: List of seasons
            cache: Whether to use/store cache
            
        Returns:
            Weekly stats DataFrame
        """
        cache_key = f"weekly_{'-'.join(map(str, seasons))}"
        
        if cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached weekly data for seasons {seasons}")
                return cached_data
        
        logger.info(f"Loading weekly data for seasons {seasons}")
        
        dfs = []
        for season in seasons:
            try:
                df = self.nfl.import_weekly_data([season])
                if df is not None and not df.empty:
                    dfs.append(df)
                    logger.info(f"Loaded weekly data for {len(df)} player-weeks in season {season}")
                else:
                    logger.warning(f"No weekly data found for season {season}")
            except Exception as e:
                logger.error(f"Failed to load weekly data for season {season}: {e}")
                continue
        
        if not dfs:
            logger.warning("No weekly data loaded for any season")
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        if cache:
            self._store_in_cache(cache_key, combined_df)
        
        return combined_df
    
    def load_schedules(self, seasons: List[int], cache: bool = True) -> pd.DataFrame:
        """
        Load schedule data for specified seasons.
        
        Args:
            seasons: List of seasons
            cache: Whether to use/store cache
            
        Returns:
            Schedule DataFrame
        """
        cache_key = f"schedules_{'-'.join(map(str, seasons))}"
        
        if cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached schedule data for seasons {seasons}")
                return cached_data
        
        logger.info(f"Loading schedule data for seasons {seasons}")
        
        dfs = []
        for season in seasons:
            try:
                df = self.nfl.import_schedules([season])
                if df is not None and not df.empty:
                    dfs.append(df)
                    logger.info(f"Loaded {len(df)} games for season {season}")
                else:
                    logger.warning(f"No schedule data found for season {season}")
            except Exception as e:
                logger.error(f"Failed to load schedule data for season {season}: {e}")
                continue
        
        if not dfs:
            logger.warning("No schedule data loaded for any season")  
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        if cache:
            self._store_in_cache(cache_key, combined_df)
        
        return combined_df
    
    def load_team_info(self, cache: bool = True) -> pd.DataFrame:
        """
        Load team information.
        
        Args:
            cache: Whether to use/store cache
            
        Returns:
            Team info DataFrame
        """
        cache_key = "team_info"
        
        if cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info("Using cached team info")
                return cached_data
        
        logger.info("Loading team info")
        
        try:
            df = self.nfl.import_team_desc()
            if df is not None and not df.empty:
                logger.info(f"Loaded info for {len(df)} teams")
                if cache:
                    self._store_in_cache(cache_key, df)
                return df
            else:
                logger.warning("No team info data found")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load team info: {e}")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache cleared")
    
    def get_cache_info(self) -> Dict[str, dict]:
        """Get information about cached data."""
        info = {}
        for key in self._cache.keys():
            df = self._cache[key]
            timestamp = self._cache_timestamps[key]
            info[key] = {
                'rows': len(df),
                'columns': len(df.columns),
                'cached_at': timestamp.isoformat(),
                'expires_at': (timestamp + timedelta(hours=self.cache_hours)).isoformat()
            }
        return info


def get_available_seasons() -> List[int]:
    """
    Get list of available seasons from nfl_data_py.
    
    Returns:
        List of available season years
    """
    # nfl_data_py typically has data from 1999 to current season
    current_year = datetime.now().year
    
    # NFL season starts in September, so if we're before September,
    # the current NFL season is the previous year
    if datetime.now().month < 9:
        current_season = current_year - 1
    else:
        current_season = current_year
    
    # Return seasons from 1999 to current
    return list(range(1999, current_season + 1))


def validate_data_availability(loader: NFLDataLoader, seasons: List[int]) -> Dict[str, bool]:
    """
    Check what data is available for the requested seasons.
    
    Args:
        loader: NFLDataLoader instance
        seasons: List of seasons to check
        
    Returns:
        Dictionary indicating data availability by type
    """
    availability = {
        'play_by_play': False,
        'weekly_data': False, 
        'schedules': False
    }
    
    # Try to load small sample to check availability
    test_season = seasons[0] if seasons else 2023
    
    try:
        pbp = loader.load_play_by_play([test_season])
        availability['play_by_play'] = not pbp.empty
    except Exception:
        pass
    
    try:
        weekly = loader.load_weekly_data([test_season])
        availability['weekly_data'] = not weekly.empty
    except Exception:
        pass
    
    try:
        schedules = loader.load_schedules([test_season])
        availability['schedules'] = not schedules.empty
    except Exception:
        pass
    
    return availability


# Global loader instance
_global_loader = None


def get_global_loader(cache_hours: int = 24) -> NFLDataLoader:
    """Get global NFL data loader instance."""
    global _global_loader
    if _global_loader is None:
        _global_loader = NFLDataLoader(cache_hours=cache_hours)
    return _global_loader