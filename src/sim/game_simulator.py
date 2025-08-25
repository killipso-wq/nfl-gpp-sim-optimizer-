"""
Monte Carlo game simulator for NFL fantasy football projections.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings

from ..ingest.scoring import apply_dk_bonuses_to_samples


class GameSimulator:
    """Monte Carlo simulator for NFL fantasy game simulation."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize simulator with optional seed for reproducibility.
        
        Args:
            seed: Random seed for reproducible results
        """
        self.seed = seed
        self.rng = np.random.Generator(np.random.PCG64(seed))
    
    def simulate_player(
        self, 
        mean: float, 
        variance: float, 
        position: str,
        n_sims: int = 10000,
        site_projection: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Simulate fantasy points for a single player.
        
        Args:
            mean: Expected fantasy points
            variance: Variance in fantasy points
            position: Player position (QB, RB, WR, TE, DST)
            n_sims: Number of Monte Carlo simulations
            site_projection: Site projection for comparison (optional)
            
        Returns:
            Dictionary with simulation results
        """
        # Generate samples based on position distribution
        samples = self._generate_samples(mean, variance, position, n_sims)
        
        # Apply DraftKings bonuses
        samples = apply_dk_bonuses_to_samples(pd.Series(samples), position).values
        
        # Calculate statistics
        results = self._calculate_statistics(samples, site_projection)
        
        return results
    
    def _generate_samples(self, mean: float, variance: float, position: str, n_sims: int) -> np.ndarray:
        """
        Generate Monte Carlo samples based on position-specific distributions.
        
        Args:
            mean: Expected value
            variance: Variance
            position: Player position
            n_sims: Number of simulations
            
        Returns:
            Array of simulated fantasy points
        """
        std = np.sqrt(variance)
        
        # Position-specific distributions
        if position in ['QB', 'WR', 'DST']:
            # Lognormal distribution for boom/bust positions
            if mean <= 0:
                # Fallback to normal for edge cases
                samples = self.rng.normal(mean, std, n_sims)
            else:
                # Convert to lognormal parameters
                log_mean = np.log(mean**2 / np.sqrt(variance + mean**2))
                log_std = np.sqrt(np.log(1 + variance / mean**2))
                samples = self.rng.lognormal(log_mean, log_std, n_sims)
        else:
            # Normal distribution for RB, TE
            samples = self.rng.normal(mean, std, n_sims)
        
        # Clamp at 0 minimum
        samples = np.maximum(samples, 0)
        
        return samples
    
    def _calculate_statistics(self, samples: np.ndarray, site_projection: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate summary statistics from simulation samples.
        
        Args:
            samples: Array of simulated values
            site_projection: Site projection for comparison
            
        Returns:
            Dictionary with statistical results
        """
        results = {
            'sim_mean': float(np.mean(samples)),
            'floor_p10': float(np.percentile(samples, 10)),
            'p75': float(np.percentile(samples, 75)),
            'ceiling_p90': float(np.percentile(samples, 90)),
            'p95': float(np.percentile(samples, 95)),
        }
        
        # Add site comparison if available
        if site_projection is not None:
            results['beat_site_prob'] = float(np.mean(samples >= site_projection))
        
        return results
    
    def simulate_slate(
        self,
        players_df: pd.DataFrame,
        player_priors: pd.DataFrame,
        team_priors: pd.DataFrame,
        boom_thresholds: Dict[str, float],
        position_variance: Dict[str, float],
        n_sims: int = 10000
    ) -> pd.DataFrame:
        """
        Simulate entire slate of players.
        
        Args:
            players_df: Site players data
            player_priors: Historical player priors
            team_priors: Historical team priors  
            boom_thresholds: Position-specific boom thresholds
            position_variance: Default variance by position
            n_sims: Number of simulations per player
            
        Returns:
            DataFrame with simulation results
        """
        results = []
        
        for idx, player in players_df.iterrows():
            # Get player projection parameters
            mean, variance, is_rookie = self._get_player_parameters(
                player, player_priors, team_priors, position_variance
            )
            
            # Run simulation
            sim_results = self.simulate_player(
                mean=mean,
                variance=variance,
                position=player['POS'],
                n_sims=n_sims,
                site_projection=player.get('FPTS')
            )
            
            # Calculate boom probability
            boom_threshold = boom_thresholds.get(player['POS'], 20.0)
            boom_prob = self._calculate_boom_probability(sim_results, boom_threshold)
            
            # Combine results
            player_result = {
                'player_id': player.get('player_id', ''),
                'PLAYER': player.get('PLAYER', ''),
                'POS': player.get('POS', ''),
                'TEAM': player.get('TEAM', ''),
                'OPP': player.get('OPP', ''),
                'SAL': player.get('SAL'),
                'rookie_fallback': is_rookie,
                'boom_prob': boom_prob,
                **sim_results
            }
            
            results.append(player_result)
        
        return pd.DataFrame(results)
    
    def _get_player_parameters(
        self,
        player: pd.Series,
        player_priors: pd.DataFrame,
        team_priors: pd.DataFrame,
        position_variance: Dict[str, float]
    ) -> Tuple[float, float, bool]:
        """
        Get simulation parameters for a player.
        
        Args:
            player: Player row from site data
            player_priors: Historical player data
            team_priors: Historical team data
            position_variance: Default variance by position
            
        Returns:
            Tuple of (mean, variance, is_rookie_fallback)
        """
        player_id = player.get('player_id', '')
        position = player.get('POS', '')
        team = player.get('TEAM', '')
        
        # Try to find player in historical priors
        if player_id and len(player_priors) > 0:
            player_prior = player_priors[player_priors['player_id'] == player_id]
            if not player_prior.empty:
                prior_mean = player_prior.iloc[0].get('mean_fantasy_points', 0)
                prior_variance = player_prior.iloc[0].get('variance_fantasy_points', 0)
                
                # Apply mild vegas adjustment
                vegas_multiplier = self._get_vegas_multiplier(player, team_priors)
                adjusted_mean = prior_mean * vegas_multiplier
                
                return adjusted_mean, prior_variance, False
        
        # Rookie fallback: use site projection
        site_projection = player.get('FPTS', 0)
        if pd.isna(site_projection):
            site_projection = 0
        
        fallback_variance = position_variance.get(position, 25.0)  # Default variance
        
        return float(site_projection), fallback_variance, True
    
    def _get_vegas_multiplier(self, player: pd.Series, team_priors: pd.DataFrame) -> float:
        """
        Calculate mild vegas adjustment multiplier.
        
        Args:
            player: Player data
            team_priors: Team historical data
            
        Returns:
            Vegas adjustment multiplier
        """
        team = player.get('TEAM', '')
        game_total = player.get('O/U')
        
        if pd.isna(game_total) or len(team_priors) == 0:
            return 1.0  # No adjustment
        
        # Get team's historical average
        team_prior = team_priors[team_priors['team'] == team]
        if team_prior.empty:
            season_avg = 45.0  # Default NFL game total
        else:
            season_avg = team_prior.iloc[0].get('avg_game_total', 45.0)
        
        # Conservative adjustment based on game total
        multiplier = (float(game_total) / season_avg) ** 0.3
        
        # Clamp to reasonable range
        return np.clip(multiplier, 0.8, 1.2)
    
    def _calculate_boom_probability(self, sim_results: Dict[str, float], boom_threshold: float) -> float:
        """
        Calculate boom probability from simulation results.
        
        This is an approximation since we don't store all samples.
        """
        ceiling = sim_results.get('ceiling_p90', 0)
        p95 = sim_results.get('p95', 0)
        
        # Rough approximation: if p90 > threshold, boom prob is high
        if ceiling >= boom_threshold:
            if p95 >= boom_threshold * 1.2:
                return 0.25  # High boom probability
            else:
                return 0.15  # Moderate boom probability
        else:
            return 0.05  # Low boom probability


def get_position_variance_defaults() -> Dict[str, float]:
    """
    Get default variance values by position.
    
    Returns:
        Dictionary mapping position to default variance
    """
    return {
        'QB': 30.0,
        'RB': 25.0,
        'WR': 35.0,
        'TE': 20.0,
        'DST': 40.0,
        'K': 15.0
    }


def get_position_distributions() -> Dict[str, str]:
    """
    Get distribution types by position.
    
    Returns:
        Dictionary mapping position to distribution type
    """
    return {
        'QB': 'lognormal',
        'RB': 'normal',
        'WR': 'lognormal',
        'TE': 'normal',
        'DST': 'lognormal',
        'K': 'normal'
    }