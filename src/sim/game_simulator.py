"""
Game simulation engine for NFL DFS projections.

This module simulates game environments and individual player performances
using Monte Carlo methods and historical priors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
from scipy.special import gammaln

from ..ingest.scoring import calculate_dk_points

logger = logging.getLogger(__name__)


class GameSimulator:
    """
    Simulates NFL games and player fantasy performances.
    """
    
    def __init__(self, n_sims: int = 10000, random_seed: Optional[int] = None):
        """
        Initialize game simulator.
        
        Args:
            n_sims: Number of simulations to run
            random_seed: Random seed for reproducibility
        """
        self.n_sims = n_sims
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Default league parameters
        self.league_params = {
            'avg_pace': 65,
            'pace_std': 5,
            'avg_pass_rate': 0.6,
            'pass_rate_std': 0.08,
            'avg_epa': 0.0,
            'epa_std': 0.15
        }
    
    def simulate_game_environment(self, game_info: Dict[str, Any],
                                 team_priors: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
        """
        Simulate game environment (pace, total plays, pass rate).
        
        Args:
            game_info: Game information (teams, over/under, spread)
            team_priors: Team prior parameters
            
        Returns:
            Dictionary with simulated game environment arrays
        """
        home_team = game_info.get('home_team', '')
        away_team = game_info.get('away_team', '')
        over_under = game_info.get('over_under', 45.0)
        spread = game_info.get('spread', 0.0)
        
        # Get team priors or use league defaults
        home_priors = team_priors.get(home_team, {})
        away_priors = team_priors.get(away_team, {})
        
        # Simulate game pace
        home_pace = home_priors.get('pace', self.league_params['avg_pace'])
        away_pace = away_priors.get('pace', self.league_params['avg_pace'])
        
        # Average the team paces and add game script variation
        base_pace = (home_pace + away_pace) / 2
        
        # Adjust for over/under (higher O/U = more pace)
        ou_adjustment = (over_under - 45) * 0.2
        
        # Simulate pace with some randomness
        game_pace = np.random.normal(
            base_pace + ou_adjustment,
            self.league_params['pace_std'],
            self.n_sims
        )
        game_pace = np.clip(game_pace, 45, 85)  # Reasonable bounds
        
        # Simulate total plays (based on pace)
        total_plays = np.random.poisson(game_pace, self.n_sims)
        total_plays = np.clip(total_plays, 40, 100)
        
        # Simulate pass rates for each team
        home_pass_rate = self._simulate_pass_rate(home_priors, spread, True)
        away_pass_rate = self._simulate_pass_rate(away_priors, spread, False)
        
        return {
            'total_plays': total_plays,
            'home_pass_rate': home_pass_rate,
            'away_pass_rate': away_pass_rate,
            'game_pace': game_pace
        }
    
    def _simulate_pass_rate(self, team_priors: Dict[str, float], 
                           spread: float, is_home: bool) -> np.ndarray:
        """Simulate pass rate for a team given game context."""
        
        base_pass_rate = team_priors.get('pass_rate', self.league_params['avg_pass_rate'])
        
        # Adjust for game script (spread)
        effective_spread = spread if is_home else -spread
        
        # Teams expected to be behind pass more
        if effective_spread < -3:  # Underdog by 3+
            base_pass_rate += 0.05
        elif effective_spread > 7:  # Favorite by 7+
            base_pass_rate -= 0.03
        
        # Add randomness
        pass_rates = np.random.normal(
            base_pass_rate,
            self.league_params['pass_rate_std'],
            self.n_sims
        )
        
        return np.clip(pass_rates, 0.35, 0.85)
    
    def simulate_team_volume_allocation(self, team_players: pd.DataFrame,
                                      team_plays: np.ndarray,
                                      pass_rate: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Allocate team volume among players using Dirichlet distribution.
        
        Args:
            team_players: DataFrame with team's players and their priors
            team_plays: Array of total team plays per simulation
            pass_rate: Array of team pass rates per simulation
            
        Returns:
            Dictionary mapping player names to volume arrays
        """
        # Separate players by position group
        qbs = team_players[team_players['position'] == 'QB']
        rbs = team_players[team_players['position'] == 'RB']
        receivers = team_players[team_players['position'].isin(['WR', 'TE'])]
        
        allocations = {}
        
        # QB allocation (simple - usually 1 QB gets all dropbacks)
        if len(qbs) > 0:
            qb_name = qbs.iloc[0]['player_name']  # Primary QB
            dropbacks = team_plays * pass_rate
            allocations[qb_name] = {
                'dropbacks': dropbacks.astype(int),
                'rushes': np.zeros(self.n_sims, dtype=int)
            }
        
        # RB allocation (carries and targets)
        if len(rbs) > 0:
            rb_allocations = self._allocate_rb_volume(rbs, team_plays, pass_rate)
            allocations.update(rb_allocations)
        
        # Receiver allocation (targets)
        if len(receivers) > 0:
            rec_allocations = self._allocate_receiver_volume(receivers, team_plays, pass_rate)
            allocations.update(rec_allocations)
        
        return allocations
    
    def _allocate_rb_volume(self, rbs: pd.DataFrame,
                           team_plays: np.ndarray,
                           pass_rate: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Allocate rushing and receiving volume among RBs."""
        
        allocations = {}
        
        if len(rbs) == 1:
            # Single RB gets most of the volume
            rb_name = rbs.iloc[0]['player_name']
            rush_plays = team_plays * (1 - pass_rate)
            target_share = rbs.iloc[0].get('target_share', 0.05)
            
            allocations[rb_name] = {
                'rushes': (rush_plays * 0.8).astype(int),  # RB gets 80% of rush attempts
                'targets': (team_plays * pass_rate * target_share).astype(int)
            }
        else:
            # Multiple RBs - use Dirichlet to allocate shares
            n_rbs = len(rbs)
            
            # Get historical shares as alpha parameters
            carry_shares = rbs['carry_share'].fillna(1.0 / n_rbs).values
            target_shares = rbs['target_share'].fillna(0.02).values
            
            # Convert shares to Dirichlet alphas (higher alpha = more concentrated)
            carry_alphas = carry_shares * 10  # Scale up for reasonable concentration
            target_alphas = target_shares * 50
            
            # Sample from Dirichlet for each simulation
            carry_distributions = np.random.dirichlet(carry_alphas, self.n_sims)
            target_distributions = np.random.dirichlet(target_alphas, self.n_sims)
            
            rush_plays = team_plays * (1 - pass_rate) * 0.8  # 80% to RBs
            
            for i, rb_name in enumerate(rbs['player_name'].values):
                allocations[rb_name] = {
                    'rushes': (rush_plays * carry_distributions[:, i]).astype(int),
                    'targets': (team_plays * pass_rate * target_distributions[:, i]).astype(int)
                }
        
        return allocations
    
    def _allocate_receiver_volume(self, receivers: pd.DataFrame,
                                 team_plays: np.ndarray,
                                 pass_rate: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Allocate passing targets among receivers."""
        
        allocations = {}
        
        if len(receivers) == 0:
            return allocations
        
        # Get target shares
        target_shares = receivers['target_share'].fillna(0.05).values
        target_shares = target_shares / target_shares.sum()  # Normalize
        
        # Use Dirichlet distribution for allocation
        # Higher alpha = more consistent allocation
        alphas = target_shares * 20  # Concentration parameter
        
        # Sample allocations for each simulation
        target_distributions = np.random.dirichlet(alphas, self.n_sims)
        
        # Total targets available (subtract RB targets)
        total_targets = team_plays * pass_rate
        
        for i, receiver_name in enumerate(receivers['player_name'].values):
            allocations[receiver_name] = {
                'targets': (total_targets * target_distributions[:, i]).astype(int),
                'rushes': np.zeros(self.n_sims, dtype=int)  # Receivers rarely rush
            }
        
        return allocations
    
    def simulate_player_efficiency(self, player_name: str,
                                  player_priors: Dict[str, float],
                                  position: str) -> Dict[str, np.ndarray]:
        """
        Simulate per-opportunity efficiency for a player.
        
        Args:
            player_name: Player name
            player_priors: Player's historical metrics/priors
            position: Player position
            
        Returns:
            Dictionary with efficiency arrays
        """
        if position == 'QB':
            return self._simulate_qb_efficiency(player_priors)
        elif position == 'RB':
            return self._simulate_rb_efficiency(player_priors)
        elif position in ['WR', 'TE']:
            return self._simulate_receiver_efficiency(player_priors)
        elif position == 'K':
            return self._simulate_kicker_efficiency(player_priors)
        elif position == 'DST':
            return self._simulate_dst_efficiency(player_priors)
        else:
            return {}
    
    def _simulate_qb_efficiency(self, priors: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Simulate QB per-dropback efficiency."""
        
        # Base metrics
        comp_pct = priors.get('comp_pct', 0.65)
        epa_per_db = priors.get('epa_per_db', 0.05)
        td_rate = priors.get('td_rate', 0.04)  # TDs per attempt
        int_rate = priors.get('int_rate', 0.02)  # INTs per attempt
        
        # Add variance
        comp_pcts = np.random.beta(
            comp_pct * 50, (1 - comp_pct) * 50, self.n_sims
        )
        
        # Yards per attempt (derived from EPA and completion rate)
        base_ypa = 7.0 + (epa_per_db * 10)  # Rough conversion
        ypa = np.random.normal(base_ypa, 1.5, self.n_sims)
        ypa = np.clip(ypa, 4, 12)
        
        # TD and INT rates
        td_rates = np.random.beta(td_rate * 100, (1 - td_rate) * 100, self.n_sims)
        int_rates = np.random.beta(int_rate * 100, (1 - int_rate) * 100, self.n_sims)
        
        return {
            'comp_pct': comp_pcts,
            'yards_per_attempt': ypa,
            'td_rate': td_rates,
            'int_rate': int_rates
        }
    
    def _simulate_rb_efficiency(self, priors: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Simulate RB per-opportunity efficiency."""
        
        # Rushing efficiency
        ypc = priors.get('ypc', 4.2)
        rush_td_rate = priors.get('rush_td_rate', 0.06)
        
        # Receiving efficiency  
        catch_rate = priors.get('catch_rate', 0.75)
        ypr = priors.get('ypr', 8.0)  # Yards per reception
        rec_td_rate = priors.get('rec_td_rate', 0.05)
        
        return {
            'yards_per_carry': np.random.normal(ypc, 1.0, self.n_sims),
            'rush_td_rate': np.random.beta(rush_td_rate * 50, (1 - rush_td_rate) * 50, self.n_sims),
            'catch_rate': np.random.beta(catch_rate * 30, (1 - catch_rate) * 30, self.n_sims),
            'yards_per_reception': np.random.normal(ypr, 2.0, self.n_sims),
            'rec_td_rate': np.random.beta(rec_td_rate * 50, (1 - rec_td_rate) * 50, self.n_sims)
        }
    
    def _simulate_receiver_efficiency(self, priors: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Simulate WR/TE per-target efficiency."""
        
        catch_rate = priors.get('catch_rate', 0.65)
        ypr = priors.get('yards_per_reception', 11.0)
        td_rate = priors.get('rec_td_rate', 0.06)
        
        return {
            'catch_rate': np.random.beta(catch_rate * 30, (1 - catch_rate) * 30, self.n_sims),
            'yards_per_reception': np.random.normal(ypr, 3.0, self.n_sims),
            'rec_td_rate': np.random.beta(td_rate * 50, (1 - td_rate) * 50, self.n_sims)
        }
    
    def _simulate_kicker_efficiency(self, priors: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Simulate kicker efficiency."""
        
        fg_pct = priors.get('fg_pct', 0.85)
        pat_pct = priors.get('pat_pct', 0.95)
        fg_attempts_per_game = priors.get('fg_attempts', 2.0)
        
        return {
            'fg_pct': np.random.beta(fg_pct * 20, (1 - fg_pct) * 20, self.n_sims),
            'pat_pct': np.random.beta(pat_pct * 40, (1 - pat_pct) * 40, self.n_sims),
            'fg_attempts': np.random.poisson(fg_attempts_per_game, self.n_sims)
        }
    
    def _simulate_dst_efficiency(self, priors: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Simulate defense/special teams efficiency."""
        
        sack_rate = priors.get('sack_rate', 0.07)
        int_rate = priors.get('int_rate', 0.02)
        fum_rec_rate = priors.get('fum_rec_rate', 0.01)
        
        return {
            'sacks_per_game': np.random.poisson(sack_rate * 35, self.n_sims),  # ~35 opponent dropbacks
            'ints_per_game': np.random.poisson(int_rate * 35, self.n_sims),
            'fum_rec_per_game': np.random.poisson(fum_rec_rate * 60, self.n_sims)  # ~60 opponent plays
        }
    
    def calculate_fantasy_points(self, player_name: str, position: str,
                                volume: Dict[str, np.ndarray],
                                efficiency: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate fantasy points from volume and efficiency.
        
        Args:
            player_name: Player name
            position: Player position
            volume: Volume allocation arrays
            efficiency: Efficiency arrays
            
        Returns:
            Array of fantasy points for each simulation
        """
        if position == 'QB':
            return self._calculate_qb_fantasy_points(volume, efficiency)
        elif position == 'RB':
            return self._calculate_rb_fantasy_points(volume, efficiency)
        elif position in ['WR', 'TE']:
            return self._calculate_receiver_fantasy_points(volume, efficiency)
        elif position == 'K':
            return self._calculate_kicker_fantasy_points(volume, efficiency)
        elif position == 'DST':
            return self._calculate_dst_fantasy_points(volume, efficiency)
        else:
            return np.zeros(self.n_sims)
    
    def _calculate_qb_fantasy_points(self, volume: Dict[str, np.ndarray],
                                   efficiency: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate QB fantasy points."""
        
        dropbacks = volume.get('dropbacks', np.zeros(self.n_sims))
        rushes = volume.get('rushes', np.zeros(self.n_sims))
        
        comp_pct = efficiency.get('comp_pct', np.full(self.n_sims, 0.65))
        ypa = efficiency.get('yards_per_attempt', np.full(self.n_sims, 7.0))
        td_rate = efficiency.get('td_rate', np.full(self.n_sims, 0.04))
        int_rate = efficiency.get('int_rate', np.full(self.n_sims, 0.02))
        
        # Passing stats
        attempts = dropbacks
        completions = np.random.binomial(attempts, comp_pct)
        pass_yards = completions * ypa
        pass_tds = np.random.binomial(attempts, td_rate)
        interceptions = np.random.binomial(attempts, int_rate)
        
        # Rushing stats (simplified)
        rush_yards = rushes * 3.5  # ~3.5 YPC for QBs
        rush_tds = np.random.binomial(rushes, 0.08)  # 8% TD rate for QB rushes
        
        # Fantasy points (DraftKings scoring)
        points = (
            pass_yards * 0.04 +  # 1 pt per 25 pass yards
            pass_tds * 4 +      # 4 pts per pass TD
            interceptions * -1 + # -1 per INT
            rush_yards * 0.1 +   # 1 pt per 10 rush yards
            rush_tds * 6         # 6 pts per rush TD
        )
        
        return np.maximum(0, points)
    
    def _calculate_rb_fantasy_points(self, volume: Dict[str, np.ndarray],
                                   efficiency: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate RB fantasy points."""
        
        rushes = volume.get('rushes', np.zeros(self.n_sims))
        targets = volume.get('targets', np.zeros(self.n_sims))
        
        ypc = efficiency.get('yards_per_carry', np.full(self.n_sims, 4.2))
        rush_td_rate = efficiency.get('rush_td_rate', np.full(self.n_sims, 0.06))
        catch_rate = efficiency.get('catch_rate', np.full(self.n_sims, 0.75))
        ypr = efficiency.get('yards_per_reception', np.full(self.n_sims, 8.0))
        rec_td_rate = efficiency.get('rec_td_rate', np.full(self.n_sims, 0.05))
        
        # Rushing stats
        rush_yards = rushes * ypc
        rush_tds = np.random.binomial(rushes, rush_td_rate)
        
        # Receiving stats
        receptions = np.random.binomial(targets, catch_rate)
        rec_yards = receptions * ypr
        rec_tds = np.random.binomial(receptions, rec_td_rate)
        
        # Fantasy points
        points = (
            rush_yards * 0.1 +   # 1 pt per 10 rush yards
            rush_tds * 6 +       # 6 pts per rush TD
            receptions * 1 +     # 1 pt per reception (PPR)
            rec_yards * 0.1 +    # 1 pt per 10 rec yards
            rec_tds * 6          # 6 pts per rec TD
        )
        
        return np.maximum(0, points)
    
    def _calculate_receiver_fantasy_points(self, volume: Dict[str, np.ndarray],
                                         efficiency: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate WR/TE fantasy points."""
        
        targets = volume.get('targets', np.zeros(self.n_sims))
        
        catch_rate = efficiency.get('catch_rate', np.full(self.n_sims, 0.65))
        ypr = efficiency.get('yards_per_reception', np.full(self.n_sims, 11.0))
        td_rate = efficiency.get('rec_td_rate', np.full(self.n_sims, 0.06))
        
        # Receiving stats
        receptions = np.random.binomial(targets, catch_rate)
        rec_yards = receptions * ypr
        rec_tds = np.random.binomial(receptions, td_rate)
        
        # Fantasy points
        points = (
            receptions * 1 +     # 1 pt per reception
            rec_yards * 0.1 +    # 1 pt per 10 yards
            rec_tds * 6          # 6 pts per TD
        )
        
        return np.maximum(0, points)
    
    def _calculate_kicker_fantasy_points(self, volume: Dict[str, np.ndarray],
                                       efficiency: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate kicker fantasy points."""
        
        fg_attempts = efficiency.get('fg_attempts', np.full(self.n_sims, 2))
        fg_pct = efficiency.get('fg_pct', np.full(self.n_sims, 0.85))
        
        # Simplified kicker scoring
        fg_made = np.random.binomial(fg_attempts, fg_pct)
        fg_missed = fg_attempts - fg_made
        
        # Assume 1 extra point per game on average
        pat_made = np.random.poisson(1.2, self.n_sims)
        
        points = fg_made * 3 + fg_missed * -1 + pat_made * 1
        
        return np.maximum(0, points)
    
    def _calculate_dst_fantasy_points(self, volume: Dict[str, np.ndarray],
                                    efficiency: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate DST fantasy points."""
        
        sacks = efficiency.get('sacks_per_game', np.full(self.n_sims, 2))
        ints = efficiency.get('ints_per_game', np.full(self.n_sims, 1))
        fum_rec = efficiency.get('fum_rec_per_game', np.full(self.n_sims, 1))
        
        # Simplified DST scoring (would need more detail for full implementation)
        points = sacks * 1 + ints * 2 + fum_rec * 2
        
        # Add random points allowed penalty/bonus
        points_allowed_bonus = np.random.choice([5, 3, 1, 0, -1, -3], size=self.n_sims, 
                                              p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
        points += points_allowed_bonus
        
        return np.maximum(0, points)


def simulate_player_projections(player_data: pd.DataFrame,
                              game_info: Dict[str, Any],
                              team_priors: Dict[str, Dict[str, float]],
                              player_priors: Dict[str, Dict[str, float]],
                              n_sims: int = 10000) -> pd.DataFrame:
    """
    Convenience function to simulate projections for all players in a game.
    
    Args:
        player_data: DataFrame with player information
        game_info: Game environment information
        team_priors: Team priors dictionary
        player_priors: Player priors dictionary  
        n_sims: Number of simulations
        
    Returns:
        DataFrame with simulation results
    """
    simulator = GameSimulator(n_sims=n_sims)
    
    # Simulate game environment
    game_env = simulator.simulate_game_environment(game_info, team_priors)
    
    results = []
    
    # Group players by team
    for team in player_data['team'].unique():
        team_players = player_data[player_data['team'] == team]
        
        # Determine if home or away
        is_home = (team == game_info.get('home_team', ''))
        pass_rate = game_env['home_pass_rate'] if is_home else game_env['away_pass_rate']
        
        # Allocate volume
        volume_allocations = simulator.simulate_team_volume_allocation(
            team_players, game_env['total_plays'], pass_rate
        )
        
        # Simulate each player
        for _, player in team_players.iterrows():
            player_name = player['player_name']
            position = player['position']
            
            if player_name not in volume_allocations:
                continue
            
            # Get player priors
            p_priors = player_priors.get(player_name, {})
            
            # Simulate efficiency
            efficiency = simulator.simulate_player_efficiency(player_name, p_priors, position)
            
            # Calculate fantasy points
            volume = volume_allocations[player_name]
            fantasy_points = simulator.calculate_fantasy_points(player_name, position, volume, efficiency)
            
            # Calculate percentiles
            results.append({
                'player_name': player_name,
                'position': position,
                'team': team,
                'proj_mean': fantasy_points.mean(),
                'floor': np.percentile(fantasy_points, 10),
                'p75': np.percentile(fantasy_points, 75),
                'ceiling': np.percentile(fantasy_points, 90),
                'p95': np.percentile(fantasy_points, 95),
                'std_dev': fantasy_points.std()
            })
    
    return pd.DataFrame(results)