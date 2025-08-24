"""Game simulation engine for NFL fantasy projections."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings


class GameSimulator:
    """NFL game simulation engine for fantasy projections."""
    
    def __init__(self, team_priors: pd.DataFrame, player_priors: pd.DataFrame):
        """Initialize simulator with team and player priors.
        
        Args:
            team_priors: Team prior parameters DataFrame
            player_priors: Player prior parameters DataFrame
        """
        self.team_priors = team_priors.set_index('team') if 'team' in team_priors.columns else team_priors
        self.player_priors = player_priors.set_index('player_id') if 'player_id' in player_priors.columns else player_priors
    
    def simulate_week(self, site_players: pd.DataFrame, n_sims: int = 10000) -> pd.DataFrame:
        """Simulate a full week of games.
        
        Args:
            site_players: Site players DataFrame with game environment data
            n_sims: Number of simulations to run
            
        Returns:
            DataFrame with simulation results for each player
        """
        print(f"Running {n_sims} simulations for {len(site_players)} players...")
        
        # Group players by game
        games = site_players.groupby('game_id')
        
        all_results = []
        
        for game_id, game_players in games:
            print(f"Simulating game {game_id} with {len(game_players)} players...")
            
            # Simulate this game
            game_results = self._simulate_game(game_players, n_sims)
            all_results.append(game_results)
        
        # Combine all game results
        results_df = pd.concat(all_results, ignore_index=True)
        
        print(f"Simulation complete. Generated projections for {len(results_df)} players.")
        
        return results_df
    
    def _simulate_game(self, game_players: pd.DataFrame, n_sims: int) -> pd.DataFrame:
        """Simulate a single game.
        
        Args:
            game_players: Players in this game
            n_sims: Number of simulations
            
        Returns:
            DataFrame with player simulation results
        """
        # Get team info
        teams = game_players[['team', 'opponent', 'total', 'spread']].drop_duplicates()
        
        if len(teams) == 0:
            return pd.DataFrame()
        
        # Use first row for game environment (should be consistent across players)
        game_info = game_players.iloc[0]
        team1 = game_info['team']
        team2 = game_info['opponent']  
        total_line = game_info['total']
        spread_line = game_info['spread']
        
        # Simulate game environment
        game_envs = self._simulate_game_environment(team1, team2, total_line, spread_line, n_sims)
        
        # Simulate each player
        player_results = []
        
        for _, player in game_players.iterrows():
            player_sims = self._simulate_player(player, game_envs, n_sims)
            
            # Calculate quantiles and summary stats
            fantasy_points = player_sims['fantasy_points']
            
            result = {
                'player_id': player['player_id'],
                'name': player['name'],
                'position': player['position'],
                'team': player['team'],
                'salary': player['salary'],
                'proj_mean': fantasy_points.mean(),
                'p10': fantasy_points.quantile(0.1),
                'p25': fantasy_points.quantile(0.25),
                'p50': fantasy_points.quantile(0.5),
                'p75': fantasy_points.quantile(0.75),
                'p90': fantasy_points.quantile(0.9),
                'p95': fantasy_points.quantile(0.95),
                'std': fantasy_points.std(),
                'floor': fantasy_points.quantile(0.1),  # Alias for p10
                'ceiling': fantasy_points.quantile(0.9),  # Alias for p90
            }
            
            player_results.append(result)
        
        return pd.DataFrame(player_results)
    
    def _simulate_game_environment(self, team1: str, team2: str, 
                                 total_line: float, spread_line: float, 
                                 n_sims: int) -> pd.DataFrame:
        """Simulate game-level environment factors.
        
        Args:
            team1: Team 1 abbreviation
            team2: Team 2 abbreviation (opponent)
            total_line: Over/under line
            spread_line: Point spread (positive = team1 favored)
            n_sims: Number of simulations
            
        Returns:
            DataFrame with game environment simulations
        """
        # Get team priors
        team1_priors = self._get_team_priors(team1)
        team2_priors = self._get_team_priors(team2)
        
        envs = []
        
        for i in range(n_sims):
            # Sample pace for each team
            team1_pace = np.random.normal(
                team1_priors.get('plays_per_game_prior', 65), 
                team1_priors.get('plays_per_game_std', 8)
            )
            team2_pace = np.random.normal(
                team2_priors.get('plays_per_game_prior', 65),
                team2_priors.get('plays_per_game_std', 8) 
            )
            
            # Average pace for the game
            game_pace = (team1_pace + team2_pace) / 2
            
            # Adjust pace based on total line (higher total = more plays)
            pace_adjustment = (total_line - 45) * 0.3  # Rough adjustment factor
            game_pace += pace_adjustment
            
            # Sample pass rates
            team1_neutral_pass_rate = np.random.normal(
                team1_priors.get('neutral_pass_rate_prior', 0.58),
                0.05  # Some variance
            )
            team2_neutral_pass_rate = np.random.normal(
                team2_priors.get('neutral_pass_rate_prior', 0.58),
                0.05
            )
            
            # Apply PROE adjustments
            team1_proe = team1_priors.get('proe_neutral_prior', 0)
            team2_proe = team2_priors.get('proe_neutral_prior', 0)
            
            team1_pass_rate = np.clip(team1_neutral_pass_rate + team1_proe, 0.3, 0.8)
            team2_pass_rate = np.clip(team2_neutral_pass_rate + team2_proe, 0.3, 0.8)
            
            # Game script adjustments (spread affects pass rate)
            if spread_line < -3:  # team1 favored by 3+
                team1_pass_rate *= 0.95  # Slightly fewer passes when ahead
                team2_pass_rate *= 1.05   # More passes when behind
            elif spread_line > 3:  # team1 underdog by 3+
                team1_pass_rate *= 1.05   # More passes when behind  
                team2_pass_rate *= 0.95   # Fewer passes when ahead
            
            envs.append({
                'sim': i,
                'game_pace': max(game_pace, 50),  # Minimum pace
                'team1_pace': max(team1_pace, 50),
                'team2_pace': max(team2_pace, 50), 
                'team1_pass_rate': np.clip(team1_pass_rate, 0.3, 0.8),
                'team2_pass_rate': np.clip(team2_pass_rate, 0.3, 0.8),
                'total_line': total_line,
                'spread_line': spread_line,
            })
        
        return pd.DataFrame(envs)
    
    def _simulate_player(self, player: pd.Series, game_envs: pd.DataFrame, n_sims: int) -> pd.DataFrame:
        """Simulate individual player performance.
        
        Args:
            player: Player row from site_players
            game_envs: Game environment simulations
            n_sims: Number of simulations
            
        Returns:
            DataFrame with player simulation results
        """
        position = player['position']
        team = player['team']
        
        # Get player priors
        player_priors = self._get_player_priors(player['player_id'])
        
        # Determine if this team is team1 or team2 in game environment
        team_prefix = 'team1' if team == player['team'] else 'team2'
        
        simulations = []
        
        for i in range(n_sims):
            game_env = game_envs.iloc[i]
            
            if position == 'QB':
                sim_result = self._simulate_qb(player_priors, game_env, team_prefix)
            elif position == 'RB':
                sim_result = self._simulate_rb(player_priors, game_env, team_prefix)
            elif position in ['WR', 'TE']:
                sim_result = self._simulate_receiver(player_priors, game_env, team_prefix)
            elif position == 'DST':
                sim_result = self._simulate_dst(player_priors, game_env, team_prefix)
            else:
                sim_result = {'fantasy_points': 0}
            
            sim_result['sim'] = i
            simulations.append(sim_result)
        
        return pd.DataFrame(simulations)
    
    def _simulate_qb(self, priors: Dict[str, Any], game_env: pd.Series, team_prefix: str) -> Dict[str, Any]:
        """Simulate QB performance."""
        pace = game_env[f'{team_prefix}_pace']
        pass_rate = game_env[f'{team_prefix}_pass_rate']
        
        # Sample attempts based on pace and pass rate
        attempts = np.random.poisson(pace * pass_rate * 0.7)  # Rough conversion
        attempts = max(attempts, 15)  # Minimum attempts
        
        # Sample completion rate
        completion_rate = np.random.normal(
            priors.get('completion_pct_prior', 0.65), 
            0.05
        )
        completion_rate = np.clip(completion_rate, 0.4, 0.8)
        
        completions = np.random.binomial(attempts, completion_rate)
        
        # Sample yards per attempt
        ypa = np.random.normal(
            priors.get('pass_ypa_prior', 7.0),
            1.0
        )
        ypa = max(ypa, 4.0)
        
        passing_yards = completions * ypa
        
        # Sample passing TDs
        td_rate = priors.get('pass_td_rate_prior', 0.05)
        passing_tds = np.random.binomial(attempts, td_rate)
        
        # Sample interceptions
        int_rate = priors.get('int_rate_prior', 0.03)
        interceptions = np.random.binomial(attempts, int_rate)
        
        # Sample rushing
        rush_attempts = np.random.poisson(priors.get('rush_attempts_prior', 4))
        rush_ypc = np.random.normal(priors.get('rush_ypc_prior', 4.5), 2.0)
        rush_ypc = max(rush_ypc, 1.0)
        
        rushing_yards = rush_attempts * rush_ypc
        
        rush_td_rate = priors.get('rush_td_rate_prior', 0.1)
        rushing_tds = np.random.binomial(rush_attempts, rush_td_rate)
        
        # Calculate fantasy points (DraftKings scoring)
        fantasy_points = (
            passing_yards * 0.04 +      # 1 pt per 25 yards
            passing_tds * 4 -           # 4 pts per passing TD
            interceptions * 1 +         # -1 pt per INT
            rushing_yards * 0.1 +       # 1 pt per 10 yards  
            rushing_tds * 6             # 6 pts per rushing TD
        )
        
        # Bonuses
        if passing_yards >= 300:
            fantasy_points += 3
        if rushing_yards >= 100:
            fantasy_points += 3
        
        return {
            'attempts': attempts,
            'completions': completions,
            'passing_yards': passing_yards,
            'passing_tds': passing_tds,
            'interceptions': interceptions,
            'rush_attempts': rush_attempts,
            'rushing_yards': rushing_yards,
            'rushing_tds': rushing_tds,
            'fantasy_points': fantasy_points,
        }
    
    def _simulate_rb(self, priors: Dict[str, Any], game_env: pd.Series, team_prefix: str) -> Dict[str, Any]:
        """Simulate RB performance."""
        pace = game_env[f'{team_prefix}_pace'] 
        pass_rate = game_env[f'{team_prefix}_pass_rate']
        
        # Sample carries based on pace and run rate
        team_carries = pace * (1 - pass_rate) * 0.8  # Rough conversion
        carry_share = priors.get('carry_share_prior', 0.3)
        carries = np.random.poisson(team_carries * carry_share)
        
        # Sample yards per carry
        ypc = np.random.normal(
            priors.get('rush_ypc_prior', 4.2),
            1.5
        )
        ypc = max(ypc, 1.0)
        
        rushing_yards = carries * ypc
        
        # Sample rushing TDs  
        td_rate = priors.get('rush_td_rate_prior', 0.05)
        rushing_tds = np.random.binomial(carries, td_rate)
        
        # Sample receiving
        team_targets = pace * pass_rate * 0.9  # Rough conversion
        target_share = priors.get('target_share_prior', 0.08)
        targets = np.random.poisson(team_targets * target_share)
        
        catch_rate = priors.get('catch_rate_prior', 0.75)
        receptions = np.random.binomial(targets, catch_rate)
        
        yards_per_target = priors.get('yards_per_target_prior', 6.0)
        receiving_yards = receptions * yards_per_target
        
        # Receiving TDs (lower rate)
        rec_td_rate = 0.05  # Fixed low rate for RBs
        receiving_tds = np.random.binomial(receptions, rec_td_rate)
        
        # Fantasy points
        fantasy_points = (
            rushing_yards * 0.1 +       # 1 pt per 10 rush yards
            rushing_tds * 6 +           # 6 pts per rush TD
            receptions * 1 +            # 1 pt per reception (PPR)
            receiving_yards * 0.1 +     # 1 pt per 10 rec yards
            receiving_tds * 6           # 6 pts per rec TD
        )
        
        # Bonuses
        if rushing_yards >= 100:
            fantasy_points += 3
        if receiving_yards >= 100:
            fantasy_points += 3
        
        return {
            'carries': carries,
            'rushing_yards': rushing_yards,
            'rushing_tds': rushing_tds,
            'targets': targets,
            'receptions': receptions,
            'receiving_yards': receiving_yards,
            'receiving_tds': receiving_tds,
            'fantasy_points': fantasy_points,
        }
    
    def _simulate_receiver(self, priors: Dict[str, Any], game_env: pd.Series, team_prefix: str) -> Dict[str, Any]:
        """Simulate WR/TE performance."""
        pace = game_env[f'{team_prefix}_pace']
        pass_rate = game_env[f'{team_prefix}_pass_rate']
        
        # Sample targets
        team_targets = pace * pass_rate * 0.9  # Rough conversion
        target_share = priors.get('target_share_prior', 0.15)
        targets = np.random.poisson(team_targets * target_share)
        
        # Sample catch rate
        catch_rate = np.random.normal(
            priors.get('catch_rate_prior', 0.65),
            0.1
        )
        catch_rate = np.clip(catch_rate, 0.3, 0.9)
        
        receptions = np.random.binomial(targets, catch_rate)
        
        # Sample yards per target
        ypt = np.random.normal(
            priors.get('yards_per_target_prior', 8.0),
            2.0
        )
        ypt = max(ypt, 3.0)
        
        receiving_yards = receptions * ypt
        
        # Sample TDs
        td_rate = priors.get('rec_td_rate_prior', 0.06)
        receiving_tds = np.random.binomial(targets, td_rate)
        
        # Fantasy points  
        fantasy_points = (
            receptions * 1 +            # 1 pt per reception (PPR)
            receiving_yards * 0.1 +     # 1 pt per 10 yards
            receiving_tds * 6           # 6 pts per TD
        )
        
        # Bonus
        if receiving_yards >= 100:
            fantasy_points += 3
        
        return {
            'targets': targets,
            'receptions': receptions,
            'receiving_yards': receiving_yards,
            'receiving_tds': receiving_tds,
            'fantasy_points': fantasy_points,
        }
    
    def _simulate_dst(self, priors: Dict[str, Any], game_env: pd.Series, team_prefix: str) -> Dict[str, Any]:
        """Simulate DST performance."""
        # Get opponent info (other team's performance affects DST)
        opp_prefix = 'team2' if team_prefix == 'team1' else 'team1'
        opp_pace = game_env[f'{opp_prefix}_pace']
        opp_pass_rate = game_env[f'{opp_prefix}_pass_rate']
        
        # Sample sacks (based on opponent pass rate)
        opp_dropbacks = opp_pace * opp_pass_rate * 0.7
        sack_rate = priors.get('opp_sack_rate_prior', 0.08)
        sacks = np.random.binomial(int(opp_dropbacks), sack_rate)
        
        # Sample interceptions
        int_rate = priors.get('opp_int_rate_prior', 0.03) 
        interceptions = np.random.binomial(int(opp_dropbacks), int_rate)
        
        # Sample fumble recoveries
        fumble_rate = priors.get('opp_fumble_rate_prior', 0.02)
        fumbles = np.random.binomial(int(opp_pace), fumble_rate)
        
        # Sample TDs (defensive/special teams)
        td_rate = 0.05  # Low base rate
        tds = np.random.binomial(1, td_rate)
        
        # Sample points allowed (affects scoring)
        points_allowed = np.random.normal(
            priors.get('points_allowed_prior', 22),
            6
        )
        points_allowed = max(points_allowed, 0)
        
        # Calculate DST fantasy points
        fantasy_points = (
            sacks * 1 +                 # 1 pt per sack
            interceptions * 2 +         # 2 pts per INT
            fumbles * 2 +               # 2 pts per fumble recovery
            tds * 6                     # 6 pts per TD
        )
        
        # Points allowed scoring
        if points_allowed <= 6:
            fantasy_points += 10
        elif points_allowed <= 13:
            fantasy_points += 7
        elif points_allowed <= 20:
            fantasy_points += 4
        elif points_allowed <= 27:
            fantasy_points += 1
        elif points_allowed <= 34:
            fantasy_points += 0
        else:
            fantasy_points -= 1
        
        return {
            'sacks': sacks,
            'interceptions': interceptions,
            'fumbles': fumbles,
            'tds': tds,
            'points_allowed': points_allowed,
            'fantasy_points': fantasy_points,
        }
    
    def _get_team_priors(self, team: str) -> Dict[str, Any]:
        """Get team prior parameters."""
        if team in self.team_priors.index:
            return self.team_priors.loc[team].to_dict()
        else:
            # Return league averages if team not found
            warnings.warn(f"Team {team} not found in priors, using league averages")
            return {
                'plays_per_game_prior': 65.0,
                'plays_per_game_std': 8.0,
                'pass_rate_prior': 0.60,
                'neutral_pass_rate_prior': 0.58,
                'proe_neutral_prior': 0.0,
                'epa_per_play_prior': 0.0,
                'success_rate_prior': 0.45,
            }
    
    def _get_player_priors(self, player_id: str) -> Dict[str, Any]:
        """Get player prior parameters."""
        if player_id in self.player_priors.index:
            return self.player_priors.loc[player_id].to_dict()
        else:
            # Return position-based defaults if player not found
            warnings.warn(f"Player {player_id} not found in priors, using position defaults")
            
            # Try to infer position from player_id
            if '_QB_' in player_id:
                return self._get_default_qb_priors()
            elif '_RB_' in player_id:
                return self._get_default_rb_priors()
            elif '_WR_' in player_id or '_TE_' in player_id:
                return self._get_default_receiver_priors()
            elif '_DST' in player_id:
                return self._get_default_dst_priors()
            else:
                return self._get_default_receiver_priors()  # Fallback
    
    def _get_default_qb_priors(self) -> Dict[str, Any]:
        """Get default QB priors."""
        return {
            'pass_attempts_prior': 32.0,
            'completion_pct_prior': 0.65,
            'pass_ypa_prior': 7.0,
            'pass_td_rate_prior': 0.05,
            'int_rate_prior': 0.03,
            'rush_attempts_prior': 4.0,
            'rush_ypc_prior': 4.5,
            'rush_td_rate_prior': 0.1,
        }
    
    def _get_default_rb_priors(self) -> Dict[str, Any]:
        """Get default RB priors."""
        return {
            'carry_share_prior': 0.3,
            'rush_ypc_prior': 4.2,
            'rush_td_rate_prior': 0.05,
            'target_share_prior': 0.08,
            'catch_rate_prior': 0.75,
            'yards_per_target_prior': 6.0,
        }
    
    def _get_default_receiver_priors(self) -> Dict[str, Any]:
        """Get default WR/TE priors.""" 
        return {
            'target_share_prior': 0.15,
            'catch_rate_prior': 0.65,
            'yards_per_target_prior': 8.0,
            'rec_td_rate_prior': 0.06,
        }
    
    def _get_default_dst_priors(self) -> Dict[str, Any]:
        """Get default DST priors."""
        return {
            'opp_sack_rate_prior': 0.08,
            'opp_int_rate_prior': 0.03,
            'opp_fumble_rate_prior': 0.02,
            'points_allowed_prior': 22.0,
        }