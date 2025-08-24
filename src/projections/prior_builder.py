"""
Prior building from historical metrics with empirical Bayes shrinkage.

This module takes calculated team and player metrics and converts them
into priors suitable for simulation, applying appropriate shrinkage
toward league averages.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PriorBuilder:
    """
    Builds simulation priors from historical metrics.
    """
    
    def __init__(self, shrinkage_strength: float = 0.3):
        """
        Initialize prior builder.
        
        Args:
            shrinkage_strength: Amount of regression to mean (0.0 = no shrinkage, 1.0 = full shrinkage to mean)
        """
        self.shrinkage_strength = shrinkage_strength
        self.league_averages = {}
        self.team_priors = None
        self.player_priors = None
    
    def build_team_priors(self, team_metrics_df: pd.DataFrame, 
                          min_weeks: int = 8) -> pd.DataFrame:
        """
        Build team priors from historical metrics.
        
        Args:
            team_metrics_df: DataFrame with team-season metrics
            min_weeks: Minimum weeks played to include team
            
        Returns:
            DataFrame with team priors
        """
        logger.info("Building team priors")
        
        if team_metrics_df.empty:
            logger.warning("No team metrics provided")
            return pd.DataFrame()
        
        # Filter teams with sufficient data
        valid_teams = team_metrics_df[team_metrics_df['weeks_played'] >= min_weeks].copy()
        
        if valid_teams.empty:
            logger.warning(f"No teams with {min_weeks}+ weeks of data")
            return pd.DataFrame()
        
        # Calculate league averages  
        self._calculate_league_averages(valid_teams)
        
        # Build priors for each team
        team_priors = []
        
        for team in valid_teams['team'].unique():
            team_data = valid_teams[valid_teams['team'] == team]
            
            if len(team_data) == 0:
                continue
            
            # Use most recent season or average across seasons
            if len(team_data) == 1:
                recent_data = team_data.iloc[0]
            else:
                # Weight recent seasons more heavily
                weights = np.linspace(0.5, 1.0, len(team_data))
                recent_data = team_data.multiply(weights, axis=0).sum() / weights.sum()
            
            # Apply shrinkage to league averages
            prior = self._apply_team_shrinkage(recent_data, team_data)
            prior['team'] = team
            
            team_priors.append(prior)
        
        self.team_priors = pd.DataFrame(team_priors)
        
        logger.info(f"Built priors for {len(self.team_priors)} teams")
        
        return self.team_priors
    
    def build_player_priors(self, player_metrics_df: pd.DataFrame,
                           team_priors_df: Optional[pd.DataFrame] = None,
                           min_opportunities: int = 50) -> pd.DataFrame:
        """
        Build player priors from historical metrics.
        
        Args:
            player_metrics_df: DataFrame with player-season metrics
            team_priors_df: Optional team priors for context
            min_opportunities: Minimum opportunities to include player
            
        Returns:
            DataFrame with player priors
        """
        logger.info("Building player priors")
        
        if player_metrics_df.empty:
            logger.warning("No player metrics provided")
            return pd.DataFrame()
        
        # Build priors by position
        position_priors = []
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_priors = self._build_position_priors(
                player_metrics_df[player_metrics_df['position'] == position],
                position, 
                min_opportunities,
                team_priors_df
            )
            if not pos_priors.empty:
                position_priors.append(pos_priors)
        
        if position_priors:
            self.player_priors = pd.concat(position_priors, ignore_index=True)
        else:
            self.player_priors = pd.DataFrame()
        
        logger.info(f"Built priors for {len(self.player_priors)} players")
        
        return self.player_priors
    
    def _calculate_league_averages(self, team_metrics_df: pd.DataFrame):
        """Calculate league average metrics."""
        
        # Weight by total plays when calculating averages
        weights = team_metrics_df['total_plays']
        
        self.league_averages = {
            'pace': np.average(team_metrics_df['avg_pace'], weights=weights),
            'pass_rate': np.average(team_metrics_df['pass_rate'], weights=weights),
            'neutral_pass_rate': np.average(team_metrics_df['neutral_pass_rate'], weights=weights),
            'epa_per_play': np.average(team_metrics_df['epa_per_play'], weights=weights),
            'success_rate': np.average(team_metrics_df['success_rate'], weights=weights),
            'red_zone_td_rate': np.average(team_metrics_df['red_zone_td_rate'], weights=weights),
            'proe': np.average(team_metrics_df['proe'], weights=weights)
        }
        
        logger.info(f"League averages: {self.league_averages}")
    
    def _apply_team_shrinkage(self, team_data: pd.Series, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Apply empirical Bayes shrinkage to team metrics."""
        
        # Calculate reliability based on sample size and consistency
        n_seasons = len(historical_data)
        total_plays = team_data.get('total_plays', 1000)
        
        # More data = less shrinkage
        reliability = min(0.8, (n_seasons * total_plays) / 10000)
        shrinkage = self.shrinkage_strength * (1 - reliability)
        
        prior = {}
        
        for metric, league_avg in self.league_averages.items():
            team_value = team_data.get(metric, league_avg)
            
            # Apply shrinkage: prior = (1 - shrinkage) * team_value + shrinkage * league_avg
            prior[metric] = (1 - shrinkage) * team_value + shrinkage * league_avg
        
        # Add metadata
        prior.update({
            'n_seasons': n_seasons,
            'total_plays': total_plays,
            'reliability': reliability,
            'shrinkage_applied': shrinkage
        })
        
        return prior
    
    def _build_position_priors(self, pos_df: pd.DataFrame, position: str,
                              min_opportunities: int, team_priors_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Build priors for a specific position."""
        
        if pos_df.empty:
            return pd.DataFrame()
        
        # Position-specific opportunity thresholds
        opp_thresholds = {
            'QB': ('dropbacks', 100),
            'RB': ('opportunities', 50),  # carries + targets
            'WR': ('targets', 30),
            'TE': ('targets', 20)
        }
        
        opp_col, min_opp = opp_thresholds.get(position, ('targets', min_opportunities))
        
        # Filter players with sufficient opportunities
        if opp_col in pos_df.columns:
            qualified_players = pos_df[pos_df[opp_col] >= min_opp].copy()
        else:
            qualified_players = pos_df.copy()
        
        if qualified_players.empty:
            return pd.DataFrame()
        
        # Calculate position averages for shrinkage
        pos_averages = self._calculate_position_averages(qualified_players, position)
        
        player_priors = []
        
        for player_name in qualified_players['player_name'].unique():
            player_data = qualified_players[qualified_players['player_name'] == player_name]
            
            if len(player_data) == 0:
                continue
            
            # Get most recent season or weighted average
            if len(player_data) == 1:
                recent_data = player_data.iloc[0]
            else:
                # More recent seasons weighted more heavily
                weights = np.linspace(0.6, 1.0, len(player_data))
                numeric_cols = player_data.select_dtypes(include=[np.number]).columns
                weighted_avg = player_data[numeric_cols].multiply(weights, axis=0).sum() / weights.sum()
                recent_data = weighted_avg
                recent_data['player_name'] = player_name
                recent_data['position'] = position
                recent_data['team'] = player_data.iloc[-1]['team']  # Most recent team
            
            # Apply position-specific shrinkage
            prior = self._apply_player_shrinkage(recent_data, player_data, pos_averages, position)
            
            # Add team context if available
            if team_priors_df is not None:
                team_context = self._get_team_context(recent_data.get('team', ''), team_priors_df)
                prior.update(team_context)
            
            player_priors.append(prior)
        
        return pd.DataFrame(player_priors)
    
    def _calculate_position_averages(self, pos_df: pd.DataFrame, position: str) -> Dict[str, float]:
        """Calculate position-level averages."""
        
        # Position-specific key metrics
        key_metrics = {
            'QB': ['epa_per_db', 'success_rate', 'cpoe', 'pressure_rate'],
            'RB': ['rush_epa', 'rec_epa', 'opportunity_share', 'hvt_carry_rate'],
            'WR': ['target_epa', 'wopr', 'racr', 'deep_target_rate'],
            'TE': ['target_epa', 'wopr', 'racr', 'rz_target_share']
        }
        
        metrics = key_metrics.get(position, [])
        averages = {}
        
        for metric in metrics:
            if metric in pos_df.columns:
                values = pos_df[metric].dropna()
                if len(values) > 0:
                    averages[metric] = values.mean()
        
        return averages
    
    def _apply_player_shrinkage(self, player_data: pd.Series, historical_data: pd.DataFrame,
                               pos_averages: Dict[str, float], position: str) -> Dict[str, float]:
        """Apply empirical Bayes shrinkage to player metrics."""
        
        # Calculate reliability based on sample size and career length
        n_seasons = len(historical_data)
        
        # Position-specific reliability calculation
        if position == 'QB':
            sample_size = player_data.get('dropbacks', 200)
            reliability = min(0.7, (n_seasons * sample_size) / 5000)
        elif position == 'RB':
            sample_size = player_data.get('opportunities', 100)
            reliability = min(0.6, (n_seasons * sample_size) / 3000)
        else:  # WR/TE
            sample_size = player_data.get('targets', 50)
            reliability = min(0.5, (n_seasons * sample_size) / 1500)
        
        shrinkage = self.shrinkage_strength * (1 - reliability)
        
        prior = {
            'player_name': player_data.get('player_name', ''),
            'position': position,
            'team': player_data.get('team', ''),
            'n_seasons': n_seasons,
            'sample_size': sample_size,
            'reliability': reliability,
            'shrinkage_applied': shrinkage
        }
        
        # Apply shrinkage to key metrics
        for metric, pos_avg in pos_averages.items():
            player_value = player_data.get(metric, pos_avg)
            prior[metric] = (1 - shrinkage) * player_value + shrinkage * pos_avg
        
        # Add market share metrics without shrinkage (these are structural)
        share_metrics = ['target_share', 'carry_share', 'opportunity_share', 'air_yard_share']
        for metric in share_metrics:
            if metric in player_data.index:
                prior[metric] = player_data[metric]
        
        return prior
    
    def _get_team_context(self, team: str, team_priors_df: pd.DataFrame) -> Dict[str, float]:
        """Get team context for player priors."""
        
        if team == '' or team_priors_df.empty:
            return {}
        
        team_row = team_priors_df[team_priors_df['team'] == team]
        if team_row.empty:
            return {}
        
        team_data = team_row.iloc[0]
        
        return {
            'team_pace': team_data.get('pace', 65),
            'team_pass_rate': team_data.get('pass_rate', 0.6),
            'team_epa': team_data.get('epa_per_play', 0.0)
        }
    
    def save_priors(self, output_dir: Path):
        """Save priors to CSV files."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.team_priors is not None and not self.team_priors.empty:
            team_path = output_dir / 'team_priors.csv'
            self.team_priors.to_csv(team_path, index=False)
            logger.info(f"Saved team priors to {team_path}")
        
        if self.player_priors is not None and not self.player_priors.empty:
            player_path = output_dir / 'player_priors.csv'
            self.player_priors.to_csv(player_path, index=False)
            logger.info(f"Saved player priors to {player_path}")
    
    def get_summary(self) -> Dict[str, Dict]:
        """Get summary of built priors."""
        
        summary = {}
        
        if self.team_priors is not None and not self.team_priors.empty:
            summary['team_priors'] = {
                'n_teams': len(self.team_priors),
                'avg_pace': self.team_priors['pace'].mean(),
                'avg_pass_rate': self.team_priors['pass_rate'].mean(),
                'avg_shrinkage': self.team_priors['shrinkage_applied'].mean()
            }
        
        if self.player_priors is not None and not self.player_priors.empty:
            summary['player_priors'] = {
                'n_players': len(self.player_priors),
                'position_counts': self.player_priors['position'].value_counts().to_dict(),
                'avg_reliability': self.player_priors['reliability'].mean(),
                'avg_shrinkage': self.player_priors['shrinkage_applied'].mean()
            }
        
        if self.league_averages:
            summary['league_averages'] = self.league_averages
        
        return summary


def build_priors_from_metrics(team_metrics_df: pd.DataFrame,
                             player_metrics_df: pd.DataFrame,
                             output_dir: Optional[str] = None,
                             shrinkage_strength: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to build priors from metrics.
    
    Args:
        team_metrics_df: Team metrics DataFrame
        player_metrics_df: Player metrics DataFrame  
        output_dir: Optional output directory
        shrinkage_strength: Shrinkage parameter
        
    Returns:
        Tuple of (team_priors, player_priors) DataFrames
    """
    builder = PriorBuilder(shrinkage_strength=shrinkage_strength)
    
    # Build team priors first
    team_priors = builder.build_team_priors(team_metrics_df)
    
    # Build player priors with team context
    player_priors = builder.build_player_priors(player_metrics_df, team_priors)
    
    # Save if requested
    if output_dir:
        builder.save_priors(Path(output_dir))
    
    # Log summary
    summary = builder.get_summary()
    logger.info(f"Prior building summary: {summary}")
    
    return team_priors, player_priors