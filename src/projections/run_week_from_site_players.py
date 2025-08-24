#!/usr/bin/env python3
"""
Main simulation runner that processes site player CSV and outputs comprehensive results.

Usage:
    python -m src.projections.run_week_from_site_players --season 2025 --week 1 \
           --players-site path/to/players.csv \
           --team-priors data/baseline/team_priors.csv \
           --player-priors data/baseline/player_priors.csv \
           --boom-thresholds data/baseline/boom_thresholds.json \
           --sims 10000 --out data/sim_week
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from src.ingest.site_players import load_site_players_csv
from src.sim.game_simulator import GameSimulator, simulate_player_projections
from src.projections.value_metrics import calculate_value_metrics, calculate_salary_tiers, calculate_ceiling_analysis
from src.projections.boom_score import calculate_boom_scores, identify_contrarian_plays
from src.projections.diagnostics import (
    calculate_projection_diagnostics, identify_projection_flags, 
    export_diagnostics_report, validate_projection_reasonableness
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_priors_and_thresholds(team_priors_path: str, 
                              player_priors_path: str,
                              boom_thresholds_path: str) -> tuple:
    """Load all required prior data."""
    
    logger = logging.getLogger(__name__)
    
    # Load team priors
    logger.info(f"Loading team priors from {team_priors_path}")
    team_priors_df = pd.read_csv(team_priors_path)
    team_priors = {
        row['team']: row.to_dict() 
        for _, row in team_priors_df.iterrows()
    }
    logger.info(f"Loaded priors for {len(team_priors)} teams")
    
    # Load player priors  
    logger.info(f"Loading player priors from {player_priors_path}")
    player_priors_df = pd.read_csv(player_priors_path)
    
    # Handle both 'player_name' and 'player_id' columns
    player_name_col = 'player_name' if 'player_name' in player_priors_df.columns else 'player_id'
    
    player_priors = {
        row[player_name_col]: row.to_dict() 
        for _, row in player_priors_df.iterrows()
    }
    logger.info(f"Loaded priors for {len(player_priors)} players")
    
    # Load boom thresholds
    logger.info(f"Loading boom thresholds from {boom_thresholds_path}")
    with open(boom_thresholds_path, 'r') as f:
        boom_thresholds = json.load(f)
    logger.info(f"Loaded boom thresholds for {list(boom_thresholds.keys())}")
    
    return team_priors, player_priors, boom_thresholds


def build_game_info(players_df: pd.DataFrame) -> dict:
    """Build game environment info from player data."""
    
    # Extract teams
    teams = players_df['team'].unique()
    
    # Try to infer game matchup from opponent data
    if 'opponent_clean' in players_df.columns and 'home_away' in players_df.columns:
        # Find a clear home/away matchup
        for _, player in players_df.iterrows():
            if player.get('home_away') == 'home':
                home_team = player['team']
                away_team = player.get('opponent_clean', '')
                break
            elif player.get('home_away') == 'away':
                away_team = player['team']
                home_team = player.get('opponent_clean', '')
                break
        else:
            # Fallback: alphabetical order
            sorted_teams = sorted([t for t in teams if t])
            home_team = sorted_teams[0] if len(sorted_teams) > 0 else ''
            away_team = sorted_teams[1] if len(sorted_teams) > 1 else ''
    else:
        # Fallback: use first two teams
        team_list = list(teams)
        home_team = team_list[0] if len(team_list) > 0 else ''
        away_team = team_list[1] if len(team_list) > 1 else ''
    
    # Extract game environment data
    over_under = players_df['over_under'].dropna().iloc[0] if 'over_under' in players_df.columns else 45.0
    spread = players_df['spread'].dropna().iloc[0] if 'spread' in players_df.columns else 0.0
    
    game_info = {
        'home_team': home_team,
        'away_team': away_team,
        'over_under': float(over_under) if pd.notna(over_under) else 45.0,
        'spread': float(spread) if pd.notna(spread) else 0.0
    }
    
    return game_info


def run_simulation(players_df: pd.DataFrame,
                  team_priors: dict,
                  player_priors: dict,
                  n_sims: int = 10000) -> pd.DataFrame:
    """Run the main simulation."""
    
    logger = logging.getLogger(__name__)
    
    # Build game info
    game_info = build_game_info(players_df)
    logger.info(f"Game info: {game_info}")
    
    # Prepare player data for simulation
    sim_players_df = players_df[['player', 'position', 'team']].copy()
    sim_players_df = sim_players_df.rename(columns={'player': 'player_name'})
    
    # Run simulation
    logger.info(f"Running {n_sims:,} simulations...")
    simulator = GameSimulator(n_sims=n_sims)
    
    # Simulate game environment
    game_env = simulator.simulate_game_environment(game_info, team_priors)
    
    results = []
    
    # Group players by team and simulate
    for team in sim_players_df['team'].unique():
        if not team:
            continue
            
        team_players = sim_players_df[sim_players_df['team'] == team]
        logger.info(f"Simulating {len(team_players)} players from {team}")
        
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
                # Create minimal allocation for players not in main allocation
                volume_allocations[player_name] = {
                    'targets': np.zeros(n_sims, dtype=int),
                    'rushes': np.zeros(n_sims, dtype=int),
                    'dropbacks': np.zeros(n_sims, dtype=int)
                }
            
            # Get player priors (use defaults if not found)
            p_priors = player_priors.get(player_name, {})
            
            # Simulate efficiency
            efficiency = simulator.simulate_player_efficiency(player_name, p_priors, position)
            
            # Calculate fantasy points
            volume = volume_allocations[player_name]
            fantasy_points = simulator.calculate_fantasy_points(player_name, position, volume, efficiency)
            
            # Calculate statistics
            results.append({
                'player': player_name,
                'position': position,
                'team': team,
                'proj_mean': round(fantasy_points.mean(), 2),
                'floor': round(np.percentile(fantasy_points, 10), 2),
                'p75': round(np.percentile(fantasy_points, 75), 2),
                'ceiling': round(np.percentile(fantasy_points, 90), 2),
                'p95': round(np.percentile(fantasy_points, 95), 2),
                'std_dev': round(fantasy_points.std(), 2),
                'min_sim': round(fantasy_points.min(), 2),
                'max_sim': round(fantasy_points.max(), 2),
                'sim_data': fantasy_points  # Keep for boom probability calculation
            })
    
    return pd.DataFrame(results)


def calculate_boom_probabilities(sim_results: pd.DataFrame, 
                                boom_thresholds: dict,
                                site_fpts: pd.Series = None) -> pd.DataFrame:
    """Calculate boom probabilities from simulation results."""
    
    logger = logging.getLogger(__name__)
    logger.info("Calculating boom probabilities")
    
    boom_probs = []
    beat_site_probs = []
    
    for _, row in sim_results.iterrows():
        position = row['position']
        fantasy_points = row['sim_data']
        
        # Position boom threshold
        pos_threshold = boom_thresholds.get(position, 15.0)
        
        # Site comparison threshold if available
        player_name = row['player']
        site_fpts_val = site_fpts.get(player_name) if site_fpts is not None else None
        
        if pd.notna(site_fpts_val):
            # Boom threshold is max of position threshold, 1.2x site, or site+5
            effective_threshold = max(pos_threshold, site_fpts_val * 1.2, site_fpts_val + 5)
            beat_site_prob = (fantasy_points >= site_fpts_val).mean() * 100
        else:
            effective_threshold = pos_threshold
            beat_site_prob = np.nan
        
        boom_prob = (fantasy_points >= effective_threshold).mean() * 100
        
        boom_probs.append(round(boom_prob, 1))
        beat_site_probs.append(round(beat_site_prob, 1) if not np.isnan(beat_site_prob) else np.nan)
    
    sim_results['boom_prob'] = boom_probs
    sim_results['beat_site_prob'] = beat_site_probs
    
    # Remove sim_data column as it's large
    sim_results = sim_results.drop(columns=['sim_data'])
    
    return sim_results


def create_comparison_output(sim_results: pd.DataFrame, 
                           players_df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive comparison output with all metrics."""
    
    logger = logging.getLogger(__name__)
    logger.info("Creating comparison output")
    
    # Merge simulation results with original player data
    compare_df = sim_results.merge(
        players_df[['player', 'salary', 'ownership', 'fpts', 'over_under', 'spread', 
                   'money_line', 'team_total', 'value']],
        on='player',
        how='left'
    )
    
    return compare_df


def main():
    parser = argparse.ArgumentParser(description='Run NFL DFS simulation from site player data')
    parser.add_argument('--season', type=int, required=True, help='Season year (e.g., 2025)')
    parser.add_argument('--week', type=int, required=True, help='Week number')
    parser.add_argument('--players-site', type=str, required=True, help='Path to site players CSV')
    parser.add_argument('--team-priors', type=str, required=True, help='Path to team priors CSV')
    parser.add_argument('--player-priors', type=str, required=True, help='Path to player priors CSV')
    parser.add_argument('--boom-thresholds', type=str, required=True, help='Path to boom thresholds JSON')
    parser.add_argument('--sims', type=int, default=10000, help='Number of simulations (default: 10000)')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("NFL DFS SIMULATION RUNNER")
    logger.info("=" * 60)
    logger.info(f"Season: {args.season}, Week: {args.week}")
    logger.info(f"Simulations: {args.sims:,}")
    logger.info(f"Players file: {args.players_site}")
    logger.info(f"Output directory: {args.out}")
    
    try:
        # Step 1: Load site players
        logger.info("STEP 1: Loading site player data")
        players_df, column_mapping, validation_errors = load_site_players_csv(args.players_site)
        
        logger.info(f"Loaded {len(players_df)} players")
        logger.info(f"Column mapping: {column_mapping}")
        
        if validation_errors:
            logger.warning("Validation issues found:")
            for error in validation_errors:
                logger.warning(f"  - {error}")
        
        # Step 2: Load priors and thresholds
        logger.info("STEP 2: Loading priors and thresholds")
        team_priors, player_priors, boom_thresholds = load_priors_and_thresholds(
            args.team_priors, args.player_priors, args.boom_thresholds
        )
        
        # Step 3: Run simulation
        logger.info("STEP 3: Running simulation")
        if args.seed is not None:
            np.random.seed(args.seed)
            logger.info(f"Using random seed: {args.seed}")
        
        sim_results = run_simulation(
            players_df, team_priors, player_priors, args.sims
        )
        
        # Step 4: Calculate boom probabilities  
        logger.info("STEP 4: Calculating boom probabilities")
        site_fpts = players_df.set_index('player')['fpts'] if 'fpts' in players_df.columns else None
        sim_results = calculate_boom_probabilities(sim_results, boom_thresholds, site_fpts)
        
        # Step 5: Calculate value metrics
        logger.info("STEP 5: Calculating value and boom metrics")
        compare_df = create_comparison_output(sim_results, players_df)
        
        # Value metrics
        compare_df = calculate_value_metrics(compare_df, players_df)
        compare_df = calculate_salary_tiers(compare_df)
        compare_df = calculate_ceiling_analysis(compare_df)
        
        # Boom scores
        compare_df = calculate_boom_scores(compare_df, boom_thresholds)
        
        # Step 6: Diagnostics and flags
        logger.info("STEP 6: Running diagnostics")
        diagnostics = calculate_projection_diagnostics(sim_results, players_df)
        flags_df = identify_projection_flags(sim_results, players_df)
        
        # Validation
        validation_issues = validate_projection_reasonableness(sim_results)
        if validation_issues:
            logger.warning("Projection validation issues:")
            for issue in validation_issues:
                logger.warning(f"  - {issue}")
        
        # Step 7: Output results
        logger.info("STEP 7: Saving outputs")
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Main outputs
        sim_results.to_csv(output_dir / 'sim_players.csv', index=False)
        compare_df.to_csv(output_dir / 'compare.csv', index=False)
        
        # Export diagnostics
        if 'fpts' in players_df.columns:
            export_diagnostics_report(diagnostics, flags_df, str(output_dir))
        
        # Summary outputs
        dart_throws = compare_df[
            (compare_df.get('ownership', 100) <= 5) & 
            (compare_df['boom_score'] >= 70)
        ]
        
        if len(dart_throws) > 0:
            dart_throws.to_csv(output_dir / 'dart_throws.csv', index=False)
            logger.info(f"Identified {len(dart_throws)} dart throw candidates")
        
        value_plays = compare_df[compare_df.get('value_per_1k', 0) >= 3.0]
        if len(value_plays) > 0:
            value_plays.to_csv(output_dir / 'value_plays.csv', index=False)
            logger.info(f"Identified {len(value_plays)} value plays")
        
        # Step 8: Final summary
        logger.info("=" * 60)
        logger.info("SIMULATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Simulated {len(sim_results)} players")
        logger.info(f"Average projection: {sim_results['proj_mean'].mean():.1f}")
        logger.info(f"Boom score leaders:")
        
        top_boom = compare_df.nlargest(5, 'boom_score')[['player', 'position', 'boom_score', 'ownership']]
        for _, player in top_boom.iterrows():
            logger.info(f"  {player['player']} ({player['position']}): {player['boom_score']:.0f} boom, {player.get('ownership', 'N/A')}% owned")
        
        logger.info(f"\nOutput files saved to: {output_dir}")
        logger.info(f"  - sim_players.csv: {len(sim_results)} rows")
        logger.info(f"  - compare.csv: {len(compare_df)} rows")
        
        if len(flags_df) > 0:
            logger.info(f"  - flags.csv: {len(flags_df)} rows")
        
        logger.info("\nSimulation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()