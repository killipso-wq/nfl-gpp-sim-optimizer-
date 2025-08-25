"""
CLI entry point for running weekly simulations from site players.csv
"""
import argparse
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import warnings

import pandas as pd
import numpy as np

from ..ingest.site_players import load_site_players
from ..sim.game_simulator import GameSimulator, get_position_variance_defaults
from ..projections.value_metrics import add_value_metrics, add_delta_metrics
from ..projections.boom_score import add_boom_scores
from ..projections.diagnostics import generate_diagnostics_summary, identify_extreme_deltas, create_diagnostic_flags


def load_priors_and_thresholds(
    team_priors_path: str,
    player_priors_path: str,
    boom_thresholds_path: str
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Load baseline priors and boom thresholds.
    
    Args:
        team_priors_path: Path to team priors CSV
        player_priors_path: Path to player priors CSV
        boom_thresholds_path: Path to boom thresholds JSON
        
    Returns:
        Tuple of (team_priors_df, player_priors_df, boom_thresholds_dict)
    """
    # Load team priors
    team_priors = pd.DataFrame()
    if os.path.exists(team_priors_path):
        team_priors = pd.read_csv(team_priors_path)
    else:
        warnings.warn(f"Team priors file not found: {team_priors_path}")
    
    # Load player priors
    player_priors = pd.DataFrame()
    if os.path.exists(player_priors_path):
        player_priors = pd.read_csv(player_priors_path)
    else:
        warnings.warn(f"Player priors file not found: {player_priors_path}")
    
    # Load boom thresholds
    boom_thresholds = {}
    if os.path.exists(boom_thresholds_path):
        with open(boom_thresholds_path, 'r') as f:
            boom_thresholds = json.load(f)
    else:
        warnings.warn(f"Boom thresholds file not found: {boom_thresholds_path}")
        # Use default thresholds
        boom_thresholds = {
            'QB': 25.0, 'RB': 20.0, 'WR': 18.0, 'TE': 15.0, 'DST': 12.0, 'K': 10.0
        }
    
    return team_priors, player_priors, boom_thresholds


def run_simulation(
    players_df: pd.DataFrame,
    team_priors: pd.DataFrame,
    player_priors: pd.DataFrame,
    boom_thresholds: Dict[str, float],
    n_sims: int = 10000,
    seed: int = None
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation for all players.
    
    Args:
        players_df: Site players data
        team_priors: Team historical data
        player_priors: Player historical data
        boom_thresholds: Boom thresholds by position
        n_sims: Number of simulations
        seed: Random seed
        
    Returns:
        DataFrame with simulation results
    """
    # Initialize simulator
    simulator = GameSimulator(seed=seed)
    
    # Get position variance defaults
    position_variance = get_position_variance_defaults()
    
    # Run simulation
    sim_results = simulator.simulate_slate(
        players_df=players_df,
        player_priors=player_priors,
        team_priors=team_priors,
        boom_thresholds=boom_thresholds,
        position_variance=position_variance,
        n_sims=n_sims
    )
    
    return sim_results


def create_compare_df(sim_results: pd.DataFrame, original_players: pd.DataFrame) -> pd.DataFrame:
    """
    Create comparison DataFrame joining simulation results with original site data.
    
    Args:
        sim_results: Simulation results
        original_players: Original site players data
        
    Returns:
        Combined comparison DataFrame
    """
    # Start with simulation results
    compare_df = sim_results.copy()
    
    # Add site fields if available
    site_columns = ['FPTS', 'RST%', 'O/U', 'SPRD', 'ML', 'TM/P', 'VAL']
    
    for col in site_columns:
        if col in original_players.columns:
            # Merge on player_id
            if 'player_id' in compare_df.columns and 'player_id' in original_players.columns:
                site_data = original_players[['player_id', col]].drop_duplicates('player_id')
                compare_df = compare_df.merge(site_data, on='player_id', how='left', suffixes=('', '_site'))
            
            # Rename FPTS to site_fpts for clarity
            if col == 'FPTS':
                compare_df = compare_df.rename(columns={'FPTS': 'site_fpts'})
    
    # Add value metrics
    compare_df = add_value_metrics(compare_df)
    
    # Add delta metrics if site projection available
    if 'site_fpts' in compare_df.columns:
        compare_df = add_delta_metrics(compare_df, 'site_fpts')
    
    # Add boom scores
    compare_df = add_boom_scores(compare_df)
    
    return compare_df


def create_output_files(
    sim_results: pd.DataFrame,
    compare_df: pd.DataFrame,
    original_players: pd.DataFrame,
    metadata: Dict[str, Any],
    output_dir: str
) -> Dict[str, str]:
    """
    Create all output files.
    
    Args:
        sim_results: Simulation results
        compare_df: Comparison DataFrame
        original_players: Original players data
        metadata: Run metadata
        output_dir: Output directory
        
    Returns:
        Dictionary of created file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    file_paths = {}
    
    # 1. sim_players.csv - Core simulation results
    sim_output_cols = [
        'player_id', 'PLAYER', 'POS', 'TEAM', 'OPP', 'sim_mean', 
        'floor_p10', 'p75', 'ceiling_p90', 'p95', 'boom_prob', 'rookie_fallback'
    ]
    
    # Add SAL if available
    if 'SAL' in sim_results.columns:
        sim_output_cols.insert(-1, 'SAL')  # Insert before rookie_fallback
    
    available_sim_cols = [col for col in sim_output_cols if col in sim_results.columns]
    sim_players = sim_results[available_sim_cols]
    
    sim_players_path = os.path.join(output_dir, 'sim_players.csv')
    sim_players.to_csv(sim_players_path, index=False)
    file_paths['sim_players'] = sim_players_path
    
    # 2. compare.csv - Full comparison
    compare_path = os.path.join(output_dir, 'compare.csv')
    compare_df.to_csv(compare_path, index=False)
    file_paths['compare'] = compare_path
    
    # 3. diagnostics_summary.csv - Only if we have actual data
    # For MVP, we'll create a placeholder structure
    diagnostics_data = [{
        'Position': 'Overall',
        'Count': len(sim_results),
        'MAE': None,  # Would need actual data
        'RMSE': None,
        'Correlation': None,
        'Coverage_80': None,
        'Note': 'Requires actual game results for validation'
    }]
    
    diagnostics_df = pd.DataFrame(diagnostics_data)
    diagnostics_path = os.path.join(output_dir, 'diagnostics_summary.csv')
    diagnostics_df.to_csv(diagnostics_path, index=False)
    file_paths['diagnostics'] = diagnostics_path
    
    # 4. flags.csv - Data quality flags and extreme deltas
    flags = create_diagnostic_flags(compare_df)
    
    # Add top absolute and percentage deltas if available
    flag_data = []
    for flag in flags:
        flag_data.append({'Type': 'Data Quality', 'Description': flag})
    
    # Add top delta cases if we have site projections
    if 'delta_mean' in compare_df.columns:
        top_deltas = compare_df.nlargest(5, 'delta_mean')[['PLAYER', 'POS', 'sim_mean', 'site_fpts', 'delta_mean']]
        for _, row in top_deltas.iterrows():
            flag_data.append({
                'Type': 'High Positive Delta',
                'Description': f"{row['PLAYER']} ({row['POS']}): Sim {row['sim_mean']:.1f} vs Site {row['site_fpts']:.1f} (+{row['delta_mean']:.1f})"
            })
        
        bottom_deltas = compare_df.nsmallest(5, 'delta_mean')[['PLAYER', 'POS', 'sim_mean', 'site_fpts', 'delta_mean']]
        for _, row in bottom_deltas.iterrows():
            flag_data.append({
                'Type': 'High Negative Delta',
                'Description': f"{row['PLAYER']} ({row['POS']}): Sim {row['sim_mean']:.1f} vs Site {row['site_fpts']:.1f} ({row['delta_mean']:.1f})"
            })
    
    flags_df = pd.DataFrame(flag_data)
    flags_path = os.path.join(output_dir, 'flags.csv')
    flags_df.to_csv(flags_path, index=False)
    file_paths['flags'] = flags_path
    
    # 5. metadata.json
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    file_paths['metadata'] = metadata_path
    
    return file_paths


def create_zip_bundle(file_paths: Dict[str, str], output_dir: str) -> str:
    """
    Create ZIP bundle with all outputs.
    
    Args:
        file_paths: Dictionary of file paths
        output_dir: Output directory
        
    Returns:
        Path to created ZIP file
    """
    zip_path = os.path.join(output_dir, 'simulator_outputs.zip')
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_type, file_path in file_paths.items():
            if os.path.exists(file_path):
                # Use just filename in ZIP
                zip_file.write(file_path, os.path.basename(file_path))
    
    return zip_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Run weekly NFL fantasy simulation')
    
    parser.add_argument('--season', type=int, required=True, help='Season year (e.g., 2025)')
    parser.add_argument('--week', type=int, required=True, help='Week number')
    parser.add_argument('--players-site', required=True, help='Path to site players.csv file')
    parser.add_argument('--team-priors', required=True, help='Path to team priors CSV')
    parser.add_argument('--player-priors', required=True, help='Path to player priors CSV')
    parser.add_argument('--boom-thresholds', required=True, help='Path to boom thresholds JSON')
    parser.add_argument('--sims', type=int, default=10000, help='Number of simulations (default: 10000)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--out', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Loading site players from {args.players_site}...")
    
    # Load site players
    try:
        players_df, column_mapping, warnings_list = load_site_players(args.players_site)
        print(f"Loaded {len(players_df)} players")
        
        if warnings_list:
            print("Warnings:")
            for warning in warnings_list:
                print(f"  - {warning}")
    except Exception as e:
        print(f"Error loading site players: {e}")
        return 1
    
    print("Loading priors and thresholds...")
    
    # Load priors and thresholds
    try:
        team_priors, player_priors, boom_thresholds = load_priors_and_thresholds(
            args.team_priors, args.player_priors, args.boom_thresholds
        )
        print(f"Loaded {len(team_priors)} team priors, {len(player_priors)} player priors")
    except Exception as e:
        print(f"Error loading priors: {e}")
        return 1
    
    print(f"Running {args.sims} simulations with seed {args.seed}...")
    
    # Run simulation
    try:
        sim_results = run_simulation(
            players_df=players_df,
            team_priors=team_priors,
            player_priors=player_priors,
            boom_thresholds=boom_thresholds,
            n_sims=args.sims,
            seed=args.seed
        )
        print(f"Simulation complete for {len(sim_results)} players")
    except Exception as e:
        print(f"Error running simulation: {e}")
        return 1
    
    print("Creating comparison data and outputs...")
    
    # Create comparison DataFrame
    compare_df = create_compare_df(sim_results, players_df)
    
    # Create metadata
    metadata = {
        'season': args.season,
        'week': args.week,
        'generated_at_utc': datetime.utcnow().isoformat() + 'Z',
        'sims': args.sims,
        'seed': args.seed,
        'players_count': len(players_df),
        'column_mapping': column_mapping,
        'boom_thresholds': boom_thresholds,
        'warnings': warnings_list
    }
    
    # Create output files
    try:
        file_paths = create_output_files(
            sim_results=sim_results,
            compare_df=compare_df,
            original_players=players_df,
            metadata=metadata,
            output_dir=args.out
        )
        
        # Create ZIP bundle
        zip_path = create_zip_bundle(file_paths, args.out)
        file_paths['zip'] = zip_path
        
        print(f"Output files created in {args.out}:")
        for file_type, file_path in file_paths.items():
            print(f"  - {file_type}: {os.path.basename(file_path)}")
            
    except Exception as e:
        print(f"Error creating outputs: {e}")
        return 1
    
    print("Simulation complete!")
    return 0


if __name__ == '__main__':
    exit(main())