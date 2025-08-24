"""Run week simulation from site players CSV.

CLI: python -m src.projections.run_week_from_site_players --season 2025 --week 1 --players-site path/to/players_2025.csv --team-priors data/baseline/team_priors.csv --player-priors data/baseline/player_priors.csv --boom-thresholds data/baseline/boom_thresholds.json --sims 10000 --out data/sim_week
"""

import argparse
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ingest.site_players import load_site_players, validate_site_data
from src.sim.game_simulator import GameSimulator
from src.projections.value_metrics import calculate_value_metrics, calculate_positional_values, identify_value_plays
from src.projections.boom_score import calculate_boom_metrics, load_boom_thresholds, identify_boom_candidates
from src.projections.diagnostics import calculate_projection_diagnostics, identify_projection_flags, validate_simulation_results
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Run week simulation from site players CSV')
    parser.add_argument('--season', type=int, default=2025, help='Season year')
    parser.add_argument('--week', type=int, default=1, help='Week number')
    parser.add_argument('--players-site', type=str, required=True, help='Path to site players CSV')
    parser.add_argument('--team-priors', type=str, required=True, help='Path to team priors CSV')
    parser.add_argument('--player-priors', type=str, required=True, help='Path to player priors CSV')
    parser.add_argument('--boom-thresholds', type=str, required=True, help='Path to boom thresholds JSON')
    parser.add_argument('--sims', type=int, default=10000, help='Number of simulations')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Running week {args.week} simulation for {args.season} season...")
    print(f"Players file: {args.players_site}")
    print(f"Simulations: {args.sims}")
    print(f"Output directory: {args.out}")
    
    try:
        # Load site players data
        print("Loading site players data...")
        site_players = load_site_players(args.players_site)
        
        # Validate site data
        validation_issues = validate_site_data(site_players)
        if validation_issues:
            print("Site data validation warnings:")
            for issue in validation_issues:
                print(f"  - {issue}")
        
        print(f"Loaded {len(site_players)} players")
        
        # Load priors
        print("Loading team and player priors...")
        team_priors = pd.read_csv(args.team_priors)
        player_priors = pd.read_csv(args.player_priors)
        
        print(f"Loaded priors for {len(team_priors)} teams and {len(player_priors)} players")
        
        # Load boom thresholds
        print("Loading boom thresholds...")
        boom_thresholds = load_boom_thresholds(args.boom_thresholds)
        print(f"Loaded boom thresholds: {boom_thresholds}")
        
        # Initialize simulator
        print("Initializing game simulator...")
        simulator = GameSimulator(team_priors, player_priors)
        
        # Run simulations
        print(f"Running {args.sims} simulations...")
        sim_results = simulator.simulate_week(site_players, args.sims)
        
        # Validate simulation results
        print("Validating simulation results...")
        validation_issues = validate_simulation_results(sim_results)
        for issue in validation_issues:
            print(f"  - {issue}")
        
        print(f"Generated projections for {len(sim_results)} players")
        
        # Calculate value metrics
        print("Calculating value metrics...")
        sim_with_value = calculate_value_metrics(sim_results, site_players)
        sim_with_value = calculate_positional_values(sim_with_value)
        sim_with_value = identify_value_plays(sim_with_value)
        
        # Calculate boom metrics  
        print("Calculating boom metrics...")
        sim_with_boom = calculate_boom_metrics(sim_with_value, boom_thresholds, site_players)
        
        # Create compare dataframe (joins sim + site data)
        print("Creating comparison data...")
        compare_df = create_compare_dataframe(sim_with_boom, site_players)
        
        # Generate diagnostics
        print("Generating diagnostics...")
        diagnostics_df = calculate_projection_diagnostics(sim_with_boom, site_players)
        flags_df = identify_projection_flags(sim_with_boom, site_players)
        
        # Create output directory
        os.makedirs(args.out, exist_ok=True)
        
        # Save outputs
        print("Saving results...")
        
        # 1. sim_players.csv - core projections
        sim_players_cols = [
            'player_id', 'name', 'position', 'team', 'salary',
            'proj_mean', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'std',
            'floor', 'ceiling'
        ]
        available_sim_cols = [col for col in sim_players_cols if col in sim_with_boom.columns]
        sim_players_df = sim_with_boom[available_sim_cols]
        sim_players_df.to_csv(os.path.join(args.out, 'sim_players.csv'), index=False)
        
        # 2. compare.csv - full comparison with value and boom metrics
        compare_df.to_csv(os.path.join(args.out, 'compare.csv'), index=False)
        
        # 3. diagnostics_summary.csv
        diagnostics_df.to_csv(os.path.join(args.out, 'diagnostics_summary.csv'), index=False)
        
        # 4. flags.csv - projection mismatches
        flags_df.to_csv(os.path.join(args.out, 'flags.csv'), index=False)
        
        # 5. Summary JSON with key metrics
        summary = create_simulation_summary(sim_with_boom, compare_df, diagnostics_df, flags_df, args)
        with open(os.path.join(args.out, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("SIMULATION COMPLETE")
        print("="*50)
        print(f"Players simulated: {len(sim_with_boom)}")
        print(f"Value plays (>3.0 per $1k): {(sim_with_boom['value_per_1k'] > 3.0).sum() if 'value_per_1k' in sim_with_boom.columns else 'N/A'}")
        print(f"Dart flags (low owned + boom): {sim_with_boom['dart_flag'].sum() if 'dart_flag' in sim_with_boom.columns else 'N/A'}")
        print(f"Projection flags: {len(flags_df)}")
        print(f"\nFiles saved to {args.out}/")
        print("  - sim_players.csv")
        print("  - compare.csv") 
        print("  - diagnostics_summary.csv")
        print("  - flags.csv")
        print("  - summary.json")
        
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_compare_dataframe(sim_df: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive comparison DataFrame combining sim and site data.
    
    Args:
        sim_df: Simulation results with value and boom metrics
        site_df: Site players data
        
    Returns:
        Combined DataFrame with all relevant columns
    """
    # Start with simulation data
    compare = sim_df.copy()
    
    # Add site-specific columns that aren't already merged
    site_merge_cols = ['player_id', 'opponent', 'total', 'spread', 'ownership']
    if 'site_proj' in site_df.columns:
        site_merge_cols.append('site_proj')
    
    # Only merge columns that don't already exist
    existing_cols = set(compare.columns)
    new_site_cols = [col for col in site_merge_cols if col not in existing_cols and col in site_df.columns]
    
    if new_site_cols:
        site_subset = site_df[['player_id'] + [col for col in new_site_cols if col != 'player_id']]
        compare = compare.merge(site_subset, on='player_id', how='left')
    
    # Reorder columns for readability
    col_order = [
        'player_id', 'name', 'position', 'team', 'opponent', 'salary',
        'proj_mean', 'p10', 'p50', 'p90', 'p95', 'ceiling', 'floor',
        'value_per_1k', 'ceil_per_1k', 'boom_prob', 'boom_score', 'dart_flag'
    ]
    
    # Add site comparison columns if available
    if 'site_proj' in compare.columns:
        col_order.extend(['site_proj', 'delta_vs_site', 'pct_delta_vs_site', 'beat_site_prob'])
    
    # Add ownership if available
    if 'ownership' in compare.columns:
        col_order.append('ownership')
    
    # Filter to available columns and reorder
    available_cols = [col for col in col_order if col in compare.columns]
    remaining_cols = [col for col in compare.columns if col not in available_cols]
    
    final_cols = available_cols + remaining_cols
    
    return compare[final_cols]


def create_simulation_summary(sim_df: pd.DataFrame, compare_df: pd.DataFrame, 
                            diagnostics_df: pd.DataFrame, flags_df: pd.DataFrame,
                            args) -> dict:
    """Create summary statistics for the simulation run.
    
    Args:
        sim_df: Simulation results
        compare_df: Comparison data
        diagnostics_df: Diagnostics results
        flags_df: Flagged projections
        args: Command line arguments
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'run_info': {
            'season': args.season,
            'week': args.week,
            'simulations': args.sims,
            'players_simulated': len(sim_df),
            'total_salary': sim_df['salary'].sum() if 'salary' in sim_df.columns else None,
        },
        'projections': {
            'total_projected_points': sim_df['proj_mean'].sum(),
            'avg_projection': sim_df['proj_mean'].mean(),
            'highest_projection': {
                'player': sim_df.loc[sim_df['proj_mean'].idxmax(), 'name'] if len(sim_df) > 0 else None,
                'points': sim_df['proj_mean'].max(),
            },
            'by_position': sim_df.groupby('position').agg({
                'proj_mean': ['count', 'mean', 'max'],
                'salary': 'mean'
            }).round(2).to_dict() if len(sim_df) > 0 else {},
        },
        'value_analysis': {},
        'boom_analysis': {},
        'diagnostics': {},
        'flags': {
            'total_flags': len(flags_df),
            'flag_types': flags_df['flag_reason'].value_counts().to_dict() if len(flags_df) > 0 else {},
        }
    }
    
    # Value metrics
    if 'value_per_1k' in sim_df.columns:
        summary['value_analysis'] = {
            'avg_value_per_1k': sim_df['value_per_1k'].mean(),
            'top_value_plays': sim_df.nlargest(5, 'value_per_1k')[['name', 'position', 'value_per_1k']].to_dict('records'),
            'value_plays_count': (sim_df['value_per_1k'] > 3.0).sum(),
        }
    
    # Boom metrics
    if 'boom_score' in sim_df.columns:
        summary['boom_analysis'] = {
            'avg_boom_score': sim_df['boom_score'].mean(),
            'dart_flags_count': sim_df['dart_flag'].sum() if 'dart_flag' in sim_df.columns else 0,
            'top_boom_scores': sim_df.nlargest(5, 'boom_score')[['name', 'position', 'boom_score', 'boom_prob']].to_dict('records'),
        }
    
    # Diagnostics summary
    if len(diagnostics_df) > 0:
        overall_diag = diagnostics_df[diagnostics_df['position'] == 'ALL']
        if len(overall_diag) > 0:
            overall = overall_diag.iloc[0]
            summary['diagnostics'] = {
                'players_compared': overall.get('player_count', 0),
                'correlation': overall.get('correlation'),
                'mae': overall.get('mae'),
                'rmse': overall.get('rmse'),
                'coverage_p10_p90': overall.get('coverage_p10_p90'),
            }
    
    return summary


if __name__ == '__main__':
    main()