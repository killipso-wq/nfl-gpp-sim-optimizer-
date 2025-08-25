"""
Build boom thresholds from historical distribution data.

This script calculates position-specific boom thresholds (default p90)
from historical fantasy point distributions.
"""
import argparse
import json
import os
from typing import Dict

import pandas as pd
import numpy as np

# NOTE: Would import nfl_data_py when available
# import nfl_data_py as nfl


def fetch_historical_fantasy_data(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch historical fantasy point data.
    
    Args:
        start_year: Starting season year
        end_year: Ending season year
        
    Returns:
        DataFrame with historical fantasy points
    """
    # NOTE: This would use nfl_data_py when available
    # For MVP, we'll create mock data to demonstrate the structure
    
    print("NOTE: This script requires nfl_data_py package for historical data.")
    print("Creating mock data structure for demonstration...")
    
    # Mock historical fantasy data
    # This would be replaced with actual nfl_data_py calls:
    # years = list(range(start_year, end_year + 1))
    # weekly_data = nfl.import_weekly_data(years)
    
    # Generate mock data with realistic distributions
    np.random.seed(42)  # For reproducible mock data
    
    positions = ['QB', 'RB', 'WR', 'TE', 'DST', 'K']
    n_games_per_position = 1000  # Mock sample size
    
    mock_data = []
    
    # Position-specific distributions for mock data
    position_params = {
        'QB': {'mean': 18.5, 'std': 7.2},
        'RB': {'mean': 12.8, 'std': 6.8},
        'WR': {'mean': 11.4, 'std': 7.1},
        'TE': {'mean': 8.9, 'std': 5.2},
        'DST': {'mean': 8.1, 'std': 6.9},
        'K': {'mean': 7.3, 'std': 3.8}
    }
    
    for position in positions:
        params = position_params[position]
        
        # Generate samples with realistic skew
        if position in ['QB', 'WR', 'DST']:
            # Right-skewed distributions for boom/bust positions
            samples = np.random.lognormal(
                mean=np.log(params['mean']**2 / np.sqrt(params['std']**2 + params['mean']**2)),
                sigma=np.sqrt(np.log(1 + params['std']**2 / params['mean']**2)),
                size=n_games_per_position
            )
        else:
            # Normal distributions for more consistent positions
            samples = np.random.normal(params['mean'], params['std'], n_games_per_position)
        
        # Clamp negative values
        samples = np.maximum(samples, 0)
        
        for i, fantasy_points in enumerate(samples):
            mock_data.append({
                'position': position,
                'fantasy_points': fantasy_points,
                'season': np.random.choice([start_year, end_year]),
                'week': np.random.randint(1, 18),
                'player_id': f'MOCK_{position}_{i:04d}'
            })
    
    return pd.DataFrame(mock_data)


def calculate_boom_thresholds(
    fantasy_data: pd.DataFrame, 
    quantile: float = 0.90,
    min_samples: int = 50
) -> Dict[str, float]:
    """
    Calculate boom thresholds by position.
    
    Args:
        fantasy_data: DataFrame with fantasy points by position
        quantile: Quantile to use for boom threshold (default 0.90 = p90)
        min_samples: Minimum samples required per position
        
    Returns:
        Dictionary with boom thresholds by position
    """
    thresholds = {}
    
    # Calculate threshold for each position
    for position in fantasy_data['position'].unique():
        pos_data = fantasy_data[fantasy_data['position'] == position]['fantasy_points']
        
        if len(pos_data) < min_samples:
            print(f"Warning: Only {len(pos_data)} samples for {position}, using default")
            # Use reasonable defaults if insufficient data
            defaults = {
                'QB': 28.0, 'RB': 22.0, 'WR': 20.0, 'TE': 16.0, 'DST': 18.0, 'K': 12.0
            }
            thresholds[position] = defaults.get(position, 20.0)
        else:
            threshold = float(np.percentile(pos_data, quantile * 100))
            thresholds[position] = round(threshold, 1)
    
    return thresholds


def validate_thresholds(thresholds: Dict[str, float]) -> list:
    """
    Validate boom thresholds and return warnings.
    
    Args:
        thresholds: Dictionary of boom thresholds
        
    Returns:
        List of warning messages
    """
    warnings_list = []
    
    # Expected ranges for boom thresholds
    expected_ranges = {
        'QB': (20, 35),
        'RB': (18, 30),
        'WR': (15, 28),
        'TE': (12, 22),
        'DST': (12, 25),
        'K': (8, 16)
    }
    
    for position, threshold in thresholds.items():
        if position in expected_ranges:
            min_val, max_val = expected_ranges[position]
            if threshold < min_val or threshold > max_val:
                warnings_list.append(
                    f"{position} threshold {threshold} outside expected range [{min_val}, {max_val}]"
                )
    
    # Check for missing common positions
    common_positions = ['QB', 'RB', 'WR', 'TE', 'DST']
    missing_positions = [pos for pos in common_positions if pos not in thresholds]
    if missing_positions:
        warnings_list.append(f"Missing thresholds for positions: {missing_positions}")
    
    return warnings_list


def save_boom_thresholds(thresholds: Dict[str, float], output_path: str) -> str:
    """
    Save boom thresholds to JSON file.
    
    Args:
        thresholds: Dictionary of boom thresholds
        output_path: Output file path
        
    Returns:
        Path to saved file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create output with metadata
    output_data = {
        'boom_thresholds': thresholds,
        'metadata': {
            'description': 'Position-specific boom thresholds for fantasy football',
            'calculation_method': 'Historical percentile analysis',
            'created_at': pd.Timestamp.now().isoformat()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return output_path


def print_threshold_summary(thresholds: Dict[str, float], quantile: float):
    """Print summary of calculated thresholds."""
    print(f"\nBoom Thresholds (p{int(quantile*100)}):")
    print("-" * 30)
    
    # Sort by position for consistent display
    sorted_positions = sorted(thresholds.keys())
    
    for position in sorted_positions:
        threshold = thresholds[position]
        print(f"{position:>3}: {threshold:5.1f} points")
    
    print(f"\nMean threshold: {np.mean(list(thresholds.values())):.1f}")
    print(f"Range: {min(thresholds.values()):.1f} - {max(thresholds.values()):.1f}")


def analyze_threshold_distribution(fantasy_data: pd.DataFrame, thresholds: Dict[str, float]):
    """Analyze what percentage of performances exceed boom thresholds."""
    print(f"\nBoom Threshold Analysis:")
    print("-" * 40)
    
    for position, threshold in thresholds.items():
        pos_data = fantasy_data[fantasy_data['position'] == position]['fantasy_points']
        
        if len(pos_data) > 0:
            boom_rate = (pos_data >= threshold).mean() * 100
            avg_points = pos_data.mean()
            print(f"{position:>3}: {boom_rate:5.1f}% exceed {threshold:5.1f} (avg: {avg_points:5.1f})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Build boom thresholds from historical data')
    parser.add_argument('--start', type=int, required=True, help='Start year (e.g., 2023)')
    parser.add_argument('--end', type=int, required=True, help='End year (e.g., 2024)')
    parser.add_argument('--quantile', type=float, default=0.90, help='Quantile for threshold (default: 0.90)')
    parser.add_argument('--out', required=True, help='Output JSON file path')
    parser.add_argument('--min-samples', type=int, default=50, help='Minimum samples per position')
    
    args = parser.parse_args()
    
    # Validate quantile
    if not 0.5 <= args.quantile <= 0.99:
        print("Error: Quantile must be between 0.5 and 0.99")
        return 1
    
    print(f"Calculating boom thresholds for {args.start}-{args.end} seasons...")
    print(f"Using p{int(args.quantile*100)} quantile")
    
    # Fetch historical data
    try:
        fantasy_data = fetch_historical_fantasy_data(args.start, args.end)
        print(f"Loaded {len(fantasy_data)} fantasy performance records")
        
        position_counts = fantasy_data['position'].value_counts()
        print(f"Positions: {dict(position_counts)}")
        
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return 1
    
    # Calculate boom thresholds
    try:
        thresholds = calculate_boom_thresholds(
            fantasy_data, 
            quantile=args.quantile,
            min_samples=args.min_samples
        )
        
    except Exception as e:
        print(f"Error calculating thresholds: {e}")
        return 1
    
    # Validate thresholds
    warnings_list = validate_thresholds(thresholds)
    if warnings_list:
        print("\nWarnings:")
        for warning in warnings_list:
            print(f"  - {warning}")
    
    # Save thresholds
    try:
        output_path = save_boom_thresholds(thresholds, args.out)
        print(f"\nSaved boom thresholds to: {output_path}")
        
    except Exception as e:
        print(f"Error saving thresholds: {e}")
        return 1
    
    # Print summaries
    print_threshold_summary(thresholds, args.quantile)
    analyze_threshold_distribution(fantasy_data, thresholds)
    
    print(f"\nNOTE: This script requires 'nfl_data_py' package for production use.")
    print("Install with: pip install nfl_data_py")
    print("Mock data was generated for demonstration purposes.")
    
    return 0


if __name__ == '__main__':
    exit(main())