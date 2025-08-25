"""
Value metrics calculations for fantasy football projections.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_value_per_1k(fantasy_points: float, salary: float) -> float:
    """
    Calculate value per $1K salary.
    
    Args:
        fantasy_points: Projected fantasy points
        salary: Player salary
        
    Returns:
        Value per $1K (points per 1000 salary)
    """
    if pd.isna(salary) or salary <= 0:
        return 0.0
    
    return (fantasy_points / salary) * 1000


def calculate_ceil_per_1k(ceiling_points: float, salary: float) -> float:
    """
    Calculate ceiling value per $1K salary.
    
    Args:
        ceiling_points: Ceiling projection (typically p90)
        salary: Player salary
        
    Returns:
        Ceiling value per $1K
    """
    if pd.isna(salary) or salary <= 0:
        return 0.0
    
    return (ceiling_points / salary) * 1000


def calculate_floor_per_1k(floor_points: float, salary: float) -> float:
    """
    Calculate floor value per $1K salary.
    
    Args:
        floor_points: Floor projection (typically p10)
        salary: Player salary
        
    Returns:
        Floor value per $1K
    """
    if pd.isna(salary) or salary <= 0:
        return 0.0
    
    return (floor_points / salary) * 1000


def calculate_site_value(site_projection: float, salary: float) -> float:
    """
    Calculate site projection value per $1K.
    
    Args:
        site_projection: Site fantasy projection
        salary: Player salary
        
    Returns:
        Site value per $1K
    """
    return calculate_value_per_1k(site_projection, salary)


def calculate_value_deltas(
    sim_mean: float,
    site_projection: float,
    salary: float
) -> Dict[str, float]:
    """
    Calculate value deltas between simulation and site projections.
    
    Args:
        sim_mean: Simulation mean projection
        site_projection: Site projection
        salary: Player salary
        
    Returns:
        Dictionary with delta calculations
    """
    deltas = {}
    
    # Point deltas
    deltas['delta_mean'] = sim_mean - site_projection if pd.notna(site_projection) else 0
    
    # Percentage delta
    if pd.notna(site_projection) and site_projection > 0:
        deltas['pct_delta'] = (deltas['delta_mean'] / site_projection) * 100
    else:
        deltas['pct_delta'] = 0
    
    # Value deltas
    sim_value = calculate_value_per_1k(sim_mean, salary)
    site_value = calculate_value_per_1k(site_projection, salary) if pd.notna(site_projection) else 0
    deltas['value_delta'] = sim_value - site_value
    
    return deltas


def add_value_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add value metrics to simulation results DataFrame.
    
    Args:
        df: DataFrame with simulation results
        
    Returns:
        DataFrame with added value metrics
    """
    result = df.copy()
    
    # Calculate value metrics
    result['value_per_1k'] = result.apply(
        lambda row: calculate_value_per_1k(row.get('sim_mean', 0), row.get('SAL', 0)),
        axis=1
    )
    
    result['ceil_per_1k'] = result.apply(
        lambda row: calculate_ceil_per_1k(row.get('ceiling_p90', 0), row.get('SAL', 0)),
        axis=1
    )
    
    result['floor_per_1k'] = result.apply(
        lambda row: calculate_floor_per_1k(row.get('floor_p10', 0), row.get('SAL', 0)),
        axis=1
    )
    
    # Site value if available
    if 'site_fpts' in result.columns:
        result['site_val'] = result.apply(
            lambda row: calculate_site_value(row.get('site_fpts', 0), row.get('SAL', 0)),
            axis=1
        )
    
    return result


def add_delta_metrics(df: pd.DataFrame, site_projection_col: str = 'site_fpts') -> pd.DataFrame:
    """
    Add delta metrics comparing simulation to site projections.
    
    Args:
        df: DataFrame with simulation results
        site_projection_col: Column name for site projections
        
    Returns:
        DataFrame with added delta metrics
    """
    result = df.copy()
    
    if site_projection_col in result.columns:
        # Calculate deltas for each row
        delta_results = result.apply(
            lambda row: calculate_value_deltas(
                row.get('sim_mean', 0),
                row.get(site_projection_col, np.nan),
                row.get('SAL', 0)
            ),
            axis=1
        )
        
        # Extract delta columns
        for delta_col in ['delta_mean', 'pct_delta', 'value_delta']:
            result[delta_col] = [deltas.get(delta_col, 0) for deltas in delta_results]
    
    return result


def calculate_salary_efficiency_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate salary efficiency metrics for the slate.
    
    Args:
        df: DataFrame with player data including salary and projections
        
    Returns:
        Dictionary with efficiency metrics
    """
    valid_data = df.dropna(subset=['SAL', 'sim_mean'])
    
    if len(valid_data) == 0:
        return {}
    
    # Calculate efficiency percentiles
    value_per_1k = valid_data.apply(
        lambda row: calculate_value_per_1k(row['sim_mean'], row['SAL']),
        axis=1
    )
    
    metrics = {
        'value_per_1k_p50': float(np.percentile(value_per_1k, 50)),
        'value_per_1k_p75': float(np.percentile(value_per_1k, 75)),
        'value_per_1k_p90': float(np.percentile(value_per_1k, 90)),
        'max_value_per_1k': float(value_per_1k.max()),
        'min_value_per_1k': float(value_per_1k.min())
    }
    
    return metrics


def get_value_thresholds_by_position() -> Dict[str, Dict[str, float]]:
    """
    Get typical value thresholds by position for GPP analysis.
    
    Returns:
        Dictionary with value thresholds by position
    """
    return {
        'QB': {
            'excellent': 4.5,
            'good': 3.5,
            'average': 3.0,
            'poor': 2.5
        },
        'RB': {
            'excellent': 4.0,
            'good': 3.5,
            'average': 3.0,
            'poor': 2.5
        },
        'WR': {
            'excellent': 4.0,
            'good': 3.5,
            'average': 3.0,
            'poor': 2.5
        },
        'TE': {
            'excellent': 4.5,
            'good': 3.5,
            'average': 3.0,
            'poor': 2.5
        },
        'DST': {
            'excellent': 3.5,
            'good': 3.0,
            'average': 2.5,
            'poor': 2.0
        },
        'K': {
            'excellent': 3.5,
            'good': 3.0,
            'average': 2.5,
            'poor': 2.0
        }
    }


def categorize_value(value_per_1k: float, position: str) -> str:
    """
    Categorize value per $1K into tiers.
    
    Args:
        value_per_1k: Value per $1K
        position: Player position
        
    Returns:
        Value category (Excellent, Good, Average, Poor)
    """
    thresholds = get_value_thresholds_by_position().get(position, {
        'excellent': 4.0, 'good': 3.5, 'average': 3.0, 'poor': 2.5
    })
    
    if value_per_1k >= thresholds['excellent']:
        return 'Excellent'
    elif value_per_1k >= thresholds['good']:
        return 'Good'
    elif value_per_1k >= thresholds['average']:
        return 'Average'
    else:
        return 'Poor'


def add_value_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add value category labels to DataFrame.
    
    Args:
        df: DataFrame with value_per_1k and POS columns
        
    Returns:
        DataFrame with value_category column
    """
    result = df.copy()
    
    if 'value_per_1k' in result.columns and 'POS' in result.columns:
        result['value_category'] = result.apply(
            lambda row: categorize_value(row.get('value_per_1k', 0), row.get('POS', '')),
            axis=1
        )
    
    return result