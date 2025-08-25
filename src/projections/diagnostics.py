"""
Diagnostic metrics for model validation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def calculate_mae(predictions: pd.Series, actuals: pd.Series) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        
    Returns:
        Mean Absolute Error
    """
    valid_pairs = _get_valid_pairs(predictions, actuals)
    if len(valid_pairs) == 0:
        return np.nan
    
    pred_vals, actual_vals = valid_pairs
    return float(np.mean(np.abs(pred_vals - actual_vals)))


def calculate_rmse(predictions: pd.Series, actuals: pd.Series) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        
    Returns:
        Root Mean Squared Error
    """
    valid_pairs = _get_valid_pairs(predictions, actuals)
    if len(valid_pairs) == 0:
        return np.nan
    
    pred_vals, actual_vals = valid_pairs
    return float(np.sqrt(np.mean((pred_vals - actual_vals) ** 2)))


def calculate_correlation(predictions: pd.Series, actuals: pd.Series) -> float:
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        
    Returns:
        Correlation coefficient
    """
    valid_pairs = _get_valid_pairs(predictions, actuals)
    if len(valid_pairs) == 0:
        return np.nan
    
    pred_vals, actual_vals = valid_pairs
    
    if len(pred_vals) < 2:
        return np.nan
    
    return float(np.corrcoef(pred_vals, actual_vals)[0, 1])


def calculate_coverage(
    actuals: pd.Series,
    lower_bounds: pd.Series,
    upper_bounds: pd.Series,
    confidence_level: float = 0.8
) -> float:
    """
    Calculate coverage percentage for confidence intervals.
    
    Args:
        actuals: Actual values
        lower_bounds: Lower confidence bounds (e.g., p10)
        upper_bounds: Upper confidence bounds (e.g., p90)
        confidence_level: Expected coverage level (e.g., 0.8 for 80%)
        
    Returns:
        Actual coverage percentage
    """
    # Get valid data where all three series have non-null values
    valid_mask = ~(actuals.isna() | lower_bounds.isna() | upper_bounds.isna())
    
    if valid_mask.sum() == 0:
        return np.nan
    
    valid_actuals = actuals[valid_mask]
    valid_lower = lower_bounds[valid_mask]
    valid_upper = upper_bounds[valid_mask]
    
    # Check how many actuals fall within bounds
    within_bounds = (valid_actuals >= valid_lower) & (valid_actuals <= valid_upper)
    coverage = within_bounds.mean()
    
    return float(coverage)


def _get_valid_pairs(series1: pd.Series, series2: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Get valid (non-null) pairs from two series."""
    valid_mask = ~(series1.isna() | series2.isna())
    return series1[valid_mask].values, series2[valid_mask].values


def calculate_position_diagnostics(
    df: pd.DataFrame,
    prediction_col: str = 'sim_mean',
    actual_col: str = 'actual_points',
    exclude_rookies: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Calculate diagnostic metrics by position.
    
    Args:
        df: DataFrame with predictions and actuals
        prediction_col: Column name for predictions
        actual_col: Column name for actual values
        exclude_rookies: Whether to exclude rookie players
        
    Returns:
        Dictionary with diagnostics by position
    """
    diagnostics = {}
    
    # Filter data
    analysis_df = df.copy()
    if exclude_rookies and 'rookie_fallback' in analysis_df.columns:
        analysis_df = analysis_df[~analysis_df['rookie_fallback']]
    
    # Calculate diagnostics for each position
    positions = analysis_df['POS'].unique() if 'POS' in analysis_df.columns else []
    
    for position in positions:
        pos_data = analysis_df[analysis_df['POS'] == position]
        
        if len(pos_data) == 0:
            continue
        
        pos_diagnostics = {}
        
        # Basic metrics
        if prediction_col in pos_data.columns and actual_col in pos_data.columns:
            pos_diagnostics['mae'] = calculate_mae(pos_data[prediction_col], pos_data[actual_col])
            pos_diagnostics['rmse'] = calculate_rmse(pos_data[prediction_col], pos_data[actual_col])
            pos_diagnostics['correlation'] = calculate_correlation(pos_data[prediction_col], pos_data[actual_col])
            pos_diagnostics['count'] = len(pos_data)
        
        # Coverage analysis
        if all(col in pos_data.columns for col in [actual_col, 'floor_p10', 'ceiling_p90']):
            pos_diagnostics['coverage_80'] = calculate_coverage(
                pos_data[actual_col], pos_data['floor_p10'], pos_data['ceiling_p90'], 0.8
            )
        
        diagnostics[position] = pos_diagnostics
    
    return diagnostics


def calculate_overall_diagnostics(
    df: pd.DataFrame,
    prediction_col: str = 'sim_mean',
    actual_col: str = 'actual_points',
    exclude_rookies: bool = True
) -> Dict[str, float]:
    """
    Calculate overall diagnostic metrics.
    
    Args:
        df: DataFrame with predictions and actuals
        prediction_col: Column name for predictions
        actual_col: Column name for actual values
        exclude_rookies: Whether to exclude rookie players
        
    Returns:
        Dictionary with overall diagnostics
    """
    # Filter data
    analysis_df = df.copy()
    if exclude_rookies and 'rookie_fallback' in analysis_df.columns:
        analysis_df = analysis_df[~analysis_df['rookie_fallback']]
    
    diagnostics = {}
    
    if prediction_col in analysis_df.columns and actual_col in analysis_df.columns:
        diagnostics['mae'] = calculate_mae(analysis_df[prediction_col], analysis_df[actual_col])
        diagnostics['rmse'] = calculate_rmse(analysis_df[prediction_col], analysis_df[actual_col])
        diagnostics['correlation'] = calculate_correlation(analysis_df[prediction_col], analysis_df[actual_col])
        diagnostics['total_count'] = len(analysis_df)
        
        if exclude_rookies and 'rookie_fallback' in df.columns:
            diagnostics['rookies_excluded'] = int(df['rookie_fallback'].sum())
    
    # Coverage analysis
    if all(col in analysis_df.columns for col in [actual_col, 'floor_p10', 'ceiling_p90']):
        diagnostics['coverage_80'] = calculate_coverage(
            analysis_df[actual_col], analysis_df['floor_p10'], analysis_df['ceiling_p90'], 0.8
        )
    
    return diagnostics


def generate_diagnostics_summary(
    df: pd.DataFrame,
    prediction_col: str = 'sim_mean',
    actual_col: str = 'actual_points',
    exclude_rookies: bool = True
) -> pd.DataFrame:
    """
    Generate a summary DataFrame with diagnostic metrics.
    
    Args:
        df: DataFrame with predictions and actuals
        prediction_col: Column name for predictions
        actual_col: Column name for actual values
        exclude_rookies: Whether to exclude rookie players
        
    Returns:
        Summary DataFrame with diagnostics
    """
    # Overall diagnostics
    overall = calculate_overall_diagnostics(df, prediction_col, actual_col, exclude_rookies)
    
    # Position diagnostics
    position_diags = calculate_position_diagnostics(df, prediction_col, actual_col, exclude_rookies)
    
    # Build summary DataFrame
    summary_data = []
    
    # Add overall row
    overall_row = {
        'Position': 'Overall',
        'Count': overall.get('total_count', 0),
        'MAE': overall.get('mae', np.nan),
        'RMSE': overall.get('rmse', np.nan),
        'Correlation': overall.get('correlation', np.nan),
        'Coverage_80': overall.get('coverage_80', np.nan)
    }
    summary_data.append(overall_row)
    
    # Add position rows
    for position, diags in position_diags.items():
        pos_row = {
            'Position': position,
            'Count': diags.get('count', 0),
            'MAE': diags.get('mae', np.nan),
            'RMSE': diags.get('rmse', np.nan),
            'Correlation': diags.get('correlation', np.nan),
            'Coverage_80': diags.get('coverage_80', np.nan)
        }
        summary_data.append(pos_row)
    
    return pd.DataFrame(summary_data)


def identify_extreme_deltas(
    df: pd.DataFrame,
    prediction_col: str = 'sim_mean',
    actual_col: str = 'actual_points',
    absolute_threshold: float = 10.0,
    percent_threshold: float = 50.0
) -> pd.DataFrame:
    """
    Identify players with extreme prediction errors.
    
    Args:
        df: DataFrame with predictions and actuals
        prediction_col: Column name for predictions
        actual_col: Column name for actual values
        absolute_threshold: Minimum absolute error threshold
        percent_threshold: Minimum percentage error threshold
        
    Returns:
        DataFrame with extreme delta cases
    """
    if prediction_col not in df.columns or actual_col not in df.columns:
        return pd.DataFrame()
    
    analysis_df = df.copy()
    
    # Calculate errors
    analysis_df['absolute_error'] = np.abs(analysis_df[prediction_col] - analysis_df[actual_col])
    analysis_df['percent_error'] = np.where(
        analysis_df[actual_col] != 0,
        np.abs(analysis_df['absolute_error'] / analysis_df[actual_col]) * 100,
        np.nan
    )
    
    # Filter extreme cases
    extreme_mask = (
        (analysis_df['absolute_error'] >= absolute_threshold) |
        (analysis_df['percent_error'] >= percent_threshold)
    )
    
    extreme_cases = analysis_df[extreme_mask].copy()
    
    # Sort by absolute error
    extreme_cases = extreme_cases.sort_values('absolute_error', ascending=False)
    
    # Select relevant columns
    output_cols = ['PLAYER', 'POS', 'TEAM', prediction_col, actual_col, 
                   'absolute_error', 'percent_error']
    
    available_cols = [col for col in output_cols if col in extreme_cases.columns]
    
    return extreme_cases[available_cols]


def calculate_bias_metrics(
    df: pd.DataFrame,
    prediction_col: str = 'sim_mean',
    actual_col: str = 'actual_points'
) -> Dict[str, float]:
    """
    Calculate bias metrics to detect systematic over/under-prediction.
    
    Args:
        df: DataFrame with predictions and actuals
        prediction_col: Column name for predictions
        actual_col: Column name for actual values
        
    Returns:
        Dictionary with bias metrics
    """
    if prediction_col not in df.columns or actual_col not in df.columns:
        return {}
    
    valid_data = df.dropna(subset=[prediction_col, actual_col])
    
    if len(valid_data) == 0:
        return {}
    
    predictions = valid_data[prediction_col]
    actuals = valid_data[actual_col]
    
    # Calculate bias metrics
    bias_metrics = {
        'mean_bias': float((predictions - actuals).mean()),
        'median_bias': float((predictions - actuals).median()),
        'mean_abs_bias': float(np.abs(predictions - actuals).mean()),
        'over_predictions': int((predictions > actuals).sum()),
        'under_predictions': int((predictions < actuals).sum()),
        'total_predictions': len(valid_data)
    }
    
    # Percentage of over/under predictions
    bias_metrics['pct_over_predictions'] = (bias_metrics['over_predictions'] / 
                                           bias_metrics['total_predictions']) * 100
    bias_metrics['pct_under_predictions'] = (bias_metrics['under_predictions'] / 
                                            bias_metrics['total_predictions']) * 100
    
    return bias_metrics


def create_diagnostic_flags(df: pd.DataFrame) -> List[str]:
    """
    Create diagnostic flags for data quality issues.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of diagnostic flag messages
    """
    flags = []
    
    # Check for missing data
    if 'PLAYER' in df.columns:
        missing_players = df['PLAYER'].isna().sum()
        if missing_players > 0:
            flags.append(f"{missing_players} players with missing names")
    
    # Check for extreme salaries
    if 'SAL' in df.columns:
        valid_salaries = df['SAL'].dropna()
        if len(valid_salaries) > 0:
            if valid_salaries.min() < 3000:
                flags.append("Unusually low salaries detected (<$3,000)")
            if valid_salaries.max() > 15000:
                flags.append("Unusually high salaries detected (>$15,000)")
    
    # Check for extreme projections
    if 'sim_mean' in df.columns:
        valid_projections = df['sim_mean'].dropna()
        if len(valid_projections) > 0:
            if valid_projections.max() > 50:
                flags.append("Unusually high projections detected (>50 points)")
            if valid_projections.min() < 0:
                flags.append("Negative projections detected")
    
    # Check rookie fallback rate
    if 'rookie_fallback' in df.columns:
        rookie_rate = df['rookie_fallback'].mean() * 100
        if rookie_rate > 30:
            flags.append(f"High rookie fallback rate: {rookie_rate:.1f}%")
    
    return flags