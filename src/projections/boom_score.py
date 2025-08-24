"""
Boom score calculation combining boom probability, ownership, and value.

This module calculates the "Boom Score" (1-100) that identifies players
with high upside potential, considering boom probability, low ownership,
and salary efficiency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_boom_scores(df: pd.DataFrame, 
                         boom_thresholds: Dict[str, float],
                         ownership_boost: float = 2.0,
                         value_boost: float = 1.5) -> pd.DataFrame:
    """
    Calculate boom scores for all players.
    
    Args:
        df: DataFrame with projections and site data
        boom_thresholds: Dictionary mapping position to boom threshold
        ownership_boost: Multiplier for low ownership (default 2.0)
        value_boost: Multiplier for high value (default 1.5) 
        
    Returns:
        DataFrame with boom scores added
    """
    logger.info("Calculating boom scores")
    
    result_df = df.copy()
    
    # Calculate boom probability if not already present
    if 'boom_prob' not in result_df.columns:
        result_df = _calculate_boom_probability(result_df, boom_thresholds)
    
    # Calculate base boom score
    result_df['boom_score_base'] = result_df['boom_prob'].copy()
    
    # Apply ownership boost (low ownership gets higher score)
    if 'ownership' in result_df.columns:
        result_df = _apply_ownership_boost(result_df, ownership_boost)
    else:
        result_df['boom_score_with_ownership'] = result_df['boom_score_base']
        logger.warning("No ownership data available, skipping ownership boost")
    
    # Apply value boost (high value gets higher score)
    if 'value_per_1k' in result_df.columns:
        result_df = _apply_value_boost(result_df, value_boost)
    else:
        result_df['boom_score_with_value'] = result_df['boom_score_with_ownership']
        logger.warning("No value data available, skipping value boost")
    
    # Final boom score (scaled to 0-100)
    result_df['boom_score'] = _scale_boom_score(result_df['boom_score_with_value'])
    
    # Identify dart throw candidates
    result_df = _identify_dart_throws(result_df)
    
    logger.info(f"Calculated boom scores for {len(result_df)} players")
    
    return result_df


def _calculate_boom_probability(df: pd.DataFrame, 
                               boom_thresholds: Dict[str, float]) -> pd.DataFrame:
    """Calculate boom probability based on position thresholds."""
    
    def get_boom_prob(row):
        position = row.get('position', 'WR')
        threshold = boom_thresholds.get(position, 15.0)
        
        # Use site projection if available, otherwise our projection
        site_fpts = row.get('fpts', np.nan)
        our_proj = row.get('proj_mean', 0)
        ceiling = row.get('ceiling', our_proj * 1.5)
        
        # Calculate threshold (max of position threshold, 1.2x site, or site+5)
        effective_threshold = threshold
        if pd.notna(site_fpts):
            effective_threshold = max(threshold, site_fpts * 1.2, site_fpts + 5)
        
        # Estimate boom probability based on our ceiling
        if ceiling > effective_threshold:
            # Simple heuristic: prob increases with how much ceiling exceeds threshold
            excess_ratio = (ceiling - effective_threshold) / effective_threshold
            boom_prob = min(50, 10 + excess_ratio * 20)  # Cap at 50%
        else:
            # Low boom probability if ceiling doesn't exceed threshold
            boom_prob = max(1, (ceiling / effective_threshold) * 8)
        
        return boom_prob
    
    df['boom_prob'] = df.apply(get_boom_prob, axis=1).round(1)
    
    return df


def _apply_ownership_boost(df: pd.DataFrame, boost_factor: float) -> pd.DataFrame:
    """Apply ownership boost to boom score."""
    
    def ownership_multiplier(ownership):
        if pd.isna(ownership):
            return 1.0
        
        # Low ownership gets bigger boost
        if ownership <= 3:
            return boost_factor
        elif ownership <= 5:
            return boost_factor * 0.8
        elif ownership <= 10:
            return boost_factor * 0.6
        elif ownership <= 15:
            return 1.2
        else:
            return 1.0  # No boost for high ownership
    
    df['ownership_multiplier'] = df['ownership'].apply(ownership_multiplier)
    df['boom_score_with_ownership'] = (
        df['boom_score_base'] * df['ownership_multiplier']
    ).round(1)
    
    return df


def _apply_value_boost(df: pd.DataFrame, boost_factor: float) -> pd.DataFrame:
    """Apply value boost to boom score."""
    
    def value_multiplier(value_per_1k):
        if pd.isna(value_per_1k):
            return 1.0
        
        # High value gets boost
        if value_per_1k >= 3.5:
            return boost_factor
        elif value_per_1k >= 3.0:
            return boost_factor * 0.8
        elif value_per_1k >= 2.5:
            return boost_factor * 0.6
        elif value_per_1k >= 2.0:
            return 1.1
        else:
            return 1.0  # No boost for low value
    
    df['value_multiplier'] = df['value_per_1k'].apply(value_multiplier)
    df['boom_score_with_value'] = (
        df['boom_score_with_ownership'] * df['value_multiplier']
    ).round(1)
    
    return df


def _scale_boom_score(raw_scores: pd.Series) -> pd.Series:
    """Scale boom scores to 0-100 range."""
    
    if raw_scores.empty:
        return raw_scores
    
    # Use percentile-based scaling to ensure good distribution
    min_score = raw_scores.min()
    max_score = raw_scores.max()
    
    if max_score == min_score:
        return pd.Series([50] * len(raw_scores), index=raw_scores.index)
    
    # Scale to 0-100 with some floor/ceiling adjustments
    scaled = ((raw_scores - min_score) / (max_score - min_score) * 90) + 5
    
    # Ensure we use the full range for differentiation
    scaled = scaled.clip(1, 100).round(1)
    
    return scaled


def _identify_dart_throws(df: pd.DataFrame) -> pd.DataFrame:
    """Identify dart throw candidates."""
    
    # Dart throws: low ownership + high boom score
    ownership_threshold = 5.0
    boom_score_threshold = 70.0
    
    dart_conditions = (
        (df.get('ownership', 100) <= ownership_threshold) &
        (df['boom_score'] >= boom_score_threshold)
    )
    
    df['dart_flag'] = dart_conditions
    
    return df


def identify_contrarian_plays(df: pd.DataFrame,
                             ownership_threshold: float = 8.0,
                             projection_threshold: float = None) -> pd.DataFrame:
    """
    Identify contrarian plays (low ownership, decent projection).
    
    Args:
        df: DataFrame with projections and ownership
        ownership_threshold: Maximum ownership to be considered contrarian
        projection_threshold: Optional minimum projection threshold
        
    Returns:
        DataFrame filtered to contrarian plays
    """
    logger.info(f"Identifying contrarian plays (ownership <= {ownership_threshold}%)")
    
    if 'ownership' not in df.columns:
        logger.warning("No ownership data available")
        return df
    
    # Base ownership filter
    contrarian = df[df['ownership'] <= ownership_threshold].copy()
    
    # Optional projection filter
    if projection_threshold is not None:
        contrarian = contrarian[contrarian['proj_mean'] >= projection_threshold]
    
    # Sort by boom score if available, otherwise by projection
    sort_col = 'boom_score' if 'boom_score' in contrarian.columns else 'proj_mean'
    contrarian = contrarian.sort_values(sort_col, ascending=False)
    
    logger.info(f"Found {len(contrarian)} contrarian plays")
    
    return contrarian


def calculate_leverage_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate leverage scores for tournament play.
    
    Leverage = (Our Projection - Site Projection) * (1 / Ownership)
    
    Args:
        df: DataFrame with projections and site data
        
    Returns:
        DataFrame with leverage scores added
    """
    logger.info("Calculating leverage scores")
    
    if 'fpts' not in df.columns or 'ownership' not in df.columns:
        logger.warning("Missing site projection or ownership data for leverage calculation")
        return df
    
    # Calculate projection edge
    df['proj_edge'] = df['proj_mean'] - df['fpts']
    
    # Calculate leverage (edge * ownership inverse)
    # Add small constant to avoid division by zero
    df['leverage_score'] = (
        df['proj_edge'] * (100 / (df['ownership'] + 0.1))
    ).round(2)
    
    # Position-relative leverage
    df['leverage_pos_rank'] = df.groupby('position')['leverage_score'].rank(
        ascending=False, method='min'
    ).astype(int)
    
    return df


def create_boom_analysis_summary(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Create summary statistics for boom analysis.
    
    Args:
        df: DataFrame with boom analysis
        
    Returns:
        Dictionary with boom analysis summary
    """
    summary = {}
    
    # Overall boom metrics
    if 'boom_score' in df.columns:
        summary['boom_scores'] = {
            'avg_boom_score': df['boom_score'].mean().round(1),
            'max_boom_score': df['boom_score'].max(),
            'min_boom_score': df['boom_score'].min(),
            'high_boom_players': len(df[df['boom_score'] >= 70]),
            'elite_boom_players': len(df[df['boom_score'] >= 85])
        }
    
    # Dart throw analysis
    if 'dart_flag' in df.columns:
        dart_throws = df[df['dart_flag'] == True]
        summary['dart_throws'] = {
            'total_dart_throws': len(dart_throws),
            'by_position': dart_throws['position'].value_counts().to_dict() if len(dart_throws) > 0 else {},
            'avg_boom_score': dart_throws['boom_score'].mean().round(1) if len(dart_throws) > 0 else None,
            'avg_ownership': dart_throws['ownership'].mean().round(1) if len(dart_throws) > 0 else None
        }
    
    # Boom probability by position
    if 'boom_prob' in df.columns and 'position' in df.columns:
        summary['boom_by_position'] = {}
        for pos in df['position'].unique():
            pos_df = df[df['position'] == pos]
            summary['boom_by_position'][pos] = {
                'avg_boom_prob': pos_df['boom_prob'].mean().round(1),
                'max_boom_prob': pos_df['boom_prob'].max(),
                'high_boom_count': len(pos_df[pos_df['boom_prob'] >= 20])
            }
    
    # Ownership distribution
    if 'ownership' in df.columns:
        summary['ownership'] = {
            'avg_ownership': df['ownership'].mean().round(1),
            'low_owned_5pct': len(df[df['ownership'] <= 5]),
            'low_owned_10pct': len(df[df['ownership'] <= 10]),
            'high_owned_20pct': len(df[df['ownership'] >= 20])
        }
    
    return summary


def export_dart_throws(df: pd.DataFrame, output_path: str, min_boom_score: float = 70):
    """
    Export dart throw candidates to CSV.
    
    Args:
        df: DataFrame with boom analysis
        output_path: Output file path
        min_boom_score: Minimum boom score for inclusion
    """
    logger.info(f"Exporting dart throws (boom score >= {min_boom_score}) to {output_path}")
    
    # Filter to dart throws
    dart_throws = df[df['boom_score'] >= min_boom_score].copy()
    
    # Sort by boom score
    dart_throws = dart_throws.sort_values('boom_score', ascending=False)
    
    # Select relevant columns
    export_cols = [
        'player', 'position', 'team', 'salary', 'ownership',
        'proj_mean', 'ceiling', 'boom_prob', 'boom_score',
        'value_per_1k', 'dart_flag'
    ]
    
    # Add site comparison if available
    if 'fpts' in dart_throws.columns:
        export_cols.extend(['fpts', 'delta_vs_site', 'pct_delta'])
    
    # Keep only available columns
    available_cols = [col for col in export_cols if col in dart_throws.columns]
    
    export_df = dart_throws[available_cols].copy()
    export_df.to_csv(output_path, index=False)
    
    logger.info(f"Exported {len(export_df)} dart throw candidates")


def calculate_stack_leverage(df: pd.DataFrame, 
                           stack_teams: List[str]) -> pd.DataFrame:
    """
    Calculate leverage for specific team stacks.
    
    Args:
        df: DataFrame with player data
        stack_teams: List of team abbreviations to analyze
        
    Returns:
        DataFrame filtered to stack teams with leverage metrics
    """
    logger.info(f"Calculating stack leverage for teams: {stack_teams}")
    
    # Filter to stack teams
    stack_df = df[df['team'].isin(stack_teams)].copy()
    
    if stack_df.empty:
        logger.warning("No players found for specified stack teams")
        return stack_df
    
    # Calculate team-relative boom scores
    for team in stack_teams:
        team_players = stack_df[stack_df['team'] == team]
        if len(team_players) > 1:
            # Identify which positions have highest boom potential
            team_boom_leaders = team_players.nlargest(3, 'boom_score')
            logger.info(f"{team} boom leaders: {team_boom_leaders[['player', 'position', 'boom_score']].to_dict('records')}")
    
    # Add stack correlation flags (players from same team)
    stack_df['in_stack'] = True
    
    return stack_df.sort_values('boom_score', ascending=False)