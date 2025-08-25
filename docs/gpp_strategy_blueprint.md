# GPP Strategy Blueprint

## Overview

This document outlines the strategic framework for NFL GPP optimization, focusing on stack construction, ownership differentiation, and tournament-specific constraints.

## Core GPP Principles

### 1. Tournament vs. Cash Game Mentality
- **Upside over floor**: Prioritize ceiling outcomes over consistent production
- **Differentiation**: Seek unique lineup construction that separates from field
- **Correlation**: Leverage positive correlation through stacking strategies

### 2. Ownership Strategy
- **Contrarian approach**: Target lower-owned players with boom potential
- **Balanced exposure**: Avoid complete fade of obvious plays
- **Dart integration**: Include 1-2 minimum-owned players with upside

## Stack Construction

### Primary Stacks
1. **QB + WR**: Core correlation foundation
2. **QB + TE**: Lower owned alternative
3. **QB + RB**: Rushing QB specific strategy

### Bring-Back Strategy
- **Opposing team**: QB + WR vs. opposing WR/RB
- **Game script correlation**: Target high-total games
- **Salary efficiency**: Use bring-back to access premium plays

### Mini-Stacks
- **RB + DST**: Defensive scoring correlation
- **WR + WR**: Same-team receivers in high-volume offenses
- **TE + WR**: Secondary pass-catcher correlation

## Preset Configurations

### Small Field (GPP Presets: Small)
- **Ownership band**: 15-35%
- **Boom score threshold**: 60+
- **Value requirement**: 3.5+ points per $1K
- **Stack requirements**: 1 primary stack, optional bring-back
- **Dart requirement**: 1+ player <10% owned

### Mid Field (GPP Presets: Mid) 
- **Ownership band**: 10-25%
- **Boom score threshold**: 70+
- **Value requirement**: 4.0+ points per $1K
- **Stack requirements**: 1 primary stack + bring-back
- **Dart requirement**: 1-2 players <8% owned

### Large Field (GPP Presets: Large)
- **Ownership band**: 5-20%
- **Boom score threshold**: 75+
- **Value requirement**: 4.5+ points per $1K
- **Stack requirements**: 1 primary + 1 mini-stack
- **Dart requirement**: 2+ players <5% owned

## Constraint Implementation

### Ownership Constraints
```python
# Example constraint logic
if ownership_band[0] <= player_ownership <= ownership_band[1]:
    include_in_pool = True
```

### Boom Score Filtering
```python
# Filter player pool by boom score threshold
eligible_players = players_df[players_df['boom_score'] >= boom_threshold]
```

### Value Requirements
```python
# Ensure lineup meets minimum value threshold
lineup_value = sum(player['ceil_per_1k'] for player in lineup)
min_value = len(lineup) * value_per_1k_threshold
```

### Stack Requirements
```python
# Enforce primary stack
qb_team = lineup['QB']['team']
stack_players = [p for p in lineup.values() if p['team'] == qb_team]
assert len(stack_players) >= 2  # QB + at least 1 receiver

# Enforce bring-back (if enabled)
if enforce_bring_back:
    opponent_players = [p for p in lineup.values() if p['opp'] == qb_team]
    assert len(opponent_players) >= 1
```

### Dart Requirements
```python
# Ensure minimum dart plays
dart_players = [p for p in lineup.values() if p['dart_flag'] == True]
assert len(dart_players) >= dart_requirement
```

## Salary Management

### Salary Leftover Bands
- **Tight**: $0-200 remaining (maximize spend)
- **Moderate**: $200-500 remaining (balanced approach)  
- **Conservative**: $500+ remaining (leave salary for pivots)

### Position Allocation Guidelines
- **Premium positions**: QB, top-tier RB1/WR1
- **Value positions**: RB2, WR3, TE, DST
- **Flex optimization**: Use highest ceiling available

## Risk Management

### Exposure Limits
- **Single player**: Maximum 20% exposure across lineups
- **Single game**: Maximum 40% salary allocation
- **Single team**: Maximum 60% salary allocation

### Weather Considerations
- **Wind speed**: >15 mph reduces passing game reliability
- **Precipitation**: Benefits rushing attacks and unders
- **Temperature**: <32°F impacts ball security

### Injury Monitoring
- **Late scratches**: Build lineup flexibility
- **Snap count concerns**: Monitor practice participation
- **Game-time decisions**: Avoid questionable players in cash, embrace in GPP

## Advanced Strategies

### Leverage Spots
- **Narrative fades**: Target players the public is avoiding
- **Pricing inefficiencies**: Exploit DFS pricing lag
- **Matchup advantages**: Target favorable defensive matchups

### Correlation Maximization
- **Positive game scripts**: Stack teams in high-total games
- **Negative correlation**: Balance with opposing defenses
- **Secondary correlation**: Target pass-catchers in same offense

### Tournament Selection
- **Field size considerations**: Adjust ownership targets based on entries
- **Payout structure**: Top-heavy vs. flat payouts influence strategy
- **Entry distribution**: Multiple lineups vs. single bullet approach

## Implementation Notes

The GPP Presets UI provides quick configuration of these strategic principles:

1. **Preset Selector**: Choose Small/Mid/Large field approach
2. **Toggle Controls**: Enable/disable strategic elements
3. **Slider Adjustments**: Fine-tune thresholds within preset bounds
4. **Apply Preset**: Populate constraint panel with selected configuration

This framework serves as the foundation for the optimizer's constraint engine, ensuring strategic alignment with tournament objectives.