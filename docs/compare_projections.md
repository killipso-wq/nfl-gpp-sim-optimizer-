# Comparing Projections: Understanding Value, Boom, and Diagnostics

## Overview

The NFL Simulator generates projections and comparison metrics to help identify value plays, boom candidates, and validate accuracy against site projections.

## Value Metrics

### Core Value Calculations

**Value per $1k** = (Projection ÷ Salary) × 1000
- Measures fantasy points per $1000 of salary
- 3.0+ typically indicates good value
- Compare across positions for optimal lineup construction

**Ceiling Value** = (P90 Projection ÷ Salary) × 1000  
- Value using 90th percentile projection
- Identifies upside potential per dollar

**Floor Value** = (P10 Projection ÷ Salary) × 1000
- Value using 10th percentile projection  
- Risk assessment metric

### Position-Relative Value

**Value Z-Score**: Standardized value within position
- Positive = above-average value for position
- -1 to +1 = within one standard deviation of position mean

**Value Percentile**: Ranking within position (0-100)
- 80th percentile = better value than 80% of position players

## Boom Analysis

### Boom Probability

Player "booms" when fantasy points ≥ boom threshold, where:

**Boom Threshold** = max(position_threshold, 1.2×site_proj, site_proj+5)

Position thresholds (90th percentile historical):
- QB: 25.0 points
- RB: 20.0 points  
- WR: 18.0 points
- TE: 15.0 points
- DST: 10.0 points

**Boom Probability** = P(sim_points ≥ boom_threshold)
- Uses normal approximation based on projection mean/std
- 20%+ boom probability indicates significant upside

### Boom Score (1-100 Scale)

Multi-factor score combining:

1. **Base Score** (0-60 pts): Boom probability × 60
2. **Value Bonus** (0-20 pts): Positive value per $1k
3. **Ownership Boost** (0-15 pts): Low ownership premium
4. **Beat Site Bonus** (0-5 pts): Probability of beating site projection

**Ownership Boost Scale:**
- ≤ 1%: +15 points
- ≤ 3%: +12 points  
- ≤ 5%: +8 points
- ≤ 10%: +4 points
- > 10%: +0 points

### Dart Flag

**Dart Flag** = Ownership ≤ 5% AND Boom Score ≥ 70

Identifies low-owned players with high boom potential for contrarian plays.

## Site Comparison Metrics

### Projection Differences

**Delta vs Site** = Our Projection - Site Projection
- Absolute point difference
- Positive = we project higher

**Percent Delta** = (Delta ÷ Site Projection) × 100
- Relative difference as percentage
- Accounts for projection magnitude

**Beat Site Probability** = P(sim_points ≥ site_projection)
- Uses normal approximation
- 60%+ suggests confidence in our higher projection

### Coverage Analysis

**Coverage (P10-P90)** = % of site projections within our prediction interval
- Ideally ~80% for 80% prediction interval
- Higher coverage = well-calibrated uncertainty
- Lower coverage = too narrow/confident intervals

## Diagnostics

### Error Metrics

**Mean Absolute Error (MAE)**:
- Average absolute difference between projections
- Lower = better accuracy
- Interpretable in fantasy points

**Root Mean Squared Error (RMSE)**:  
- Penalizes large errors more heavily
- Always ≥ MAE
- RMSE >> MAE suggests some large outlier errors

**Correlation**:
- Linear relationship strength (-1 to +1)
- 0.7+ = excellent agreement
- 0.5-0.7 = good agreement
- 0.3-0.5 = fair agreement
- < 0.3 = poor agreement

### Bias Analysis

**Mean Bias** = Average(Our Projection - Site Projection)
- Positive = systematic over-projection
- Negative = systematic under-projection
- Near zero = unbiased on average

**Median Bias**: Same as mean bias but using median
- Less sensitive to outliers
- Better represents "typical" bias

## Flag Analysis

Players flagged for manual review based on:

1. **Large Absolute Difference**: |Delta| ≥ 5 points
2. **Large Percentage Difference**: |Percent Delta| ≥ 25%  
3. **Very High Projection**: ≥ 30 points (potential outlier)
4. **Very Low Projection**: ≤ 5 points for skill positions

Review flagged players to identify:
- Data entry errors
- Injury/status updates
- Different game environment assumptions
- Genuine edge opportunities

## Interpretation Guidelines

### Value Plays
- Value per $1k > 3.0 in cash games
- Value per $1k > 2.5 in GPP with high ceiling
- Consider position scarcity and lineup construction

### GPP Strategy
- Target boom score 60+ for ceiling plays
- Dart flags for contrarian leverage  
- Balance boom probability with projection accuracy

### Cash Game Strategy
- Focus on consistent value (high floor value)
- Avoid high-boom, high-bust players
- Weight diagnostics correlation for safety

### Projection Validation
- Correlation > 0.5 for baseline confidence
- Low MAE (< 3 points) for accurate calibration
- Review flagged players before finalizing lineups
- Coverage 70-90% suggests well-calibrated intervals

## Position-Specific Considerations

### Quarterback
- Boom threshold highest (25 pts) due to passing volume
- Rush upside creates ceiling differentiation
- Weather/dome games affect variance

### Running Back  
- TD dependency creates boom/bust profiles
- Game script heavily influences volume
- Receiving usage provides floor stability

### Wide Receiver
- Target share most predictive of floor
- Air yards share drives ceiling potential
- Red zone usage crucial for TD upside

### Tight End
- Lower boom threshold (15 pts) reflects position scoring
- Target concentration creates binary outcomes  
- Blocking usage reduces floor predictability

### Defense/Special Teams
- Lowest boom threshold (10 pts)
- Opponent strength primary factor
- Weather and game script important for sacks/turnovers