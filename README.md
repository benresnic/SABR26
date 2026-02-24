# Methodology README — HOA vs. WCA (SABR Diamond Dollars Case)

## Project Overview

This project evaluates **when power-variance hitters should maintain their default Hitter-Optimized Approach (HOA)** versus adopt a **Win-Creation Approach (WCA)** that sacrifices some season-long expected offensive value to improve **in-the-moment win probability** in specific game states.

The core question is not “what is best on average?” but rather:

- **When** is a context-dependent adjustment justified?
- **Why** does it work in those states?
- **For whom** (which hitters) is the tradeoff most beneficial?

This framing follows the SABR Diamond Dollars case prompt and our presentation approach.

---

## Key Definitions

### Hitter-Optimized Approach (HOA)
A hitter’s default approach that maximizes expected season-long offensive value (e.g., expected run value / overall production).

### Win-Creation Approach (WCA)
A context-dependent adjustment (operationalized here as a reduction in bat speed and/or swing length) that **reshapes the plate appearance outcome distribution** to improve the team’s expected win probability in a specific moment, even if it reduces expected individual output.

---

## Scope and Cohort

We focus on a **Power-Variance hitter cohort** (high-K / high-HR / low-contact type profiles), where the HOA vs. WCA tradeoff is most meaningful.

### Cohort rationale
These hitters tend to have:
- A fat-right-tail outcome distribution (HR upside)
- Meaningful zero-run risk (e.g., strikeouts / unproductive outs)
- Potential value in redistributing outcome probabilities in certain leverage states

### Cohort size
The case cohort contains **27 hitters**, profiled across the last two seasons combined.

---

## Methodology Summary (High-Level Pipeline)

1. **Define the decision problem (HOA vs. WCA)**
2. **Represent each plate appearance with a game-state vector**
3. **Estimate plate-appearance outcome probabilities**
4. **Model how bat-speed/swing-length changes affect hitter skill inputs**
5. **Map each outcome to post-PA game-state transitions**
6. **Convert transitions to expected post-PA win probability**
7. **Compare WCA vs. HOA across states, hitters, and feasible adjustments**
8. **Summarize where WCA is beneficial and where HOA remains optimal**

---

## 1) Decision Framing: Conditional Value, Not Seasonal Average

Our framework explicitly follows the case requirement to reason about **conditional value** rather than averages.

We distinguish:
- States where maximizing expected offensive output is best (HOA)
- States where increasing the probability of a specific team outcome (often exactly one run) creates more win probability (WCA)

This is a **distributional tradeoff problem**, not a simple average-value optimization.

---

## 2) Game-State Representation

Each evaluation is performed in a baseball game context that includes variables such as:

- Inning
- Top/Bottom of inning
- Score differential
- Count (balls/strikes)
- Outs
- Base state
- Pitcher quality (using xFIP proxy)
- Candidate adjustment to:
  - **Bat speed**
  - **Swing length**

This allows us to estimate value **state-by-state**, rather than using a global one-size-fits-all rule.

---

## 3) Plate Appearance Outcome Model

We model the probability distribution over end-of-PA outcomes:

- Strikeout
- Walk/HBP
- Field out (including outcome grouping with field-out/error effects in state transitions)
- Single
- Double
- Triple
- Home run

### Core modeling approach
We use a **Bayesian multinomial logistic regression** to estimate these outcome probabilities.

### Inputs (as presented in our workflow)
The model uses a combination of hitter and context inputs, including:
- **O-Swing%**
- **Z-Contact%**
- **xISO**
- **Pitcher xFIP**
- **Current count**
- Long-run baseline tendencies / averages (for stabilization and context anchoring)

### Why this model?
A multinomial outcome model is appropriate because:
- PA outcomes are mutually exclusive categories
- The question is fundamentally about **distributional shifts**
- Bayesian estimation supports uncertainty-aware inference and diagnostics

---

## 4) State Dependence: Why Count Matters More Than Some Other Context Features

Before finalizing the outcome model structure, we examined whether outcome distributions vary meaningfully across:
- Count
- Outs
- Base state

Using **chi-square tests** and **Cramér’s V (effect size)**, we found that outcome probabilities shift **more strongly by count** than by outs or base state (practical effect size framing), which informed the modeling emphasis on count in the outcome layer.

This supports the idea that **count is a primary lever** in the HOA vs. WCA decision environment.

---

## 5) Linking Swing Mechanics to Model Inputs (Bat Speed / Swing Length Adjustments)

A central challenge is that WCA is defined through changes in approach (bat speed and/or swing length), but the outcome model operates on inputs like **Z-Contact%** and **xISO**.

### Our solution
We fit a **Partial Least Squares (PLS) regression with polynomial features (degree 2)** to model how changes in:
- Bat speed
- Swing length

affect key hitter skill inputs used by the outcome model, particularly:
- **Z-Contact%**
- **xISO**

### Joint modeling motivation
PLS is used to learn a **shared latent structure**, capturing the fact that contact and power change together when swing mechanics change.

### Transformations for diminishing returns / edge behavior
We apply transformations to improve realism near boundaries:
- **logit transform** for Z-Contact%
- **log transform** for xISO

This helps reflect diminishing returns and nonlinear sensitivity near the edges of feasible performance.

---

## 6) Enumerating Candidate WCA Adjustments

For each hitter and game state, we evaluate a grid of candidate “shorten up” adjustments.

### Adjustment grid (as presented)
We test combinations of **bat speed and swing length reductions from 0% to -12% in increments of 3%**.

Examples:
- (0%, 0%) → HOA baseline
- (-3%, -3%)
- (-6%, -3%)
- ...
- (-12%, -12%)

This generates a structured decision surface for comparing WCA vs. HOA.

---

## 7) Markov Chain Win Probability Framework

The core valuation engine is a **Markov-style expected change in win probability** computation.

### Step A — Pre-PA state
For a given game state, we start from:
- current state variables
- pre-PA win probability

### Step B — Outcome probabilities
Using the multinomial model (with adjusted inputs under a candidate WCA), we estimate:
- P(K), P(BB/HBP), P(field out), P(1B), P(2B), P(3B), P(HR)

### Step C — Post-PA state transition distribution
For each outcome, base state, and outs combination, we use observed historical transitions to build a probability distribution over:
- post-PA base state
- post-PA outs
- runs scored

This is based on observed post-PA transitions (2023–2025 in our presentation workflow).

### Step D — Expected post-PA win probability
Each possible post-PA state maps to a post-PA win probability. We compute the expected post-PA win probability by weighting over:
1. Outcome probabilities
2. Transition probabilities conditional on outcome

### Step E — Value metric
We define value as the **expected change in win probability (ΔWP)** relative to the pre-PA state.

---

## 8) HOA vs. WCA Comparison and Recommendation Rule

For each hitter × state × candidate adjustment:

1. Compute **ΔWP under HOA** (baseline approach)
2. Compute **ΔWP under WCA candidate** (adjusted approach)
3. Take the difference:

\[
\Delta WP_{WCA\;vs\;HOA} = \Delta WP_{WCA} - \Delta WP_{HOA}
\]

### Interpretation
- **Positive**: WCA improves expected win probability vs. HOA
- **Negative**: Maintain HOA

### Recommendation concept
A WCA recommendation is only meaningful when:
- It produces a win-probability improvement
- The required mechanical adjustment is considered **feasible** for that hitter (see feasibility discussion below)

---

## 9) Feasibility and Hitter-Specific Adjustability

The case prompt emphasizes that hitters differ in their ability to change approach. Our framework accounts for this by treating WCA benefit as **hitter-specific** and evaluating recommended adjustments within realistic/feasible change regions.

### Why this matters
A strategy that looks optimal in theory may not be actionable if the required change in bat speed/swing length is outside a hitter’s demonstrated adjustability band.

### Practical implication
Recommendations are framed as:
- **Deploy WCA in targeted states**
- **Conditional WCA trial**
- **Keep HOA**

rather than forcing a universal prescription.

---

## 10) Diagnostics and Model Validation (as shown in the presentation)

We included multiple checks to verify that the modeling stack was behaving reasonably:

### Bayesian multinomial model diagnostics
- Trace plots
- Divergence checks (reported zero divergences in the presentation)
- Calibration curves (predicted vs. observed probabilities by outcome)
- Aggregate observed vs. predicted outcome distribution comparison

### Face-validity / baseball-validity checks
We also checked whether modeled relationships were directionally intuitive:
- Shortening up tends to increase contact-related outcomes
- Power-related outcomes (especially HR / xISO) tend to decline as swing speed/length are reduced
- The tradeoff varies by hitter, count, and pitcher quality

---

## 11) How We Answered “When, Why, and For Whom”

### When (state conditions)
We identify game states where WCA is more likely to outperform HOA, especially when:
- **A single run has outsized win value**
- There are **two strikes** and reducing strikeout risk is valuable
- **Sac-fly / productive-contact states** exist (e.g., runner on 3rd, <2 outs)
- Late-inning leverage increases the value of one-run outcomes

### Why (distributional reasoning)
WCA works not because it raises average offensive output, but because it can **reshape the outcome distribution**:
- Lowering K probability
- Increasing productive contact / 1B / ball-in-play outcomes
- Sacrificing some HR upside when the marginal value of “exactly one run” is high

### For whom (hitter profiles)
Not all power-variance hitters benefit equally. Benefit depends on:
- Baseline outcome distribution shape
- Contact/power tradeoff curvature
- Demonstrated adjustability (feasibility)
- State-specific interaction with count and pitcher quality

---

## 12) Deliverable Outputs We Emphasized

Our presentation/reporting focused on outputs that support decision-making rather than only model internals, including:

- HOA vs. WCA flow maps (state transition reasoning)
- Heatmaps of WCA value by count / score / inning
- Hitter ranking summaries (average ΔWP from WCA vs. HOA)
- Distribution shift plots (e.g., K and HR changes)
- Feasibility vs. WP gain decision boundaries
- State-specific examples illustrating why two hitters may warrant different recommendations

---

## 13) Assumptions and Limitations

This framework is designed for decision support, not perfect prediction. Key limitations include:

1. **Model dependence on historical relationships**  
   Outcome and transition estimates rely on historical data and may not fully transfer to every current context.

2. **Mechanics-to-outcome mapping simplification**  
   Bat speed and swing length adjustments are represented through modeled effects on summary hitting inputs (e.g., Z-Contact%, xISO), which abstracts away many biomechanical and pitch-level nuances.

3. **Context granularity**  
   Some factors (defensive alignment, pitch type/location sequence, weather, park effects, fatigue, etc.) are not fully modeled in the core framework.

4. **Rare-state uncertainty**  
   Certain leverage states and transitions can be data-sparse.

5. **Feasibility estimation uncertainty**  
   A hitter’s ability to execute a given adjustment in-game is imperfectly observed and may vary by pitcher, game pressure, and health.

---

## 14) Practical Interpretation for Teams / Coaches

This methodology is best used as a **situational decision-support framework** that helps answer:

- Which hitters should be candidates for a WCA adjustment?
- In which states is the upside largest?
- Which states should remain HOA by default?
- How much feasibility is required before a WCA recommendation becomes actionable?

The result is not a rigid rulebook, but a hitter-specific playbook for context-dependent approach optimization.

---

## 15) Reproducibility Notes (Suggested README Additions for Repo Use)

If you are using this in a public/private repository, consider adding:
- Data sources and date ranges
- Feature engineering scripts
- Model training scripts (Bayesian multinomial + PLS adjustment model)
- Win probability lookup / transition table generation scripts
- Inference pipeline scripts
- Shiny app / dashboard instructions
- Environment setup (`requirements.txt`, `renv.lock`, Dockerfile, etc.)

---

## Final Summary

Our methodology combines:

- **Bayesian multinomial outcome modeling**
- **Joint adjustment modeling of contact/power responses to swing changes**
- **Empirical post-PA state transition probabilities**
- **Markov-style expected win probability calculations**
- **Hitter-specific feasibility-aware decision analysis**

to determine **when a Win-Creation Approach (WCA) should replace a Hitter-Optimized Approach (HOA)** for power-variance hitters in specific game states.
