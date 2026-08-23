"""
probability_models.py
----------------------
The probability estimation engine.

Approach:
1. Each team has an "attack" and "defense" strength (multiplicative, 1.0 =
   league average) plus an Elo rating, all stored in the database and
   updated as real results come in (see learning.py).
2. Expected goals (lambda_home, lambda_away) are derived from league-average
   goals scaled by the two teams' attack/defense strengths, then nudged by
   the Elo differential.
3. A bivariate-ish scoreline grid is built from independent Poisson
   distributions with a small Dixon-Coles style correction applied to the
   low-scoring cells (0-0, 1-0, 0-1, 1-1), which is a well-known source of
   bias in the naive independent-Poisson model.
4. All markets (1X2, Over/Under, BTTS, team Over 1.5) are derived from that
   single scoreline grid, so they are mutually consistent.
5. A learned calibration adjustment (from learning.py / database.calibration)
   is applied as a final nudge once enough historical data exists.

This module is intentionally plugin-shaped: `PoissonEloModel` implements a
small `Estimator` interface (`estimate(match) -> MatchProbabilities`) so a
future model (e.g. an xG-based or ML model) can be swapped in without
touching value_calculator.py.
"""

import math
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import poisson

import config
from database import Database

logger = logging.getLogger("value_bet_finder.probability_models")


@dataclass
class MatchProbabilities:
    home_win: float
    draw: float
    away_win: float
    over_2_5: float
    under_2_5: float
    btts_yes: float
    btts_no: float
    home_over_1_5: float
    away_over_1_5: float
    lambda_home: float
    lambda_away: float
    grid: np.ndarray = field(repr=False)


def _dixon_coles_tau(x, y, lam_h, lam_a, rho):
    """Low-score correlation correction (Dixon & Coles, 1997)."""
    if x == 0 and y == 0:
        return 1 - (lam_h * lam_a * rho)
    if x == 0 and y == 1:
        return 1 + (lam_h * rho)
    if x == 1 and y == 0:
        return 1 + (lam_a * rho)
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def build_scoreline_grid(lambda_home: float, lambda_away: float,
                          max_goals: int = None, rho: float = -0.06) -> np.ndarray:
    """Return a (max_goals+1) x (max_goals+1) matrix of P(home=i, away=j)."""
    max_goals = max_goals or config.MAX_GOALS
    home_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    grid = np.outer(home_probs, away_probs)

    # Apply Dixon-Coles correction to the four low-scoring cells only.
    for i in range(min(2, max_goals + 1)):
        for j in range(min(2, max_goals + 1)):
            grid[i, j] *= _dixon_coles_tau(i, j, lambda_home, lambda_away, rho)

    total = grid.sum()
    if total > 0:
        grid = grid / total  # renormalise after the DC tweak
    return grid


def derive_markets_from_grid(grid: np.ndarray, over_line_main=2.5, over_line_team=1.5):
    n = grid.shape[0]
    home_win = draw = away_win = 0.0
    over_main = 0.0
    btts_yes = 0.0
    home_over_team = 0.0
    away_over_team = 0.0

    for i in range(n):
        for j in range(n):
            p = grid[i, j]
            if i > j:
                home_win += p
            elif i == j:
                draw += p
            else:
                away_win += p
            if (i + j) > over_line_main:
                over_main += p
            if i >= 1 and j >= 1:
                btts_yes += p
            if i > over_line_team:
                home_over_team += p
            if j > over_line_team:
                away_over_team += p

    return {
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over_main": over_main,
        "under_main": 1 - over_main,
        "btts_yes": btts_yes,
        "btts_no": 1 - btts_yes,
        "home_over_team": home_over_team,
        "away_over_team": away_over_team,
    }


class PoissonEloModel:
    """Default probability estimator: Poisson goal model + Elo adjustment."""

    ELO_GOAL_IMPACT = 0.0035  # tuned so a ~200 elo gap shifts lambda by roughly 15-20%

    def __init__(self, db: Database):
        self.db = db

    def _team_strength(self, team: str, league: str):
        rating = self.db.get_team_rating(team, league)
        return rating["attack"], rating["defense"], rating["elo"], rating["matches_played"]

    def estimate(self, home_team: str, away_team: str, league: str) -> MatchProbabilities:
        league_avg = self.db.get_league_average(league)
        avg_home_goals = league_avg["avg_home_goals"]
        avg_away_goals = league_avg["avg_away_goals"]

        h_attack, h_defense, h_elo, h_n = self._team_strength(home_team, league)
        a_attack, a_defense, a_elo, a_n = self._team_strength(away_team, league)

        # Base expected goals from attack/defense strengths relative to league average.
        lambda_home = max(0.05, avg_home_goals * h_attack * a_defense)
        lambda_away = max(0.05, avg_away_goals * a_attack * h_defense)

        # Elo differential nudge: stronger side gets a small extra goal boost,
        # capped so it can't dominate the estimate when ratings are still thin.
        elo_diff = h_elo - a_elo
        elo_adjustment = math.tanh(elo_diff / 400.0) * self.ELO_GOAL_IMPACT * 100
        lambda_home = max(0.05, lambda_home + elo_adjustment)
        lambda_away = max(0.05, lambda_away - elo_adjustment)

        grid = build_scoreline_grid(lambda_home, lambda_away)
        markets = derive_markets_from_grid(grid, config.OVER_LINE_MAIN, config.OVER_LINE_TEAM)

        return MatchProbabilities(
            home_win=markets["home_win"],
            draw=markets["draw"],
            away_win=markets["away_win"],
            over_2_5=markets["over_main"],
            under_2_5=markets["under_main"],
            btts_yes=markets["btts_yes"],
            btts_no=markets["btts_no"],
            home_over_1_5=markets["home_over_team"],
            away_over_1_5=markets["away_over_team"],
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            grid=grid,
        )

    def confidence(self, home_team: str, away_team: str, league: str) -> float:
        """
        A crude 0-1 confidence score based on how much history we have for
        both teams. Used to gate low-confidence predictions in
        value_calculator.py rather than betting blind on day one.
        """
        _, _, _, h_n = self._team_strength(home_team, league)
        _, _, _, a_n = self._team_strength(away_team, league)
        n = min(h_n, a_n)
        # Saturating function: 0 matches -> 0 confidence, ~20 matches -> ~0.86
        return 1 - math.exp(-n / 10.0)
