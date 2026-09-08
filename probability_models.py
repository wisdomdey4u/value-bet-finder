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
 
import logging
import math
from typing import Optional
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
 
 
def build_scoreline_grid(
    lambda_home: float, lambda_away: float, max_goals: int = None, rho: float = -0.06
) -> np.ndarray:
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
        return (
            rating["attack"],
            rating["defense"],
            rating["elo"],
            rating["matches_played"],
        )
 
    def get_matches_played(self, team: str, league: str) -> int:
        """Public accessor for matches_played to avoid private method coupling."""
        rating = self.db.get_team_rating(team, league)
        return rating["matches_played"]
 
    def estimate(
        self, home_team: str, away_team: str, league: str
    ) -> MatchProbabilities:
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
        markets = derive_markets_from_grid(
            grid, config.OVER_LINE_MAIN, config.OVER_LINE_TEAM
        )
 
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
 
 
class MarkovFormModel:
    """
    Markov Form Model: extends PoissonEloModel with recent-form adjustment
    via a 3-state (WIN/DRAW/LOSS) Markov chain.
 
    The model builds venue-specific transition matrices from each team's
    recent results, predicts the next-match outcome distribution, and
    converts this into form multipliers that adjust the base attack/defense
    lambdas from PoissonEloModel.
 
    States:
        0 = WIN
        1 = DRAW
        2 = LOSS
 
    Transition matrix P[i, j] = P(next_result=j | current_result=i)
    with Laplace (add-1) smoothing to handle sparse data.
    """
 
    LOOKBACK = 10
    MIN_MATCHES_FOR_MARKOV = 5
    MAX_FORM_ADJUSTMENT = 0.10
    BLEND_SATURATION = 20.0
    FORM_MULTIPLIER_SCALE = 5.0  # Scales win/loss diff to attack/defense adjustment
 
    def __init__(self, db: Database):
        """
        Initialize the Markov Form Model.
 
        Args:
            db: Database instance for fetching team ratings and recent results.
        """
        self.db = db
        self._base_model = PoissonEloModel(db)
 
    def _team_strength(self, team: str, league: str):
        """Delegate to base model for team strength."""
        return self._base_model._team_strength(team, league)
 
    def _result_to_state(self, result: str) -> int:
        """
        Convert result string to state index.
 
        Args:
            result: One of "WIN", "DRAW", "LOSS" (case-insensitive).
 
        Returns:
            Integer state: 0=WIN, 1=DRAW, 2=LOSS.
        """
        result = result.upper()
        if result == "WIN":
            return 0
        elif result == "DRAW":
            return 1
        elif result == "LOSS":
            return 2
        else:
            raise ValueError(f"Unknown result: {result}")
 
    def _build_transition_matrix(self, results: list[dict]) -> np.ndarray:
        """
        Build a 3x3 transition matrix from a sequence of results.
 
        Args:
            results: List of match dicts from db.get_team_recent_results(),
                     ordered most recent first.
 
        Returns:
            3x3 numpy array where P[i, j] = probability of transitioning
            from state i to state j, with Laplace (add-1) smoothing.
        """
        matrix = np.ones((3, 3), dtype=float)
 
        # Results are ordered most recent first (results[0] = most recent).
        # A transition's "from" state must be the EARLIER result and "to"
        # must be the LATER one, so that indexing the matrix by the most
        # recent actual result (as _predict_outcome_distribution does)
        # correctly answers "what tends to follow this state" rather than
        # "what tended to precede it". Since results[i+1] is chronologically
        # earlier than results[i], the correct pair is
        # (from=results[i+1], to=results[i]).
        for i in range(len(results) - 1):
            earlier = self._result_to_state(results[i + 1]["result"])  # happened first
            later = self._result_to_state(results[i]["result"])       # happened after
            matrix[earlier, later] += 1.0
 
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / row_sums
        return matrix
 
    def _predict_outcome_distribution(
        self, trans_matrix: np.ndarray, last_result: str
    ) -> tuple[float, float, float]:
        """
        Predict next-match outcome distribution from transition matrix.
 
        Args:
            trans_matrix: 3x3 transition matrix from _build_transition_matrix.
            last_result: The team's most recent result ("WIN", "DRAW", or "LOSS").
 
        Returns:
            Tuple of (p_win, p_draw, p_loss) summing to 1.0.
        """
        state = self._result_to_state(last_result)
        row = trans_matrix[state]
        return float(row[0]), float(row[1]), float(row[2])
 
    def _compute_form_multipliers(
        self,
        p_win: float,
        p_draw: float,
        p_loss: float,
        base_p_win: float,
        base_p_draw: float,
        base_p_loss: float,
    ) -> tuple[float, float]:
        """
        Compute attack and defense form multipliers from predicted vs base distribution.
 
        Args:
            p_win, p_draw, p_loss: Predicted outcome probabilities from Markov model.
            base_p_win, base_p_draw, base_p_loss: Base model outcome probabilities.
 
        Returns:
            Tuple of (attack_multiplier, defense_multiplier), each in [0.9, 1.1].
        """
        win_diff = p_win - base_p_win
        loss_diff = p_loss - base_p_loss
 
        attack_mult = 1.0 + (win_diff - loss_diff) * self.MAX_FORM_ADJUSTMENT * self.FORM_MULTIPLIER_SCALE
        defense_mult = 1.0 - (win_diff - loss_diff) * self.MAX_FORM_ADJUSTMENT * self.FORM_MULTIPLIER_SCALE
 
        attack_mult = max(1.0 - self.MAX_FORM_ADJUSTMENT, min(1.0 + self.MAX_FORM_ADJUSTMENT, attack_mult))
        defense_mult = max(1.0 - self.MAX_FORM_ADJUSTMENT, min(1.0 + self.MAX_FORM_ADJUSTMENT, defense_mult))
 
        return attack_mult, defense_mult
 
    def estimate(
        self, home_team: str, away_team: str, league: str
    ) -> MatchProbabilities:
        """
        Estimate match probabilities with Markov form adjustment.
 
        Args:
            home_team: Home team name.
            away_team: Away team name.
            league: League name.
 
        Returns:
            MatchProbabilities object with all market probabilities.
        """
        base_probs = self._base_model.estimate(home_team, away_team, league)
 
        h_results = self.db.get_team_recent_results(
            home_team, league, limit=self.LOOKBACK, venue="home"
        )
        a_results = self.db.get_team_recent_results(
            away_team, league, limit=self.LOOKBACK, venue="away"
        )
 
        h_n = len(h_results)
        a_n = len(a_results)
 
        min_n = min(h_n, a_n)
        if min_n < self.MIN_MATCHES_FOR_MARKOV:
            return base_probs
 
        h_trans = self._build_transition_matrix(h_results)
        a_trans = self._build_transition_matrix(a_results)
 
        h_last = h_results[0]["result"]
        a_last = a_results[0]["result"]
 
        h_p_win, h_p_draw, h_p_loss = self._predict_outcome_distribution(h_trans, h_last)
        a_p_win, a_p_draw, a_p_loss = self._predict_outcome_distribution(a_trans, a_last)
 
        base_h_win = base_probs.home_win
        base_draw = base_probs.draw
        base_a_win = base_probs.away_win
 
        h_attack_mult, h_defense_mult = self._compute_form_multipliers(
            h_p_win, h_p_draw, h_p_loss, base_h_win, base_draw, base_a_win
        )
        a_attack_mult, a_defense_mult = self._compute_form_multipliers(
            a_p_win, a_p_draw, a_p_loss, base_a_win, base_draw, base_h_win
        )
 
        weight = min(1.0, min_n / self.BLEND_SATURATION)
 
        h_attack_mult = 1.0 + weight * (h_attack_mult - 1.0)
        h_defense_mult = 1.0 + weight * (h_defense_mult - 1.0)
        a_attack_mult = 1.0 + weight * (a_attack_mult - 1.0)
        a_defense_mult = 1.0 + weight * (a_defense_mult - 1.0)
 
        h_attack, h_defense, h_elo, _ = self._team_strength(home_team, league)
        a_attack, a_defense, a_elo, _ = self._team_strength(away_team, league)
 
        league_avg = self.db.get_league_average(league)
        avg_home_goals = league_avg["avg_home_goals"]
        avg_away_goals = league_avg["avg_away_goals"]
 
        lambda_home = max(0.05, avg_home_goals * h_attack * h_attack_mult * a_defense * a_defense_mult)
        lambda_away = max(0.05, avg_away_goals * a_attack * a_attack_mult * h_defense * h_defense_mult)
 
        elo_diff = h_elo - a_elo
        elo_adjustment = math.tanh(elo_diff / 400.0) * self._base_model.ELO_GOAL_IMPACT * 100
        lambda_home = max(0.05, lambda_home + elo_adjustment)
        lambda_away = max(0.05, lambda_away - elo_adjustment)
 
        grid = build_scoreline_grid(lambda_home, lambda_away)
        markets = derive_markets_from_grid(
            grid, config.OVER_LINE_MAIN, config.OVER_LINE_TEAM
        )
 
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
        Confidence score incorporating both base rating history and Markov form data.
 
        Combines:
        - Rating confidence: based on minimum matches_played from base model (like PoissonEloModel)
        - Markov confidence: based on minimum recent results available for venue-specific form
 
        Both use saturating function: 0 matches -> 0, ~20 matches -> ~0.86
        Final blend: 0.7 * rating_confidence + 0.3 * markov_confidence
        """
        # Get matches_played from base model (rating history)
        h_n = self._base_model.get_matches_played(home_team, league)
        a_n = self._base_model.get_matches_played(away_team, league)
        rating_n = min(h_n, a_n)
 
        # Get recent result counts for venue-specific Markov data
        h_results = self.db.get_team_recent_results(
            home_team, league, limit=self.LOOKBACK, venue="home"
        )
        a_results = self.db.get_team_recent_results(
            away_team, league, limit=self.LOOKBACK, venue="away"
        )
        markov_n = min(len(h_results), len(a_results))
 
        # Saturating function: 0 -> 0, ~20 -> ~0.86
        rating_confidence = 1 - math.exp(-rating_n / 10.0)
        markov_confidence = 1 - math.exp(-markov_n / 10.0)
 
        # Blend: 70% rating confidence, 30% Markov confidence
        return 0.7 * rating_confidence + 0.3 * markov_confidence
