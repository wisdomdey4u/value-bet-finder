"""
learning.py
-----------
Two self-improvement mechanisms:

1. settle_and_update_ratings(): pull recently completed match results,
   settle any pending predictions against them, and nudge each team's
   attack/defense strength (EWMA) and Elo rating toward what actually
   happened. This is what lets the Poisson model get better at knowing
   each team's true scoring rate over the season.

2. recalibrate(): a lightweight isotonic-ish bucket calibration. Settled
   predictions are grouped into probability buckets (0-10%, 10-20%, ...)
   per market; we compare the model's average predicted probability in
   each bucket to the actual win rate observed, and store the difference
   as an additive adjustment. value_calculator.py looks this adjustment up
   and nudges future predictions toward reality.

Both are deliberately simple, transparent, and safe to run daily on a
small amount of data -- they degrade to a no-op adjustment (0.0) when
there isn't enough history yet, rather than overfitting on noise.
"""

import logging
import datetime as dt

import config
from database import Database
from api_client import fetch_recent_scores

logger = logging.getLogger("value_bet_finder.learning")

MIN_SAMPLES_PER_BUCKET = 8
BUCKET_WIDTH = 0.10  # 10 percentage-point buckets


def _expected_goals_for_team(is_home: bool, team_rating: dict, opp_rating: dict,
                              league_avg: dict) -> float:
    base = league_avg["avg_home_goals"] if is_home else league_avg["avg_away_goals"]
    attack = team_rating["attack"]
    defense = opp_rating["defense"]
    return max(0.05, base * attack * defense)


def _update_team_after_match(db: Database, team: str, league: str, goals_for: int,
                              goals_against: int, expected_goals_for: float,
                              opponent_elo: float, is_home: bool, result: str):
    rating = db.get_team_rating(team, league)
    lr = config.RATING_LEARNING_RATE

    # Attack strength: nudge toward (actual goals / expected goals) ratio.
    if expected_goals_for > 0:
        performance_ratio = goals_for / expected_goals_for
        # clip to avoid one freak result blowing up the rating
        performance_ratio = min(3.0, max(0.1, performance_ratio))
        new_attack = rating["attack"] * (1 - lr) + rating["attack"] * performance_ratio * lr
    else:
        new_attack = rating["attack"]

    # Defense strength: fewer goals conceded than expected -> defense improves
    # (defense multiplier < 1.0 means opponents score less against this team).
    league_avg_goals = 1.35  # rough overall goals/team/match reference point
    conceded_ratio = goals_against / league_avg_goals if league_avg_goals > 0 else 1.0
    conceded_ratio = min(3.0, max(0.1, conceded_ratio))
    new_defense = rating["defense"] * (1 - lr) + rating["defense"] * conceded_ratio * lr

    # Elo update (standard logistic expectation, scaled by goal difference).
    expected_score = 1.0 / (1.0 + 10 ** ((opponent_elo - rating["elo"]) / 400.0))
    actual_score = {"WIN": 1.0, "DRAW": 0.5, "LOSS": 0.0}[result]
    goal_diff = abs(goals_for - goals_against)
    margin_mult = 1.0
    if config.ELO_GOAL_DIFF_MULT and goal_diff > 1:
        margin_mult = 1 + 0.15 * (goal_diff - 1)
    new_elo = rating["elo"] + config.ELO_K_FACTOR * margin_mult * (actual_score - expected_score)

    db.upsert_team_rating(
        team, league,
        attack=round(new_attack, 4),
        defense=round(new_defense, 4),
        elo=round(new_elo, 2),
        matches_played=rating["matches_played"] + 1,
    )


def settle_and_update_ratings(db: Database):
    """Fetch recent completed scores, settle pending predictions against
    them, and update team ratings / league averages from the real results."""
    try:
        scores = fetch_recent_scores()
    except Exception:
        logger.exception("Could not fetch recent scores; skipping settlement this run")
        return

    if not scores:
        logger.info("No completed matches available to settle right now")
        return

    scores_by_id = {s["match_id"]: s for s in scores}

    # ---- Settle predictions --------------------------------------------
    pending = db.get_pending_predictions(older_than_hours=2)
    settled_count = 0
    for pred in pending:
        score = scores_by_id.get(pred["match_id"])
        if not score:
            continue
        home_score, away_score = score["home_score"], score["away_score"]
        won = _did_bet_win(pred, home_score, away_score)
        if won is None:
            continue  # couldn't determine (e.g. selection we don't know how to grade)
        db.settle_prediction(pred["id"], "WON" if won else "LOST", home_score, away_score)
        settled_count += 1
    if settled_count:
        logger.info("Settled %d predictions against real results", settled_count)

    # ---- Update team ratings + league averages ---------------------------
    updated_teams = set()
    for score in scores:
        league = score["league"]
        home, away = score["home_team"], score["away_team"]
        hs, as_ = score["home_score"], score["away_score"]
        key = (score["match_id"],)
        if key in updated_teams:
            continue
        updated_teams.add(key)

        league_avg = db.get_league_average(league)
        home_rating = db.get_team_rating(home, league)
        away_rating = db.get_team_rating(away, league)

        exp_home_goals = _expected_goals_for_team(True, home_rating, away_rating, league_avg)
        exp_away_goals = _expected_goals_for_team(False, away_rating, home_rating, league_avg)

        if hs > as_:
            home_result, away_result = "WIN", "LOSS"
        elif hs < as_:
            home_result, away_result = "LOSS", "WIN"
        else:
            home_result = away_result = "DRAW"

        _update_team_after_match(db, home, league, hs, as_, exp_home_goals,
                                  away_rating["elo"], True, home_result)
        _update_team_after_match(db, away, league, as_, hs, exp_away_goals,
                                  home_rating["elo"], False, away_result)

        db.update_league_average(league, hs, as_)

    logger.info("Updated ratings for %d completed matches", len(updated_teams))


def _did_bet_win(pred: dict, home_score: int, away_score: int):
    """Grade a settled prediction. Returns True/False, or None if the
    selection text can't be graded automatically."""
    market = pred["market"]
    selection = pred["selection"]
    home, away = pred["home_team"], pred["away_team"]
    total_goals = home_score + away_score

    if market == "1X2":
        if selection == "Home Win":
            return home_score > away_score
        if selection == "Away Win":
            return away_score > home_score
        if selection == "Draw":
            return home_score == away_score

    elif market == "Over/Under 2.5":
        if selection == "Over 2.5 Goals":
            return total_goals > 2.5
        if selection == "Under 2.5 Goals":
            return total_goals < 2.5

    elif market == "BTTS":
        both_scored = home_score > 0 and away_score > 0
        if "Yes" in selection:
            return both_scored
        if "No" in selection:
            return not both_scored

    elif market == "Team Over 1.5":
        if home in selection:
            return home_score > 1.5
        if away in selection:
            return away_score > 1.5

    return None


def recalibrate(db: Database):
    """Bucket-based probability calibration across all settled predictions."""
    markets = ["1X2", "Over/Under 2.5", "BTTS", "Team Over 1.5"]
    total_buckets_updated = 0

    for market in markets:
        settled = db.get_settled_predictions(market=market)
        if len(settled) < MIN_SAMPLES_PER_BUCKET:
            logger.info(
                "Skipping calibration for %s: only %d settled predictions so far",
                market, len(settled),
            )
            continue

        buckets = {}
        for pred in settled:
            p = pred["model_prob"]
            bucket_low = min(0.9, (p // BUCKET_WIDTH) * BUCKET_WIDTH)
            bucket_high = round(bucket_low + BUCKET_WIDTH, 2)
            buckets.setdefault((bucket_low, bucket_high), []).append(pred)

        for (low, high), preds in buckets.items():
            if len(preds) < MIN_SAMPLES_PER_BUCKET:
                continue
            predicted_avg = sum(p["model_prob"] for p in preds) / len(preds)
            actual_rate = sum(1 for p in preds if p["result"] == "WON") / len(preds)
            # Shrink the raw adjustment toward 0 for small samples so we don't
            # overreact to noise (simple empirical-Bayes style shrinkage).
            shrinkage = len(preds) / (len(preds) + 20)
            adjustment = (actual_rate - predicted_avg) * shrinkage
            db.upsert_calibration(market, low, high, predicted_avg, actual_rate,
                                   len(preds), adjustment)
            total_buckets_updated += 1
            logger.info(
                "Calibration[%s %.0f-%.0f%%]: predicted=%.1f%% actual=%.1f%% "
                "n=%d adj=%+.1f%%",
                market, low * 100, high * 100, predicted_avg * 100,
                actual_rate * 100, len(preds), adjustment * 100,
            )

    logger.info("Recalibration complete: %d buckets updated", total_buckets_updated)


def run_learning_cycle(db: Database):
    """Convenience entry point called once per daily run: settle results,
    update team ratings, then recalibrate probability buckets."""
    logger.info("Starting learning cycle (settlement + recalibration)")
    settle_and_update_ratings(db)
    recalibrate(db)
    logger.info("Learning cycle complete")
