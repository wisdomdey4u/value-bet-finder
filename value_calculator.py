"""
value_calculator.py
--------------------
Turns (model probabilities + bookmaker odds) into ranked, diversified value
bet picks.

Value definition used throughout:
    raw_implied_probability   = 1 / decimal_odds
    no_vig_market_probability = raw_implied_probability normalised across
                                 all outcomes in the same market so the
                                 bookmaker's overround is removed
    model_probability          = our estimated true probability (calibrated)
    estimated_edge             = model_probability - no_vig_market_probability
    expected_value             = model_probability * decimal_odds - 1
"""

import logging
import datetime as dt

import config
from database import Database
from probability_models import PoissonEloModel, MatchProbabilities

logger = logging.getLogger("value_bet_finder.value_calculator")

# Maps our internal market/selection vocabulary to the outcome names used by
# The Odds API for each market key.
MARKET_OUTCOME_MAP = {
    "1X2": {
        "Home Win": None,   # resolved to home_team name at runtime
        "Draw": "Draw",
        "Away Win": None,   # resolved to away_team name at runtime
    },
    "Over/Under 2.5": {
        "Over 2.5": "Over",
        "Under 2.5": "Under",
    },
    "BTTS": {
        "Yes": "Yes",
        "No": "No",
    },
    "Team Over 1.5": {
        "Home Over 1.5": "Over",
        "Away Over 1.5": "Over",
    },
}


def remove_vig(prices: dict) -> dict:
    """
    Given {outcome_name: decimal_odds} for a single market, return
    {outcome_name: no_vig_probability} with the bookmaker margin stripped
    out via simple proportional (multiplicative) normalisation.
    """
    raw_implied = {name: 1.0 / odds for name, odds in prices.items() if odds and odds > 1.0}
    total = sum(raw_implied.values())
    if total <= 0:
        return {}
    return {name: p / total for name, p in raw_implied.items()}


def best_price(match: dict, market_key: str, outcome_name: str):
    """Scan all bookmakers for a match and return the best (highest) price
    for a given market/outcome, along with which bookmaker offered it."""
    best_odds = None
    best_book = None
    for bm in match.get("bookmakers", []):
        market = bm.get("markets", {}).get(market_key)
        if not market:
            continue
        outcome = market.get(outcome_name)
        if not outcome:
            continue
        price = outcome.get("price")
        if price and (best_odds is None or price > best_odds):
            best_odds = price
            best_book = bm.get("title") or bm.get("key")
    return best_odds, best_book


def _market_no_vig_prob(match: dict, market_key: str, outcome_name: str, all_outcome_names: list):
    """Compute the no-vig probability for one outcome using the *same*
    bookmaker's full set of prices for that market (consensus: first
    bookmaker that quotes all outcomes)."""
    for bm in match.get("bookmakers", []):
        market = bm.get("markets", {}).get(market_key)
        if not market:
            continue
        if not all(name in market for name in all_outcome_names):
            continue
        prices = {name: market[name]["price"] for name in all_outcome_names if market[name].get("price")}
        no_vig = remove_vig(prices)
        if outcome_name in no_vig:
            return no_vig[outcome_name]
    return None


def generate_candidates(match: dict, probs: MatchProbabilities, db: Database, model_confidence: float) -> list:
    """Build every viable value-bet candidate for a single match across all
    supported markets. Returns a list of candidate dicts (not yet filtered
    by threshold or diversity)."""
    candidates = []
    home_team = match["home_team"]
    away_team = match["away_team"]
    league = match["league"]

    def add_candidate(market_label, selection_label, api_market_key, api_outcome_name,
                       all_outcome_names, model_prob):
        odds, bookmaker = best_price(match, api_market_key, api_outcome_name)
        if not odds or odds <= 1.01:
            return
        no_vig_prob = _market_no_vig_prob(match, api_market_key, api_outcome_name, all_outcome_names)

        # Apply learned calibration adjustment (additive, bounded).
        adjustment = db.get_calibration_adjustment(market_label, model_prob)
        calibrated_prob = min(0.98, max(0.02, model_prob + adjustment))

        edge = calibrated_prob - no_vig_prob if no_vig_prob is not None else None
        expected_value = calibrated_prob * odds - 1

        candidates.append({
            "match_id": match["match_id"],
            "league": league,
            "commence_time": match["commence_time"],
            "home_team": home_team,
            "away_team": away_team,
            "market": market_label,
            "selection": selection_label,
            "bookmaker": bookmaker,
            "odds": odds,
            "no_vig_prob": no_vig_prob,
            "model_prob": calibrated_prob,
            "edge": edge,
            "expected_value": expected_value,
            "confidence": model_confidence,
        })

    # --- 1X2 -------------------------------------------------------------
    outcomes_1x2 = [home_team, "Draw", away_team]
    add_candidate("1X2", "Home Win", "h2h", home_team, outcomes_1x2, probs.home_win)
    add_candidate("1X2", "Draw", "h2h", "Draw", outcomes_1x2, probs.draw)
    add_candidate("1X2", "Away Win", "h2h", away_team, outcomes_1x2, probs.away_win)

    # --- Over/Under 2.5 ----------------------------------------------------
    outcomes_ou = ["Over", "Under"]
    add_candidate("Over/Under 2.5", "Over 2.5 Goals", "totals", "Over", outcomes_ou, probs.over_2_5)
    add_candidate("Over/Under 2.5", "Under 2.5 Goals", "totals", "Under", outcomes_ou, probs.under_2_5)

    # --- BTTS (secondary market) ------------------------------------------
    outcomes_btts = ["Yes", "No"]
    add_candidate("BTTS", "Both Teams to Score - Yes", "btts", "Yes", outcomes_btts, probs.btts_yes)
    add_candidate("BTTS", "Both Teams to Score - No", "btts", "No", outcomes_btts, probs.btts_no)

    # --- Team Over 1.5 (secondary market, uses team_totals if the book offers it) ---
    # The Odds API exposes team totals per side; we look for outcomes named
    # after the team with an "Over"/"Under" descriptor embedded in the market.
    for bm in match.get("bookmakers", []):
        tt_market = bm.get("markets", {}).get("team_totals")
        if not tt_market:
            continue
        for outcome_name, outcome in tt_market.items():
            price = outcome.get("price")
            point = outcome.get("point")
            if price is None or point is None:
                continue
            if abs(point - config.OVER_LINE_TEAM) > 0.01:
                continue
            if home_team in outcome_name and "Over" in outcome_name:
                add_candidate("Team Over 1.5", f"{home_team} Over 1.5", "team_totals",
                               outcome_name, [outcome_name], probs.home_over_1_5)
            elif away_team in outcome_name and "Over" in outcome_name:
                add_candidate("Team Over 1.5", f"{away_team} Over 1.5", "team_totals",
                               outcome_name, [outcome_name], probs.away_over_1_5)
        break  # one bookmaker's naming convention is enough to try

    return candidates


def justify(candidate: dict) -> str:
    """Human-readable one-liner explaining why a pick was flagged."""
    edge_pct = f"{candidate['edge']*100:.1f}%" if candidate["edge"] is not None else "n/a"
    no_vig_pct = (
        f"{candidate['no_vig_prob']*100:.1f}%" if candidate["no_vig_prob"] is not None else "n/a"
    )
    return (
        f"Model estimates {candidate['model_prob']*100:.1f}% vs bookmaker's "
        f"no-vig {no_vig_pct} implied probability, an edge of {edge_pct}. "
        f"Expected value {candidate['expected_value']*100:.1f}% at odds {candidate['odds']:.2f}."
    )


def select_top_picks(all_candidates: list, min_value: float = None,
                      max_picks: int = None, max_per_market: int = None) -> list:
    """
    Rank candidates by expected value and greedily select the top picks
    subject to:
      - minimum expected value threshold
      - at most one bet per match
      - at most `max_per_market` bets from the same market
    """
    min_value = min_value if min_value is not None else config.MIN_VALUE_THRESHOLD
    max_picks = max_picks or config.MAX_PICKS_PER_DAY
    max_per_market = max_per_market or config.MAX_PICKS_PER_MARKET

    eligible = [c for c in all_candidates if c["expected_value"] >= min_value]
    eligible.sort(key=lambda c: c["expected_value"], reverse=True)

    picks = []
    used_matches = set()
    market_counts = {}

    for cand in eligible:
        if len(picks) >= max_picks:
            break
        if cand["match_id"] in used_matches:
            continue
        market_count = market_counts.get(cand["market"], 0)
        if market_count >= max_per_market:
            continue
        picks.append(cand)
        used_matches.add(cand["match_id"])
        market_counts[cand["market"]] = market_count + 1

    if len(picks) < max_picks:
        logger.warning(
            "Only found %d/%d value bets meeting the %.1f%% threshold today.",
            len(picks), max_picks, min_value * 100,
        )
    return picks


def build_candidates_for_all_matches(matches: list, db: Database, model: PoissonEloModel = None) -> list:
    """Run the probability model over every match and collect all candidate
    bets across all supported markets."""
    model = model or PoissonEloModel(db)
    all_candidates = []
    for match in matches:
        try:
            probs = model.estimate(match["home_team"], match["away_team"], match["league"])
            confidence = model.confidence(match["home_team"], match["away_team"], match["league"])
            candidates = generate_candidates(match, probs, db, confidence)
            all_candidates.extend(candidates)
        except Exception:
            logger.exception("Failed to evaluate match %s vs %s - skipping",
                              match.get("home_team"), match.get("away_team"))
            continue
    logger.info("Generated %d raw candidate bets from %d matches", len(all_candidates), len(matches))
    return all_candidates
