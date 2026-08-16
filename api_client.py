"""
api_client.py
-------------
Thin, defensive wrapper around The Odds API (https://the-odds-api.com/).
Handles retries with exponential backoff, quota-aware logging, and shapes
the raw JSON into simple dictionaries the rest of the program can use.

If you want to swap in a different provider (Football-Data.org, API-Football
via RapidAPI, etc.) implement the same two public functions --
`fetch_todays_matches()` and `fetch_recent_scores()` -- and the rest of the
codebase does not need to change.
"""

import time
import logging
import datetime as dt

import requests

import config

logger = logging.getLogger("value_bet_finder.api_client")


class ApiClientError(Exception):
    pass


def _request_with_retry(url: str, params: dict) -> requests.Response:
    """GET a URL with exponential backoff retry on transient failures."""
    last_exc = None
    for attempt in range(1, config.HTTP_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                remaining = resp.headers.get("x-requests-remaining")
                used = resp.headers.get("x-requests-used")
                if remaining is not None:
                    logger.info("Odds API quota - used=%s remaining=%s", used, remaining)
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                # transient - retry
                logger.warning(
                    "Transient HTTP %s from %s (attempt %d/%d)",
                    resp.status_code, url, attempt, config.HTTP_MAX_RETRIES,
                )
            else:
                # non-transient error (401, 404, etc.) - do not retry
                resp.raise_for_status()
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(
                "Request error on %s (attempt %d/%d): %s",
                url, attempt, config.HTTP_MAX_RETRIES, exc,
            )
        sleep_for = config.HTTP_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        time.sleep(sleep_for)
    raise ApiClientError(f"Failed to GET {url} after {config.HTTP_MAX_RETRIES} attempts") from last_exc


def _is_today_utc(iso_commence_time: str) -> bool:
    try:
        commence = dt.datetime.fromisoformat(iso_commence_time.replace("Z", "+00:00"))
    except ValueError:
        return False
    today = dt.datetime.now(dt.timezone.utc).date()
    return commence.astimezone(dt.timezone.utc).date() == today


def fetch_todays_matches() -> list:
    """
    Fetch odds for all configured soccer leagues, filtered to matches whose
    kickoff falls on today's date (UTC). Returns a list of match dicts:

    {
      "match_id": str,
      "league": str,
      "commence_time": iso str,
      "home_team": str,
      "away_team": str,
      "bookmakers": [ {"key": ..., "markets": {market_key: {outcome: price}}} ]
    }
    """
    if not config.ODDS_API_KEY:
        raise ApiClientError("ODDS_API_KEY is not configured")

    all_matches = []
    for league in config.SOCCER_LEAGUES:
        league = league.strip()
        if not league:
            continue
        url = f"{config.ODDS_API_BASE_URL}/sports/{league}/odds"
        params = {
            "apiKey": config.ODDS_API_KEY,
            "regions": config.ODDS_REGIONS,
            "markets": config.ODDS_MARKETS,
            "oddsFormat": config.ODDS_FORMAT,
            "dateFormat": "iso",
        }
        try:
            resp = _request_with_retry(url, params)
        except ApiClientError as exc:
            logger.error("Could not fetch odds for league %s: %s", league, exc)
            continue

        try:
            events = resp.json()
        except ValueError:
            logger.error("Non-JSON response for league %s", league)
            continue

        if not isinstance(events, list):
            logger.warning("Unexpected odds payload for league %s: %s", league, events)
            continue

        for ev in events:
            commence_time = ev.get("commence_time")
            if not commence_time or not _is_today_utc(commence_time):
                continue

            bookmakers = []
            for bm in ev.get("bookmakers", []):
                markets = {}
                for m in bm.get("markets", []):
                    key = m.get("key")
                    outcomes = {o["name"]: o for o in m.get("outcomes", [])}
                    markets[key] = outcomes
                bookmakers.append({"key": bm.get("key"), "title": bm.get("title"), "markets": markets})

            all_matches.append({
                "match_id": ev.get("id"),
                "league": league,
                "commence_time": commence_time,
                "home_team": ev.get("home_team"),
                "away_team": ev.get("away_team"),
                "bookmakers": bookmakers,
            })

        logger.info("League %s: %d matches today", league, len(
            [m for m in all_matches if m["league"] == league]))

    logger.info("Fetched %d total matches scheduled today (UTC) across %d leagues",
                len(all_matches), len(config.SOCCER_LEAGUES))
    return all_matches


def fetch_recent_scores(days_from: int = None) -> list:
    """
    Fetch recently completed match scores for all configured leagues so the
    program can settle its own past predictions and update team ratings.
    Returns a list of dicts:
        {"match_id", "league", "completed", "home_team", "away_team",
         "home_score", "away_score"}
    """
    days_from = days_from or config.SCORES_DAYS_FROM
    if not config.ODDS_API_KEY:
        raise ApiClientError("ODDS_API_KEY is not configured")

    all_scores = []
    for league in config.SOCCER_LEAGUES:
        league = league.strip()
        if not league:
            continue
        url = f"{config.ODDS_API_BASE_URL}/sports/{league}/scores"
        params = {
            "apiKey": config.ODDS_API_KEY,
            "daysFrom": days_from,
            "dateFormat": "iso",
        }
        try:
            resp = _request_with_retry(url, params)
        except ApiClientError as exc:
            logger.error("Could not fetch scores for league %s: %s", league, exc)
            continue

        try:
            events = resp.json()
        except ValueError:
            logger.error("Non-JSON scores response for league %s", league)
            continue

        if not isinstance(events, list):
            continue

        for ev in events:
            if not ev.get("completed"):
                continue
            scores = ev.get("scores")
            if not scores:
                continue
            score_map = {s["name"]: s.get("score") for s in scores}
            home_team = ev.get("home_team")
            away_team = ev.get("away_team")
            try:
                home_score = int(score_map.get(home_team))
                away_score = int(score_map.get(away_team))
            except (TypeError, ValueError):
                continue

            all_scores.append({
                "match_id": ev.get("id"),
                "league": league,
                "completed": True,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
            })

    logger.info("Fetched %d completed match results for settlement", len(all_scores))
    return all_scores
