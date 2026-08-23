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
    """GET a URL with exponential backoff retry on *transient* failures only.

    Non-transient client errors (401 bad/expired key, 403 forbidden, 404
    unknown sport key, 422 unsupported market for this plan, etc.) are
    NOT retried -- retrying a permanent error just burns API quota four
    times over for no benefit. Instead we fail fast and log the API's own
    error message (surfaced in the response body) so the real cause is
    visible in finder.log instead of a generic timeout-looking message.
    """
    last_exc = None
    for attempt in range(1, config.HTTP_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(
                "Request error on %s (attempt %d/%d): %s",
                url, attempt, config.HTTP_MAX_RETRIES, exc,
            )
            time.sleep(config.HTTP_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
            continue

        if resp.status_code == 200:
            remaining = resp.headers.get("x-requests-remaining")
            used = resp.headers.get("x-requests-used")
            if remaining is not None:
                logger.info("Odds API quota - used=%s remaining=%s", used, remaining)
            return resp

        # Truncate defensively -- error bodies are normally short JSON, but
        # never let a malformed/huge response blow up the log file.
        body_snippet = resp.text[:500] if resp.text else "<empty body>"

        if resp.status_code in (429, 500, 502, 503, 504):
            logger.warning(
                "Transient HTTP %s from %s (attempt %d/%d): %s",
                resp.status_code, url, attempt, config.HTTP_MAX_RETRIES, body_snippet,
            )
            time.sleep(config.HTTP_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
            continue

        # Non-transient - fail immediately with the API's own message.
        logger.error(
            "Non-transient HTTP %s from %s - NOT retrying. Response body: %s",
            resp.status_code, url, body_snippet,
        )
        raise ApiClientError(f"HTTP {resp.status_code} from {url}: {body_snippet}")

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
    leagues_with_errors = []
    total_raw_events = 0

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
            leagues_with_errors.append(league)
            continue

        try:
            events = resp.json()
        except ValueError:
            logger.error("Non-JSON response for league %s", league)
            leagues_with_errors.append(league)
            continue

        if not isinstance(events, list):
            logger.warning("Unexpected odds payload for league %s: %s", league, events)
            leagues_with_errors.append(league)
            continue

        total_raw_events += len(events)
        matches_today_this_league = 0

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
            matches_today_this_league += 1

        logger.info("League %s: %d raw events returned, %d fall within today's UTC window",
                     league, len(events), matches_today_this_league)

    if leagues_with_errors:
        logger.error(
            "%d/%d leagues failed to fetch entirely this run: %s. "
            "Check the error lines above for the API's own message "
            "(common causes: invalid/expired ODDS_API_KEY, monthly quota "
            "exhausted, or a requested market not available on your plan).",
            len(leagues_with_errors), len(config.SOCCER_LEAGUES), leagues_with_errors,
        )
    elif total_raw_events == 0:
        logger.warning(
            "All %d leagues responded successfully but returned 0 events in "
            "total (not just 0 after date filtering). This usually means "
            "the API key/plan is valid but there is genuinely no upcoming "
            "schedule data for these sport keys right now (e.g. off-season, "
            "or the leagues configured in SOCCER_LEAGUES have no fixtures "
            "currently listed) -- it is not a bug in this program.",
        )
    elif len(all_matches) == 0:
        logger.warning(
            "Leagues returned %d raw events in total, but none fall within "
            "today's UTC window. This is normal on days with no fixtures "
            "in your configured leagues (e.g. a mid-week gap, or all of "
            "today's games are actually tomorrow relative to UTC).",
            total_raw_events,
        )

    logger.info("Fetched %d total matches scheduled today (UTC) across %d leagues (%d raw events seen)",
                len(all_matches), len(config.SOCCER_LEAGUES), total_raw_events)
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
