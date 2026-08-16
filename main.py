#!/usr/bin/env python3
"""
main.py
-------
Orchestrates one full daily run of the Value Bet Finder:

  1. Set up logging.
  2. Validate required environment variables / secrets.
  3. Open (or create) the SQLite history database.
  4. Run the learning cycle: settle yesterday's (and earlier) pending
     predictions against real results, update team ratings, and
     recalibrate probability buckets from accumulated history.
  5. Fetch today's matches + odds (UTC) from The Odds API.
  6. Run the Poisson/Elo probability model over every match and build
     every candidate value bet across all supported markets.
  7. Filter by minimum value threshold and confidence, then select up to
     NUM_PICKS diversified picks (<=1 per match, <=2 per market).
  8. Persist every selected pick to the database (so it can be settled and
     learned from later).
  9. Compose and send the summary email, with retry on transient failures.
 10. Log a one-line run summary to the `run_log` table for observability.

Run this as:
    python main.py

On PythonAnywhere, point a scheduled ("Tasks") job at this exact command,
scheduled for 09:00 UTC, so there is a comfortable margin before the
10:00 UTC delivery deadline even if the odds API or SMTP is briefly slow.
"""

import sys
import logging
import datetime as dt
from logging.handlers import RotatingFileHandler

import config
from database import Database
from api_client import fetch_todays_matches, ApiClientError
from value_calculator import build_candidates_for_all_matches, select_top_picks, justify
from learning import run_learning_cycle
from email_sender import compose_email, send_email


def setup_logging():
    """Configure root logging: rotating file handler + optional console."""
    root = logging.getLogger()
    root.setLevel(config.LOG_LEVEL_VALUE)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        config.LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    # requests/urllib3 are noisy at INFO; keep them at WARNING.
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def run_daily_cycle() -> int:
    """Executes the full pipeline once. Returns a process exit code
    (0 = success, 1 = completed with issues, 2 = fatal error)."""
    logger = logging.getLogger("value_bet_finder.main")
    run_started = dt.datetime.now(dt.timezone.utc)
    logger.info("=" * 70)
    logger.info("Value Bet Finder - daily run starting at %s", run_started.isoformat())

    try:
        config.validate_required_config()
    except EnvironmentError as exc:
        logger.error(str(exc))
        return 2

    db = Database()

    # ------------------------------------------------------------------
    # 1. Learning cycle: settle past predictions + recalibrate.
    #    Safe to run every day; it is a no-op when there is nothing new
    #    to settle and the calibration step short-circuits with too few
    #    samples per market/bucket.
    # ------------------------------------------------------------------
    try:
        run_learning_cycle(db)
    except Exception:
        logger.exception("Learning cycle raised an unexpected error; continuing with today's picks anyway")

    # ------------------------------------------------------------------
    # 2. Fetch today's matches + odds.
    # ------------------------------------------------------------------
    try:
        matches = fetch_todays_matches()
    except ApiClientError:
        logger.exception("Fatal: could not fetch matches from the odds API")
        db.log_run(0, 0, 0, "FAILED", "Odds API fetch failed")
        _send_failure_email(run_started, "Could not fetch match/odds data from the API. See finder.log.")
        return 2

    if not matches:
        logger.warning("No matches found for today (UTC). Sending an empty-picks email.")
        subject, text_body, html_body = compose_email([], run_started, 0)
        _deliver(subject, text_body, html_body)
        db.log_run(0, 0, 0, "NO_MATCHES", "")
        return 0

    # ------------------------------------------------------------------
    # 3. Build candidates across every match/market.
    # ------------------------------------------------------------------
    all_candidates = build_candidates_for_all_matches(matches, db)

    # Drop candidates we don't yet have enough team history to trust, and
    # anything with an implausibly large "value" (usually bad/stale data
    # rather than a genuine edge).
    filtered = [
        c for c in all_candidates
        if c["confidence"] >= config.MIN_MODEL_CONFIDENCE
        and c["expected_value"] <= getattr(config, "MAX_VALUE_THRESHOLD", 0.60)
        and c["model_prob"] >= 0.0
    ]
    logger.info("%d/%d candidates passed confidence/sanity filters", len(filtered), len(all_candidates))

    # ------------------------------------------------------------------
    # 4. Select the final diversified picks.
    # ------------------------------------------------------------------
    picks = select_top_picks(filtered)
    for pick in picks:
        pick["justification"] = justify(pick)

    # ------------------------------------------------------------------
    # 5. Persist picks to the database so they can be settled & learned
    #    from once the matches have been played.
    # ------------------------------------------------------------------
    for pick in picks:
        try:
            pred_row = {
                "created_at": run_started.isoformat(),
                "match_id": pick["match_id"],
                "league": pick["league"],
                "commence_time": pick["commence_time"],
                "home_team": pick["home_team"],
                "away_team": pick["away_team"],
                "market": pick["market"],
                "selection": pick["selection"],
                "bookmaker": pick.get("bookmaker"),
                "odds": pick["odds"],
                "no_vig_prob": pick.get("no_vig_prob"),
                "model_prob": pick["model_prob"],
                "edge": pick.get("edge") if pick.get("edge") is not None else 0.0,
                "expected_value": pick["expected_value"],
                "justification": pick["justification"],
            }
            db.insert_prediction(pred_row)
        except Exception:
            logger.exception("Failed to persist prediction for %s vs %s",
                              pick.get("home_team"), pick.get("away_team"))

    # ------------------------------------------------------------------
    # 6. Compose + send the email.
    # ------------------------------------------------------------------
    subject, text_body, html_body = compose_email(picks, run_started, len(matches))
    sent_ok = _deliver(subject, text_body, html_body)

    status = "OK" if picks and sent_ok else ("SENT_NO_PICKS" if sent_ok else "EMAIL_FAILED")
    db.log_run(len(matches), len(all_candidates), len(picks), status,
               f"{len(picks)} picks selected from {len(filtered)} filtered candidates")

    if len(picks) < config.MAX_PICKS_PER_DAY:
        logger.warning(
            "Delivered only %d/%d picks today (not enough candidates cleared the "
            "%.0f%% value threshold / diversity rules).",
            len(picks), config.MAX_PICKS_PER_DAY, config.MIN_VALUE_THRESHOLD * 100,
        )

    if not sent_ok:
        logger.error("Email delivery ultimately failed after retries.")
        return 1

    run_finished = dt.datetime.now(dt.timezone.utc)
    logger.info("Run complete in %.1fs. Picks sent: %d", (run_finished - run_started).total_seconds(), len(picks))
    return 0


def _deliver(subject: str, text_body: str, html_body: str) -> bool:
    logger = logging.getLogger("value_bet_finder.main")
    if config.DRY_RUN:
        logger.info("DRY_RUN enabled - skipping actual email send. Subject: %s", subject)
        return True
    return send_email(subject, text_body, html_body)


def _send_failure_email(run_started: dt.datetime, reason: str):
    """Best-effort notification email when the pipeline fails before it can
    even generate picks, so a human finds out promptly rather than the
    silence being mistaken for 'no value bets today'."""
    logger = logging.getLogger("value_bet_finder.main")
    try:
        subject = f"Value Bet Finder - RUN FAILED {run_started.strftime('%Y-%m-%d')}"
        text_body = f"The daily run failed before it could produce picks.\n\nReason: {reason}\n"
        html_body = f"<p><b>The daily run failed before it could produce picks.</b></p><p>{reason}</p>"
        send_email(subject, text_body, html_body)
    except Exception:
        logger.exception("Even the failure-notification email could not be sent")


if __name__ == "__main__":
    setup_logging()
    exit_code = run_daily_cycle()
    sys.exit(exit_code)
