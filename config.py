"""
config.py
---------
Centralised configuration loaded from environment variables (or a local .env
file when developing outside PythonAnywhere). Nothing else in the codebase
should call os.environ directly -- everything goes through this module so
there is a single, auditable place where configuration lives.
"""

import os
import logging

# python-dotenv is only used for local development convenience. On
# PythonAnywhere you set real environment variables via the Web tab / bash
# console, so a missing .env file is not an error.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Required / core credentials
# ---------------------------------------------------------------------------
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_TO = os.environ.get("EMAIL_TO", EMAIL_USER)  # defaults to sending to self

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "history.db"))
LOG_PATH = os.environ.get("LOG_PATH", os.path.join(BASE_DIR, "finder.log"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL_VALUE = getattr(logging, LOG_LEVEL, logging.INFO)

# ---------------------------------------------------------------------------
# The Odds API settings
# ---------------------------------------------------------------------------
ODDS_API_BASE_URL = os.environ.get("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")

# Soccer league "sport keys" as defined by The Odds API. Extend this list to
# cover more leagues; each extra league costs additional API quota.
SOCCER_LEAGUES = os.environ.get(
    "SOCCER_LEAGUES",
    "soccer_epl,soccer_spain_la_liga,soccer_italy_serie_a,"
    "soccer_germany_bundesliga,soccer_france_ligue_one,"
    "soccer_uefa_champs_league"
).split(",")

ODDS_REGIONS = os.environ.get("ODDS_REGIONS", "eu,uk")
ODDS_FORMAT = "decimal"
# Markets requested from the API. IMPORTANT: "btts" and "team_totals" are
# billed as "additional markets" by The Odds API and are gated to paid
# plans on some tiers -- a free-tier key requesting them can get the
# *entire* request rejected (not just those markets silently dropped).
# Default to the two core markets, which are available on every plan
# including free. Once you've confirmed matches are coming through, you
# can opt back in via the ODDS_MARKETS env var, e.g.
# "h2h,totals,btts,team_totals" -- see README "Troubleshooting" section.
ODDS_MARKETS = os.environ.get("ODDS_MARKETS", "h2h,spreads,totals")

# ---------------------------------------------------------------------------
# Betting / value engine parameters
# ---------------------------------------------------------------------------
MIN_VALUE_THRESHOLD = _get_float("MIN_VALUE_THRESHOLD", 0.05)   # 5% edge minimum
# Sanity ceiling: reject anything implausibly good, which usually signals
# bad/stale data rather than a genuine opportunity.
MAX_VALUE_THRESHOLD = _get_float("MAX_VALUE_THRESHOLD", 0.60)   # 60% edge maximum
MIN_MODEL_CONFIDENCE = _get_float("MIN_MODEL_CONFIDENCE", 0.02)  # min matches of history to trust team ratings
MAX_PICKS_PER_DAY = _get_int("MAX_PICKS_PER_DAY", 3)
MAX_PICKS_PER_MARKET = _get_int("MAX_PICKS_PER_MARKET", 2)
MAX_GOALS = _get_int("MAX_GOALS", 10)            # truncation point for the Poisson score grid
OVER_LINE_MAIN = 2.5
OVER_LINE_TEAM = 1.5

# League-wide fallback averages used when we have no historical data yet.
DEFAULT_LEAGUE_AVG_HOME_GOALS = _get_float("DEFAULT_LEAGUE_AVG_HOME_GOALS", 1.5)
DEFAULT_LEAGUE_AVG_AWAY_GOALS = _get_float("DEFAULT_LEAGUE_AVG_AWAY_GOALS", 1.15)

# Elo settings
DEFAULT_ELO = _get_float("DEFAULT_ELO", 1500.0)
ELO_K_FACTOR = _get_float("ELO_K_FACTOR", 20.0)
ELO_GOAL_DIFF_MULT = _get_bool("ELO_GOAL_DIFF_MULT", True)

# Rating learning rate (EWMA) applied to attack/defense strengths on settle.
RATING_LEARNING_RATE = _get_float("RATING_LEARNING_RATE", 0.08)

# How many days back to look when settling / fetching scores.
SCORES_DAYS_FROM = _get_int("SCORES_DAYS_FROM", 3)

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = _get_int("SMTP_PORT", 587)
EMAIL_MAX_RETRIES = _get_int("EMAIL_MAX_RETRIES", 3)
EMAIL_RETRY_BACKOFF_SECONDS = _get_int("EMAIL_RETRY_BACKOFF_SECONDS", 15)

# ---------------------------------------------------------------------------
# HTTP retry settings
# ---------------------------------------------------------------------------
HTTP_MAX_RETRIES = _get_int("HTTP_MAX_RETRIES", 4)
HTTP_BACKOFF_BASE_SECONDS = _get_float("HTTP_BACKOFF_BASE_SECONDS", 1.5)
HTTP_TIMEOUT_SECONDS = _get_int("HTTP_TIMEOUT_SECONDS", 20)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
# If true, the pipeline runs and logs everything as normal but never
# actually sends an email (useful for local testing / dry runs).
DRY_RUN = _get_bool("DRY_RUN", False)


def validate_required_config():
    """Raise a clear error early if mandatory secrets are missing."""
    missing = []
    if not ODDS_API_KEY:
        missing.append("ODDS_API_KEY")
    if not EMAIL_USER:
        missing.append("EMAIL_USER")
    if not EMAIL_PASSWORD:
        missing.append("EMAIL_PASSWORD")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them in your PythonAnywhere Web/Task environment or in a local .env file."
        )
