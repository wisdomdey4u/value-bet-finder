"""
database.py
-----------
All SQLite persistence lives here: prediction history, team strength
ratings, league averages and calibration tables. Every other module talks
to the database exclusively through the Database class below.
"""

import sqlite3
import logging
import datetime as dt
from contextlib import contextmanager

import config

logger = logging.getLogger("value_bet_finder.database")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    match_id TEXT NOT NULL,
    league TEXT,
    commence_time TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    bookmaker TEXT,
    odds REAL NOT NULL,
    no_vig_prob REAL,
    model_prob REAL NOT NULL,
    edge REAL NOT NULL,
    expected_value REAL NOT NULL,
    justification TEXT,
    result TEXT DEFAULT 'PENDING',   -- PENDING / WON / LOST / VOID
    home_score INTEGER,
    away_score INTEGER,
    settled_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_predictions_result ON predictions(result);

CREATE TABLE IF NOT EXISTS team_ratings (
    team TEXT NOT NULL,
    league TEXT NOT NULL,
    attack REAL NOT NULL DEFAULT 1.0,
    defense REAL NOT NULL DEFAULT 1.0,
    elo REAL NOT NULL DEFAULT 1500.0,
    matches_played INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT,
    PRIMARY KEY (team, league)
);

CREATE TABLE IF NOT EXISTS league_averages (
    league TEXT PRIMARY KEY,
    avg_home_goals REAL NOT NULL DEFAULT 1.5,
    avg_away_goals REAL NOT NULL DEFAULT 1.15,
    matches_seen INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS calibration (
    market TEXT NOT NULL,
    bucket_low REAL NOT NULL,
    bucket_high REAL NOT NULL,
    predicted_avg REAL,
    actual_rate REAL,
    sample_size INTEGER NOT NULL DEFAULT 0,
    adjustment REAL NOT NULL DEFAULT 0.0,
    updated_at TEXT,
    PRIMARY KEY (market, bucket_low)
);

CREATE TABLE IF NOT EXISTS run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    matches_seen INTEGER,
    candidates_generated INTEGER,
    picks_sent INTEGER,
    status TEXT,
    notes TEXT
);
"""


class Database:
    def __init__(self, path=None):
        self.path = path or config.DATABASE_PATH
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)
        logger.info("Database schema ready at %s", self.path)

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    def insert_prediction(self, pred: dict) -> int:
        """Insert one prediction row and return its id."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO predictions
                (created_at, match_id, league, commence_time, home_team, away_team,
                 market, selection, bookmaker, odds, no_vig_prob, model_prob,
                 edge, expected_value, justification)
                VALUES (:created_at, :match_id, :league, :commence_time, :home_team,
                        :away_team, :market, :selection, :bookmaker, :odds,
                        :no_vig_prob, :model_prob, :edge, :expected_value, :justification)
                """,
                pred,
            )
            return cur.lastrowid

    def get_pending_predictions(self, older_than_hours: int = 2):
        """Predictions whose match should have finished by now."""
        cutoff = (dt.datetime.utcnow() - dt.timedelta(hours=older_than_hours)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions WHERE result = 'PENDING' AND commence_time <= ?",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]

    def settle_prediction(self, pred_id: int, result: str, home_score: int, away_score: int):
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE predictions
                SET result = ?, home_score = ?, away_score = ?, settled_at = ?
                WHERE id = ?
                """,
                (result, home_score, away_score, dt.datetime.utcnow().isoformat(), pred_id),
            )

    def get_settled_predictions(self, market: str = None):
        query = "SELECT * FROM predictions WHERE result IN ('WON','LOST')"
        params = ()
        if market:
            query += " AND market = ?"
            params = (market,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def count_todays_predictions(self):
        today = dt.datetime.utcnow().strftime("%Y-%m-%d")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM predictions WHERE created_at LIKE ?",
                (f"{today}%",),
            ).fetchone()
            return row["c"]

    # ------------------------------------------------------------------
    # Team ratings
    # ------------------------------------------------------------------
    def get_team_rating(self, team: str, league: str) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM team_ratings WHERE team = ? AND league = ?",
                (team, league),
            ).fetchone()
            if row:
                return dict(row)
            return {
                "team": team,
                "league": league,
                "attack": 1.0,
                "defense": 1.0,
                "elo": config.DEFAULT_ELO,
                "matches_played": 0,
                "updated_at": None,
            }

    def upsert_team_rating(self, team: str, league: str, attack: float, defense: float,
                            elo: float, matches_played: int):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO team_ratings (team, league, attack, defense, elo, matches_played, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(team, league) DO UPDATE SET
                    attack=excluded.attack,
                    defense=excluded.defense,
                    elo=excluded.elo,
                    matches_played=excluded.matches_played,
                    updated_at=excluded.updated_at
                """,
                (team, league, attack, defense, elo, matches_played, dt.datetime.utcnow().isoformat()),
            )

    # ------------------------------------------------------------------
    # League averages
    # ------------------------------------------------------------------
    def get_league_average(self, league: str) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM league_averages WHERE league = ?", (league,)
            ).fetchone()
            if row:
                return dict(row)
            return {
                "league": league,
                "avg_home_goals": config.DEFAULT_LEAGUE_AVG_HOME_GOALS,
                "avg_away_goals": config.DEFAULT_LEAGUE_AVG_AWAY_GOALS,
                "matches_seen": 0,
                "updated_at": None,
            }

    def update_league_average(self, league: str, home_goal: int, away_goal: int):
        cur_avg = self.get_league_average(league)
        n = cur_avg["matches_seen"]
        new_n = n + 1
        # incremental mean update
        new_home = cur_avg["avg_home_goals"] + (home_goal - cur_avg["avg_home_goals"]) / new_n
        new_away = cur_avg["avg_away_goals"] + (away_goal - cur_avg["avg_away_goals"]) / new_n
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO league_averages (league, avg_home_goals, avg_away_goals, matches_seen, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(league) DO UPDATE SET
                    avg_home_goals=excluded.avg_home_goals,
                    avg_away_goals=excluded.avg_away_goals,
                    matches_seen=excluded.matches_seen,
                    updated_at=excluded.updated_at
                """,
                (league, new_home, new_away, new_n, dt.datetime.utcnow().isoformat()),
            )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def upsert_calibration(self, market: str, bucket_low: float, bucket_high: float,
                            predicted_avg: float, actual_rate: float, sample_size: int,
                            adjustment: float):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO calibration
                (market, bucket_low, bucket_high, predicted_avg, actual_rate, sample_size, adjustment, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(market, bucket_low) DO UPDATE SET
                    bucket_high=excluded.bucket_high,
                    predicted_avg=excluded.predicted_avg,
                    actual_rate=excluded.actual_rate,
                    sample_size=excluded.sample_size,
                    adjustment=excluded.adjustment,
                    updated_at=excluded.updated_at
                """,
                (market, bucket_low, bucket_high, predicted_avg, actual_rate,
                 sample_size, adjustment, dt.datetime.utcnow().isoformat()),
            )

    def get_calibration_adjustment(self, market: str, predicted_prob: float) -> float:
        """Return the additive adjustment learned for this market/probability bucket, or 0.0."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT adjustment FROM calibration
                WHERE market = ? AND ? >= bucket_low AND ? < bucket_high
                ORDER BY sample_size DESC LIMIT 1
                """,
                (market, predicted_prob, predicted_prob),
            ).fetchone()
            return row["adjustment"] if row else 0.0

    # ------------------------------------------------------------------
    # Run log
    # ------------------------------------------------------------------
    def log_run(self, matches_seen, candidates_generated, picks_sent, status, notes=""):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_log (run_at, matches_seen, candidates_generated, picks_sent, status, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (dt.datetime.utcnow().isoformat(), matches_seen, candidates_generated,
                 picks_sent, status, notes),
            )
