"""
tests/test_core.py
-------------------
Lightweight unit tests for the parts of the system that are pure functions
(no network, no email). Run with:

    python -m pytest tests/ -v

or, without pytest installed:

    python -m unittest tests.test_core -v
"""

import os
import sys
import unittest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("ODDS_API_KEY", "test_key")
os.environ.setdefault("EMAIL_USER", "test@example.com")
os.environ.setdefault("EMAIL_PASSWORD", "test_password")

import config  # noqa: E402
from probability_models import build_scoreline_grid, derive_markets_from_grid  # noqa: E402
from value_calculator import remove_vig, select_top_picks  # noqa: E402
from database import Database  # noqa: E402


class TestScorelineGrid(unittest.TestCase):
    def test_grid_sums_to_one(self):
        grid = build_scoreline_grid(1.4, 1.1)
        self.assertAlmostEqual(grid.sum(), 1.0, places=6)

    def test_markets_are_internally_consistent(self):
        grid = build_scoreline_grid(1.5, 1.2)
        markets = derive_markets_from_grid(grid)
        # 1X2 probabilities must sum to (approximately) 1.
        total_1x2 = markets["home_win"] + markets["draw"] + markets["away_win"]
        self.assertAlmostEqual(total_1x2, 1.0, places=6)
        # Over + under must sum to 1.
        self.assertAlmostEqual(markets["over_main"] + markets["under_main"], 1.0, places=6)
        self.assertAlmostEqual(markets["btts_yes"] + markets["btts_no"], 1.0, places=6)

    def test_stronger_home_side_favoured(self):
        # A much higher home lambda should produce a higher home-win prob
        # than an evenly matched game.
        strong_grid = build_scoreline_grid(2.2, 0.8)
        even_grid = build_scoreline_grid(1.3, 1.3)
        strong_markets = derive_markets_from_grid(strong_grid)
        even_markets = derive_markets_from_grid(even_grid)
        self.assertGreater(strong_markets["home_win"], even_markets["home_win"])


class TestRemoveVig(unittest.TestCase):
    def test_no_vig_probabilities_sum_to_one(self):
        # Typical over-round market: 1.90 / 3.80 / 4.20 implied sum > 1.
        prices = {"Home": 1.90, "Draw": 3.80, "Away": 4.20}
        no_vig = remove_vig(prices)
        self.assertAlmostEqual(sum(no_vig.values()), 1.0, places=6)
        # Favourite should still have the highest no-vig probability.
        self.assertEqual(max(no_vig, key=no_vig.get), "Home")

    def test_handles_empty_or_invalid_prices(self):
        self.assertEqual(remove_vig({}), {})
        self.assertEqual(remove_vig({"Home": 0, "Away": None}), {})


class TestSelectTopPicks(unittest.TestCase):
    def _candidate(self, match_id, market, ev, selection="Home Win"):
        return {
            "match_id": match_id,
            "league": "soccer_epl",
            "commence_time": "2026-08-15T12:30:00Z",
            "home_team": f"Home{match_id}",
            "away_team": f"Away{match_id}",
            "market": market,
            "selection": selection,
            "bookmaker": "TestBook",
            "odds": 2.0,
            "no_vig_prob": 0.45,
            "model_prob": 0.45 + ev / 2.0,
            "edge": ev / 2.0,
            "expected_value": ev,
            "confidence": 0.9,
        }

    def test_respects_min_value_threshold(self):
        candidates = [self._candidate(1, "1X2", 0.02), self._candidate(2, "1X2", 0.10)]
        picks = select_top_picks(candidates, min_value=0.05, max_picks=3, max_per_market=2)
        self.assertEqual(len(picks), 1)
        self.assertEqual(picks[0]["match_id"], 2)

    def test_at_most_one_bet_per_match(self):
        candidates = [
            self._candidate(1, "1X2", 0.20, "Home Win"),
            self._candidate(1, "Over/Under 2.5", 0.15, "Over 2.5 Goals"),
        ]
        picks = select_top_picks(candidates, min_value=0.05, max_picks=3, max_per_market=2)
        self.assertEqual(len(picks), 1)

    def test_market_diversity_cap(self):
        candidates = [
            self._candidate(1, "1X2", 0.30),
            self._candidate(2, "1X2", 0.25),
            self._candidate(3, "1X2", 0.20),
        ]
        picks = select_top_picks(candidates, min_value=0.05, max_picks=3, max_per_market=2)
        # Only 2 allowed from the same market even though 3 qualify.
        self.assertEqual(len(picks), 2)


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(path=self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_insert_and_settle_prediction(self):
        pred = {
            "created_at": "2026-08-15T08:00:00",
            "match_id": "abc123",
            "league": "soccer_epl",
            "commence_time": "2026-08-15T12:30:00",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "market": "1X2",
            "selection": "Home Win",
            "bookmaker": "TestBook",
            "odds": 2.10,
            "no_vig_prob": 0.44,
            "model_prob": 0.50,
            "edge": 0.06,
            "expected_value": 0.05,
            "justification": "test",
        }
        pred_id = self.db.insert_prediction(pred)
        self.assertIsInstance(pred_id, int)

        self.db.settle_prediction(pred_id, "WON", 2, 1)
        settled = self.db.get_settled_predictions(market="1X2")
        self.assertEqual(len(settled), 1)
        self.assertEqual(settled[0]["result"], "WON")

    def test_team_rating_defaults(self):
        rating = self.db.get_team_rating("Nonexistent FC", "soccer_epl")
        self.assertEqual(rating["attack"], 1.0)
        self.assertEqual(rating["elo"], config.DEFAULT_ELO)


if __name__ == "__main__":
    unittest.main()
