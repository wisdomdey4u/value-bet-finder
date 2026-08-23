# Value Bet Finder

A self-improving daily football (soccer) value-betting analysis tool. Every
match day it:

1. Pulls today's fixtures + odds (1X2, Over/Under 2.5, BTTS, Team Over 1.5)
   from [The Odds API](https://the-odds-api.com/).
2. Estimates "true" probabilities with a Poisson + Elo goal model
   (Dixon-Coles corrected).
3. Strips the bookmaker's margin from the odds (no-vig probability) and
   compares it to the model's probability to find positive-expected-value
   bets.
4. Selects up to **3 diversified picks** (max 1 bet per match, max 2 bets
   from the same market) above a configurable minimum edge.
5. Emails you the picks via Gmail SMTP.
6. Records every prediction in a local SQLite database, later settles it
   against the real result, and uses that history to **recalibrate its own
   probabilities and team ratings** — so it should get better over time.

> ⚠️ **This is a statistical modelling tool, not financial advice.**
> Sports betting carries real financial risk; "positive expected value"
> according to a model is not a guarantee of profit. Bet responsibly, and
> check that sports betting is legal in your jurisdiction.

---

## 1. Project layout

```
value_bet_finder/
├── main.py                 # orchestrates the full daily run — run this
├── diagnose_api.py          # standalone Odds API connectivity/plan checker — run this FIRST if matches = 0
├── config.py                # all configuration, read from env vars
├── api_client.py            # The Odds API integration (odds + scores), retry/backoff
├── probability_models.py    # Poisson/Elo/Dixon-Coles probability engine
├── value_calculator.py      # no-vig removal, value calc, pick selection
├── database.py               # SQLite persistence (predictions, ratings, calibration)
├── learning.py               # settlement + self-recalibration
├── email_sender.py           # Gmail SMTP composition + sending, with retry
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md                 # you are here
├── .github/
│   └── workflows/
│       └── daily_value_bets.yml   # GitHub Actions schedule — see §6
└── tests/
    └── test_core.py          # unit tests for the pure-function core logic
```

Each module has a single responsibility and only talks to the others
through plain function calls / the `Database` class — there's no hidden
global state, which makes it straightforward to swap out any one piece
(e.g. plug in a different probability model, or a different odds provider)
without touching the rest.

---

## 2. How the value calculation works

For every match and every supported market/selection:

```
raw_implied_probability   = 1 / decimal_odds
no_vig_market_probability = raw_implied_probability normalised across
                             all outcomes in that market (removes the
                             bookmaker's overround)
model_probability          = Poisson/Elo estimate, plus any learned
                              calibration adjustment
estimated_edge             = model_probability − no_vig_market_probability
expected_value             = model_probability × decimal_odds − 1
```

A bet only becomes a *candidate* pick if `expected_value` clears
`MIN_VALUE_THRESHOLD` (5% by default) and the team-rating confidence for
that match is high enough (`MIN_MODEL_CONFIDENCE`). Candidates are then
ranked by expected value and the top 3 are chosen subject to:

- **At most 1 bet per match** (no correlated multi-market stacking on one
  game).
- **At most 2 bets from the same market** across the whole day's picks.

If fewer than 3 bets clear the bar, the program still emails whatever it
found (and logs a warning) rather than silently sending nothing — per the
"early warning" requirement.

---

## 3. Probability model

- **Poisson goal model**: each team has an `attack` and `defense`
  multiplier (1.0 = league average). Expected goals for a match are
  `league_avg_goals × attacker's_attack × opponent's_defense`, nudged by
  the Elo rating differential between the two sides.
- **Elo ratings** start every new team at 1500 and are updated after every
  settled result using a standard logistic expectation with a goal-margin
  multiplier.
- **Dixon-Coles correction** is applied to the four low-scoring cells
  (0-0, 1-0, 0-1, 1-1) of the scoreline grid, correcting a well-documented
  bias in the naive independent-Poisson model.
- All four markets (1X2, Over/Under 2.5, BTTS, Team Over 1.5) are derived
  from the **same scoreline grid**, so they are always mutually
  consistent with each other.
- `probability_models.PoissonEloModel` implements a small `estimate()` /
  `confidence()` interface — you can drop in an alternative model (e.g.
  an xG-based or trained ML model) by implementing the same interface and
  passing it to `value_calculator.build_candidates_for_all_matches(...)`.

### Cold start

On day one, every team starts at league-average attack/defense (1.0/1.0)
and Elo 1500, i.e. the model has no opinion yet and will mostly reflect
the bookmaker's own pricing (so it should rarely flag "value" until it has
seen a handful of results per team). This is intentional: it's safer to
be quiet than to bet confidently on zero information. `confidence()` gates
this explicitly and is checked in `main.py`.

---

## 4. Self-improvement / learning loop

Every run, before looking at today's matches, `learning.run_learning_cycle()`:

1. **Settles** predictions whose match has finished: it fetches recent
   scores from The Odds API's `/scores` endpoint, grades each pending
   prediction (WON/LOST), and stores the actual scoreline.
2. **Updates team ratings**: attack/defense multipliers are nudged with an
   exponentially-weighted moving average toward "how many goals did this
   team actually score/concede vs. how many were expected", and Elo is
   updated with a standard logistic update (scaled by goal margin).
3. **Recalibrates**: settled predictions are grouped into 10-point-wide
   probability buckets per market (e.g. "1X2 predictions between 50-60%").
   The average predicted probability in each bucket is compared to the
   actual win rate; the (shrunk, to avoid overreacting to small samples)
   difference is stored as an additive correction and applied to future
   predictions in that same probability range. This is a simple, robust
   stand-in for isotonic/Platt-scaling calibration that needs no extra
   dependencies and degrades safely to "no adjustment" until there's
   enough data (`MIN_SAMPLES_PER_BUCKET`, default 8 per bucket).

Both steps are cheap and safe to run every single day — they no-op
gracefully when there's nothing new to settle or too little data to
calibrate on, so there's no need for a separate weekly job.

---

## 5. Local setup (for testing before you deploy)

```bash
git clone <this project> value_bet_finder
cd value_bet_finder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your real ODDS_API_KEY, EMAIL_USER, EMAIL_PASSWORD

python -m unittest discover -s tests -v   # sanity check the core logic

DRY_RUN=true python main.py               # full run, no email actually sent
python main.py                            # full run, sends a real email
```

If today's run reports 0 matches, run `python diagnose_api.py` first — see
§8 Troubleshooting for what each part of its output means.

### Getting a Gmail App Password

Gmail SMTP will reject your normal account password. You need an
**App Password**:

1. Enable 2-Step Verification on the Google account:
   https://myaccount.google.com/security
2. Create an App Password at:
   https://myaccount.google.com/apppasswords
3. Use the 16-character generated password as `EMAIL_PASSWORD` (not your
   normal login password).

### Getting an Odds API key

Sign up for a free-tier key at https://the-odds-api.com/ (the free tier
has a limited monthly request quota — each league counts as its own
request per run, so keep `SOCCER_LEAGUES` to the leagues you actually
care about if you're on the free plan).

---

## 6. Deploying on GitHub Actions

GitHub Actions' free tier gives every repo 2,000 scheduled-workflow
minutes/month, which is far more than a sub-minute daily run needs, and
(unlike PythonAnywhere's free tier) it has no restriction on scheduled
("cron") jobs at all.

### 6.1 The one thing to understand: persistence

GitHub Actions runners are **ephemeral** — every run starts from a fresh
checkout of your repo and throws the filesystem away afterwards. Since the
whole point of this project is a `history.db` that accumulates predictions
and gets smarter over time, the workflow in
`.github/workflows/daily_value_bets.yml` handles this by **committing
`history.db` back to the repo** as the last step of every run:

```
checkout latest commit (includes yesterday's history.db)
 → run main.py (reads + writes ./history.db)
 → git commit + push the updated history.db
```

So `history.db` genuinely lives in your git history, growing by one commit
a day. This is simple, requires no extra infrastructure/secrets, and gives
you a free audit trail of every day's predictions. (If you'd rather not
have a binary file churn in your commit history, see the note at the
bottom of this section for an alternative using a persistent volume /
external DB service instead.)

### 6.2 Setup steps

1. **Push this project to a GitHub repository** (private is recommended,
   since `history.db` will contain your prediction history):
   ```bash
   cd value_bet_finder
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/value-bet-finder.git
   git push -u origin main
   ```

2. **Add your secrets.** In the GitHub repo, go to
   **Settings → Secrets and variables → Actions → New repository secret**
   and add each of these:

   | Secret name | Value |
   |---|---|
   | `ODDS_API_KEY` | Your The Odds API key |
   | `EMAIL_USER` | Your Gmail address (the sender) |
   | `EMAIL_PASSWORD` | Your Gmail **App Password** (16 chars — see §5) |
   | `EMAIL_TO` | Recipient address(es), comma-separated (optional — defaults to `EMAIL_USER`) |

   These map directly to the `env:` block in the workflow file — nothing
   else needs editing for a standard setup.

3. **Confirm the schedule.** The workflow is already set to run daily at
   **08:45 UTC** (`cron: "45 8 * * *"` in the workflow file), which leaves
   over an hour of buffer before the 10:00 UTC deadline even accounting
   for GitHub's scheduler occasionally being a few minutes late (this is
   normal and documented GitHub Actions behaviour — cron-triggered runs
   are best-effort, not guaranteed to the minute). Adjust the cron
   expression if you want a different time; it's always UTC.

4. **Enable the workflow.** Scheduled workflows are enabled automatically
   once the YAML file is on your default branch, but GitHub disables
   scheduled runs on repos with **no activity for 60 days** — as long as
   you (or the bot's own daily commits) touch the repo periodically this
   won't be an issue, since the workflow itself commits daily.

5. **Test it manually first.** Go to the **Actions** tab → **Daily Value
   Bet Finder** → **Run workflow**. Tick the `dry_run` input to do a full
   run (fetches data, generates picks, writes to `history.db`) without
   actually sending an email — good for confirming your secrets are
   correct before the first real send.

6. **Verify.**
   - Check the run's logs directly in the **Actions** tab.
   - The full `finder.log` is also uploaded as a downloadable build
     artifact on every run (kept for 30 days), under the run's summary
     page.
   - Check your inbox for the email (if you didn't use `dry_run`).
   - `history.db` should show a new commit from `value-bet-finder-bot`
     after each run — pull the repo locally and inspect it with
     `sqlite3 history.db "select * from predictions order by id desc limit 5;"`.

That's it — no server to keep alive, no manual restarts. GitHub re-runs the
workflow daily, and each run picks up exactly where the last one's
`history.db` left off.

### 6.3 Alternative to committing the DB (optional)

If you'd prefer not to accumulate daily commits of a binary SQLite file,
two common alternatives are:
- **`actions/cache`**: cache `history.db` between runs instead of
  committing it. Simpler diff-wise, but GitHub evicts caches that go
  unused for 7 days and caps total cache size per repo, so it's less
  durable for a long-running learning system than a committed file.
- **An external database**: point `DATABASE_PATH`-equivalent logic at a
  small hosted Postgres/SQLite-compatible service (e.g. a free-tier
  Turso/Supabase instance) instead of a local file, and adapt
  `database.py`'s connection layer accordingly. More setup, but avoids
  git churn entirely and is the more "production" answer if you expect to
  scale this up significantly.

The committed-`history.db` approach in the provided workflow is the
simplest option that satisfies the "runs unattended and keeps learning"
requirement with zero extra infrastructure, so it's the default here.

---

## 7. Configuration reference

All configuration is read from environment variables (see `.env.example`
for the full list with defaults). The most important ones:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ODDS_API_KEY` | ✅ | — | The Odds API key |
| `EMAIL_USER` | ✅ | — | Gmail address that sends the mail |
| `EMAIL_PASSWORD` | ✅ | — | Gmail App Password |
| `EMAIL_TO` | | `EMAIL_USER` | Comma-separated recipient list |
| `DATABASE_PATH` | | `./history.db` | SQLite file location |
| `LOG_PATH` | | `./finder.log` | Log file location |
| `SOCCER_LEAGUES` | | 6 major leagues | Comma-separated Odds API sport keys |
| `ODDS_REGIONS` | | `eu,uk` | Bookmaker regions queried |
| `MIN_VALUE_THRESHOLD` | | `0.05` | Minimum edge (5%) to qualify as a pick |
| `MAX_VALUE_THRESHOLD` | | `0.60` | Sanity ceiling — bigger "edges" are treated as bad data |
| `MAX_PICKS_PER_DAY` | | `3` | How many picks to try to deliver |
| `MAX_PICKS_PER_MARKET` | | `2` | Diversity cap per market |
| `DRY_RUN` | | `false` | Run everything but skip actually sending email |

---

## 8. Troubleshooting: "Matches analysed: 0" every day

If the daily email consistently reports **0 matches analysed** (not "0
picks found" — those are different problems), the issue is happening at
the API-fetch step, before any modelling or value filtering runs at all.
Do this first:

```bash
ODDS_API_KEY=your_real_key python diagnose_api.py
```

(Works locally, or via `workflow_dispatch` if you add it as a manual
step — it only needs `requests` and your API key, no other setup.) It
checks, in order:

1. **Is the key valid at all?** A non-200 from `/sports` here means every
   `/odds` call will also fail — the diagnostic prints the API's own error
   message (e.g. "Invalid API key").
2. **Is your quota exhausted?** Prints `used`/`remaining` from the
   response headers directly.
3. **Do your configured leagues return any events, and do any fall on
   today's UTC date?** Distinguishes "API is fine, just nothing today"
   from "API returned nothing at all."
4. **Are `btts`/`team_totals` the problem?** These are billed as
   "additional markets" by The Odds API and are **gated to paid plans on
   some tiers** — a free-tier key requesting them can get the entire
   request rejected, not just those two markets silently dropped. This is
   the single most common cause of "always 0 matches, since day one,"
   because it fails identically on every league, every single day,
   regardless of the real fixture list.

**The fix that's already applied in this codebase:** `config.py`'s
default `ODDS_MARKETS` is now `h2h,totals` (available on every Odds API
plan including free) instead of the earlier `h2h,totals,btts,team_totals`
default. If you want BTTS / Team Over 1.5 picks back, run the diagnostic
above first to confirm your plan actually supports those markets, then set
`ODDS_MARKETS=h2h,totals,btts,team_totals` as a repo secret.

**Also fixed:** `api_client.py` previously retried *permanent* errors
(bad key, unsupported market, etc.) four times with backoff before giving
up, and the final failure message never included the API's actual error
body — so the root cause was invisible in `finder.log`. It now fails
immediately on non-transient errors (4xx) and logs the real response body,
while still retrying genuinely transient ones (429 rate limit, 5xx).

If `diagnose_api.py` shows healthy 200s with real events but your daily
email still says 0 matches, check `finder.log` (or the workflow's
uploaded log artifact) for one of these two distinct messages, which
`fetch_todays_matches()` now logs explicitly:

- `"All N leagues responded successfully but returned 0 events in total"`
  → the API/plan is fine; there's genuinely nothing listed for your
  configured `SOCCER_LEAGUES` right now (e.g. off-season, delisted sport
  key). Check `SOCCER_LEAGUES` against the current list at
  `GET /v4/sports?apiKey=...`.
- `"Leagues returned N raw events in total, but none fall within today's
  UTC window"` → matches exist but aren't scheduled today (UTC). This is
  expected on genuinely quiet match days for your configured leagues —
  not every league plays every day.

## 9. Known limitations / things to be aware of

- **Cold start**: with no historical results yet, the model has nothing
  to differentiate teams with, so early runs will rarely (if ever) find a
  genuine edge — this is by design, not a bug. Ratings improve as more
  matches settle.
- **The Odds API doesn't provide historical goal stats** — team
  attack/defense ratings are learned purely from results *this program
  itself* observes via the `/scores` endpoint going forward. If you want
  a stronger cold start, you can seed `team_ratings` directly in the
  database from an external source (e.g. a season's worth of
  Football-Data.org results) before the first run.
- **BTTS / Team Over 1.5 markets** are not offered by every bookmaker on
  every league; `api_client.py` and `value_calculator.py` degrade
  gracefully (simply produce fewer candidates) when a market is missing
  rather than erroring out.
- **Free-tier API quota**: each configured league costs one request per
  odds fetch and one per scores fetch, per run. Trim `SOCCER_LEAGUES` if
  you're on a constrained plan.
- The no-vig calculation uses simple proportional (multiplicative)
  margin removal rather than the more elaborate Shin method; this is a
  reasonable, well-understood approximation and is easy to swap out in
  `value_calculator.remove_vig()` if you want to experiment.

---

## 10. Extending the system

- **New probability model**: implement a class with `estimate(home_team,
  away_team, league) -> MatchProbabilities` and `confidence(...) -> float`
  (see `probability_models.PoissonEloModel`), then pass an instance of it
  into `value_calculator.build_candidates_for_all_matches(matches, db,
  model=YourModel(db))`.
- **New odds provider**: implement `fetch_todays_matches()` and
  `fetch_recent_scores()` with the same return shapes documented at the
  top of `api_client.py`.
- **Telegram notifications**: `email_sender.compose_email()` already
  returns a plain-text body you can repurpose for a Telegram bot message
  — add a small `telegram_sender.py` following the same retry pattern as
  `email_sender.send_email()`.

---

## 11. Running the tests

```bash
python -m unittest discover -s tests -v
```

The test suite covers the pure-function core (scoreline grid math, no-vig
removal, pick diversity/threshold logic, and basic database round-trips)
without touching the network or sending real email.
