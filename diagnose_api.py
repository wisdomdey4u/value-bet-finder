#!/usr/bin/env python3
"""
diagnose_api.py
----------------
Standalone, minimal-dependency script to directly test your ODDS_API_KEY
against The Odds API and print exactly what comes back -- no probability
modelling, no database, no email. Run this FIRST whenever matches aren't
showing up, before digging into anything else.

Usage:
    ODDS_API_KEY=your_key python diagnose_api.py
    # or, if you already have a .env file:
    python diagnose_api.py

What it checks, in order:
  1. Can we reach the API at all with your key? (catches bad/expired keys
     immediately, with the API's own error message)
  2. What's your current quota usage? (catches "quota exhausted")
  3. For each configured league: does /odds return any events at all, and
     if so, do any fall within today's UTC date?
  4. Tries a request with the FULL default market list (h2h,totals,btts,
     team_totals) vs. the SAFE core list (h2h,totals) side-by-side, so you
     can see immediately if btts/team_totals is what's breaking things on
     your plan.
"""

import sys
import os
import datetime as dt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

API_KEY = os.getenv("ODDS_API_KEY", "")
BASE_URL = os.getenv("ODDS_API_BASE_URL", "https://api.the-odds-api.com/v4")
REGIONS = os.getenv("ODDS_REGIONS", "eu,uk")
LEAGUES = os.getenv(
    "SOCCER_LEAGUES",
    "soccer_epl,soccer_spain_la_liga,soccer_italy_serie_a,"
    "soccer_germany_bundesliga,soccer_france_ligue_one,"
    "soccer_uefa_champs_league",
).split(",")


def line():
    print("-" * 72)


def main():
    print("Value Bet Finder - API diagnostic")
    line()

    if not API_KEY:
        print("FAIL: ODDS_API_KEY is not set in the environment. Nothing else "
              "to test -- set it and re-run.")
        sys.exit(1)

    print(f"Using key: {API_KEY[:4]}...{API_KEY[-4:]} (length {len(API_KEY)})")
    print(f"Base URL:  {BASE_URL}")
    print(f"Regions:   {REGIONS}")
    print(f"Leagues:   {LEAGUES}")
    print(f"UTC now:   {dt.datetime.now(dt.timezone.utc).isoformat()}")
    line()

    # --- Step 1: list available sports (cheap, doesn't cost quota on most
    # plans, and immediately validates the key) ---------------------------
    print("Step 1: checking the key is valid via /sports ...")
    try:
        resp = requests.get(f"{BASE_URL}/sports", params={"apiKey": API_KEY}, timeout=15)
    except requests.RequestException as exc:
        print(f"FAIL: could not reach the API at all: {exc}")
        sys.exit(1)

    print(f"  HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"  Body: {resp.text[:500]}")
        print("  -> This is almost certainly why you're getting 0 matches every "
              "day. A non-200 here means every /odds call this run also fails, "
              "before any match/date filtering even happens.")
        sys.exit(1)

    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    print(f"  OK. Quota: used={used}, remaining={remaining}")
    if remaining is not None:
        try:
            if int(remaining) <= 0:
                print("  -> WARNING: 0 requests remaining. This IS your problem: "
                      "every call this run will be rejected until your quota "
                      "resets (monthly on most plans) or you upgrade.")
        except ValueError:
            pass
    line()

    sports = resp.json()
    known_keys = {s.get("key") for s in sports if isinstance(s, dict)}
    for lg in LEAGUES:
        lg = lg.strip()
        if lg and lg not in known_keys:
            print(f"  NOTE: '{lg}' was not found in the /sports list returned "
                  f"for your account/plan. It may be out of season, delisted, "
                  f"or not included on your plan tier.")
    line()

    # --- Step 2: try one real /odds call per league with SAFE markets ----
    print("Step 2: fetching /odds per league with markets='h2h,totals' (safe, "
          "available on all plans) ...")
    total_events_safe = 0
    for lg in LEAGUES:
        lg = lg.strip()
        if not lg:
            continue
        r = requests.get(
            f"{BASE_URL}/sports/{lg}/odds",
            params={"apiKey": API_KEY, "regions": REGIONS, "markets": "h2h,totals",
                    "oddsFormat": "decimal", "dateFormat": "iso"},
            timeout=15,
        )
        if r.status_code != 200:
            print(f"  {lg:32s} HTTP {r.status_code}  {r.text[:200]}")
            continue
        events = r.json()
        n = len(events) if isinstance(events, list) else 0
        total_events_safe += n
        today_utc = dt.datetime.now(dt.timezone.utc).date()
        n_today = 0
        if isinstance(events, list):
            for ev in events:
                ct = ev.get("commence_time", "")
                try:
                    ct_date = dt.datetime.fromisoformat(ct.replace("Z", "+00:00")).astimezone(dt.timezone.utc).date()
                    if ct_date == today_utc:
                        n_today += 1
                except ValueError:
                    pass
        print(f"  {lg:32s} HTTP 200  {n:3d} total upcoming events, {n_today:3d} kicking off today (UTC)")
    line()
    print(f"Total upcoming events across all leagues (safe markets): {total_events_safe}")
    line()

    # --- Step 3: try the FULL market list to see if it's the culprit -----
    print("Step 3: trying markets='h2h,totals,btts,team_totals' on the FIRST "
          "configured league only, to check if additional markets are what's "
          "breaking things on your plan ...")
    first_league = next((lg.strip() for lg in LEAGUES if lg.strip()), None)
    if first_league:
        r = requests.get(
            f"{BASE_URL}/sports/{first_league}/odds",
            params={"apiKey": API_KEY, "regions": REGIONS,
                    "markets": "h2h,totals,btts,team_totals",
                    "oddsFormat": "decimal", "dateFormat": "iso"},
            timeout=15,
        )
        print(f"  {first_league}: HTTP {r.status_code}")
        if r.status_code != 200:
            print(f"  Body: {r.text[:500]}")
            print("  -> If Step 2 succeeded but this failed, 'btts'/'team_totals' "
                  "is confirmed as the problem. Leave ODDS_MARKETS at the new "
                  "default (h2h,totals) or upgrade your Odds API plan.")
        else:
            print("  OK - additional markets are available on your plan.")
    line()

    print("Diagnostic complete.")


if __name__ == "__main__":
    main()
