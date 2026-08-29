"""Local SQLite cache for raw Riot API responses plus normalised collection tables.

Two jobs, deliberately kept separate:

1. ``api_cache`` -- verbatim JSON for every successful API response, keyed by the
   request path (which already contains the match ID / PUUID). The client checks
   this before every network call, so a re-run never burns rate limit on data we
   already hold.

2. Normalised tables (``league_snapshots``, ``league_entries``, ``matches``,
   ``match_participants``) -- the shape feature building actually wants.

Point-in-time correctness
-------------------------
``league_entries`` rows carry ``captured_at``. Rank/LP/wins/losses are a *current*
reading, not the value at the time of some past match, so feature building must
join a snapshot captured **before** the match started. Storing the timestamp is
what makes that join possible; see README.md for why this matters.

Nothing derived from in-game performance (KDA, gold, damage, objectives) is
stored in the normalised tables. The raw JSON in ``api_cache`` still holds it, so
a separate in-game model stays possible later without re-scraping.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

log = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("data/cache/riot.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_cache (
    cache_key  TEXT PRIMARY KEY,
    endpoint   TEXT NOT NULL,
    url        TEXT NOT NULL,
    response   TEXT NOT NULL,
    fetched_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_api_cache_endpoint ON api_cache (endpoint);

CREATE TABLE IF NOT EXISTS league_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at REAL NOT NULL,
    platform    TEXT NOT NULL,
    queue       TEXT NOT NULL,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS league_entries (
    snapshot_id   INTEGER NOT NULL REFERENCES league_snapshots (snapshot_id),
    puuid         TEXT    NOT NULL,
    summoner_id   TEXT,
    tier          TEXT,
    rank_division TEXT,
    league_points INTEGER,
    wins          INTEGER,
    losses        INTEGER,
    hot_streak    INTEGER,
    veteran       INTEGER,
    fresh_blood   INTEGER,
    inactive      INTEGER,
    captured_at   REAL    NOT NULL,
    PRIMARY KEY (snapshot_id, puuid)
);
CREATE INDEX IF NOT EXISTS ix_league_entries_puuid ON league_entries (puuid, captured_at);

CREATE TABLE IF NOT EXISTS matches (
    match_id      TEXT PRIMARY KEY,
    platform_id   TEXT,
    region        TEXT,
    queue_id      INTEGER,
    game_version  TEXT,
    game_start_ts REAL,
    game_duration REAL,
    winning_team  INTEGER,
    fetched_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_matches_start ON matches (game_start_ts);

CREATE TABLE IF NOT EXISTS match_participants (
    match_id      TEXT    NOT NULL REFERENCES matches (match_id),
    puuid         TEXT    NOT NULL,
    team_id       INTEGER NOT NULL,
    champion_id   INTEGER,
    team_position TEXT,
    PRIMARY KEY (match_id, puuid)
);
CREATE INDEX IF NOT EXISTS ix_participants_puuid ON match_participants (puuid);

CREATE TABLE IF NOT EXISTS collection_progress (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at REAL NOT NULL
);
"""


class Cache:
    """SQLite-backed response cache and collection store.

    Usable as a context manager::

        with Cache() as cache:
            cache.put_response(...)
    """

    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        if str(self.path) != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self) -> "Cache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- raw response cache ------------------------------------------------

    def get_response(self, cache_key: str, max_age: float | None = None) -> Any | None:
        """Return the cached payload, or ``None`` on miss / staleness.

        ``max_age`` (seconds) exists because not every endpoint is immutable: a
        finished match never changes, but a ladder page or a player's recent
        match list does. Pass ``None`` for immutable endpoints.
        """
        row = self.conn.execute(
            "SELECT response, fetched_at FROM api_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        if max_age is not None and (time.time() - row["fetched_at"]) > max_age:
            return None
        return json.loads(row["response"])

    def put_response(self, cache_key: str, endpoint: str, url: str, payload: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO api_cache (cache_key, endpoint, url, response, fetched_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (cache_key, endpoint, url, json.dumps(payload, separators=(",", ":")), time.time()),
        )
        self.conn.commit()

    def cached_endpoint_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT endpoint, COUNT(*) AS n FROM api_cache GROUP BY endpoint ORDER BY n DESC"
        ).fetchall()
        return {r["endpoint"]: r["n"] for r in rows}

    # -- league snapshots --------------------------------------------------

    def start_league_snapshot(self, platform: str, queue: str, note: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO league_snapshots (captured_at, platform, queue, note) VALUES (?, ?, ?, ?)",
            (time.time(), platform, queue, note),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_league_entries(self, snapshot_id: int, entries: Iterable[dict[str, Any]]) -> int:
        """Insert normalised ladder rows. Returns the number written."""
        row = self.conn.execute(
            "SELECT captured_at FROM league_snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown snapshot_id {snapshot_id}")
        captured_at = row["captured_at"]

        rows = [
            (
                snapshot_id,
                e["puuid"],
                e.get("summoner_id"),
                e.get("tier"),
                e.get("rank_division"),
                e.get("league_points"),
                e.get("wins"),
                e.get("losses"),
                int(bool(e.get("hot_streak"))),
                int(bool(e.get("veteran"))),
                int(bool(e.get("fresh_blood"))),
                int(bool(e.get("inactive"))),
                captured_at,
            )
            for e in entries
            if e.get("puuid")
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO league_entries (snapshot_id, puuid, summoner_id, tier,"
            " rank_division, league_points, wins, losses, hot_streak, veteran, fresh_blood,"
            " inactive, captured_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def known_puuids(self, snapshot_id: int | None = None) -> list[str]:
        if snapshot_id is None:
            rows = self.conn.execute("SELECT DISTINCT puuid FROM league_entries").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT DISTINCT puuid FROM league_entries WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchall()
        return [r["puuid"] for r in rows]

    # -- matches -----------------------------------------------------------

    def has_match(self, match_id: str) -> bool:
        return (
            self.conn.execute("SELECT 1 FROM matches WHERE match_id = ?", (match_id,)).fetchone()
            is not None
        )

    def add_match(self, match: dict[str, Any], participants: Sequence[dict[str, Any]]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO matches (match_id, platform_id, region, queue_id,"
            " game_version, game_start_ts, game_duration, winning_team, fetched_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                match["match_id"],
                match.get("platform_id"),
                match.get("region"),
                match.get("queue_id"),
                match.get("game_version"),
                match.get("game_start_ts"),
                match.get("game_duration"),
                match.get("winning_team"),
                time.time(),
            ),
        )
        self.conn.executemany(
            "INSERT OR REPLACE INTO match_participants (match_id, puuid, team_id, champion_id,"
            " team_position) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    match["match_id"],
                    p["puuid"],
                    p["team_id"],
                    p.get("champion_id"),
                    p.get("team_position"),
                )
                for p in participants
            ],
        )
        self.conn.commit()

    def match_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM matches").fetchone()["n"])

    def league_entry_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM league_entries").fetchone()["n"])

    # -- resume support ----------------------------------------------------

    def mark_done(self, key: str, value: str = "1") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO collection_progress (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, time.time()),
        )
        self.conn.commit()

    def is_done(self, key: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM collection_progress WHERE key = ?", (key,)
            ).fetchone()
            is not None
        )

    def iter_matches(self) -> Iterator[sqlite3.Row]:
        yield from self.conn.execute("SELECT * FROM matches ORDER BY game_start_ts")

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "cached_responses": sum(self.cached_endpoint_counts().values()),
            "league_entries": self.league_entry_count(),
            "matches": self.match_count(),
        }
