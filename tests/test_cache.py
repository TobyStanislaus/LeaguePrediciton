"""Tests for the SQLite response cache and collection tables."""

from __future__ import annotations

import time

import pytest

from data.cache import Cache


def test_response_round_trip(cache: Cache):
    cache.put_response("k1", "match-v5.match", "https://x/y", {"a": 1, "b": [1, 2]})
    assert cache.get_response("k1") == {"a": 1, "b": [1, 2]}


def test_miss_returns_none(cache: Cache):
    assert cache.get_response("nope") is None


def test_max_age_expires_a_stale_entry(cache: Cache, monkeypatch):
    cache.put_response("k", "league-v4.entries", "https://x", [1])
    assert cache.get_response("k", max_age=60) == [1]

    # Jump forward past the TTL.
    real_time = time.time()
    monkeypatch.setattr("data.cache.time.time", lambda: real_time + 3600)
    assert cache.get_response("k", max_age=60) is None
    # No TTL means immutable: still a hit.
    assert cache.get_response("k", max_age=None) == [1]


def test_put_response_overwrites(cache: Cache):
    cache.put_response("k", "e", "u", {"v": 1})
    cache.put_response("k", "e", "u", {"v": 2})
    assert cache.get_response("k") == {"v": 2}
    assert cache.cached_endpoint_counts() == {"e": 1}


def test_league_snapshot_and_entries(cache: Cache):
    snap = cache.start_league_snapshot("euw1", "RANKED_SOLO_5x5", note="test")
    written = cache.add_league_entries(
        snap,
        [
            {
                "puuid": "p1", "summoner_id": "s1", "tier": "DIAMOND", "rank_division": "II",
                "league_points": 42, "wins": 100, "losses": 90, "hot_streak": True,
                "veteran": False, "fresh_blood": False, "inactive": False,
            },
            {"puuid": "p2", "tier": "DIAMOND", "league_points": 10, "wins": 5, "losses": 5},
        ],
    )
    assert written == 2
    assert sorted(cache.known_puuids(snap)) == ["p1", "p2"]

    row = cache.conn.execute(
        "SELECT * FROM league_entries WHERE puuid = 'p1'"
    ).fetchone()
    assert row["league_points"] == 42
    assert row["hot_streak"] == 1
    assert row["captured_at"] > 0


def test_entries_without_puuid_are_dropped(cache: Cache):
    snap = cache.start_league_snapshot("euw1", "RANKED_SOLO_5x5")
    assert cache.add_league_entries(snap, [{"summoner_id": "s-only", "tier": "GOLD"}]) == 0


def test_captured_at_matches_the_snapshot(cache: Cache):
    snap = cache.start_league_snapshot("euw1", "RANKED_SOLO_5x5")
    cache.add_league_entries(snap, [{"puuid": "p1", "tier": "GOLD"}])
    snap_at = cache.conn.execute(
        "SELECT captured_at FROM league_snapshots WHERE snapshot_id = ?", (snap,)
    ).fetchone()["captured_at"]
    entry_at = cache.conn.execute(
        "SELECT captured_at FROM league_entries WHERE puuid = 'p1'"
    ).fetchone()["captured_at"]
    assert entry_at == snap_at


def test_unknown_snapshot_id_is_rejected(cache: Cache):
    with pytest.raises(ValueError):
        cache.add_league_entries(9999, [{"puuid": "p1"}])


def test_add_match_and_participants(cache: Cache):
    match = {
        "match_id": "EUW1_1", "platform_id": "EUW1", "region": "europe", "queue_id": 420,
        "game_version": "14.18.1", "game_start_ts": 1_726_000_060.0, "game_duration": 1940,
        "winning_team": 100,
    }
    participants = [
        {"puuid": f"p{i}", "team_id": 100 if i < 5 else 200, "champion_id": i, "team_position": "TOP"}
        for i in range(10)
    ]
    cache.add_match(match, participants)

    assert cache.has_match("EUW1_1")
    assert cache.match_count() == 1
    stored = cache.conn.execute("SELECT COUNT(*) n FROM match_participants").fetchone()["n"]
    assert stored == 10


def test_add_match_is_idempotent(cache: Cache):
    match = {"match_id": "EUW1_1", "winning_team": 100, "game_start_ts": 1.0}
    parts = [{"puuid": "p1", "team_id": 100}]
    cache.add_match(match, parts)
    cache.add_match(match, parts)
    assert cache.match_count() == 1


def test_match_participants_table_has_no_performance_columns(cache: Cache):
    """Schema-level leakage guard: outcome-correlated stats must not be storable."""
    columns = {
        r["name"]
        for r in cache.conn.execute("PRAGMA table_info(match_participants)").fetchall()
    }
    banned = {
        "kills", "deaths", "assists", "gold_earned", "damage", "vision_score",
        "cs", "champ_level", "win",
    }
    assert columns & banned == set()


def test_progress_markers(cache: Cache):
    assert not cache.is_done("job:1")
    cache.mark_done("job:1")
    assert cache.is_done("job:1")


def test_summary_counts(cache: Cache):
    cache.put_response("k", "e", "u", {})
    snap = cache.start_league_snapshot("euw1", "RANKED_SOLO_5x5")
    cache.add_league_entries(snap, [{"puuid": "p1"}])
    cache.add_match({"match_id": "m1", "winning_team": 100}, [])
    summary = cache.summary()
    assert summary["cached_responses"] == 1
    assert summary["league_entries"] == 1
    assert summary["matches"] == 1
