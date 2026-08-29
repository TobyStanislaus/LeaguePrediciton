"""Tests for the collection driver, using a stub client (no network)."""

from __future__ import annotations

from typing import Any

import pytest

from data.cache import Cache
from data.collect import collect_matches, forward_only_start_time, seed_ladder
from data.riot_client import RiotNotFoundError


class StubClient:
    """Stands in for RiotClient, recording calls and replaying canned payloads."""

    def __init__(self, ladder_size: int = 10, with_puuid: bool = True) -> None:
        self.platform = "euw1"
        self.region = "europe"
        self.ladder_size = ladder_size
        self.with_puuid = with_puuid
        self.summoner_lookups = 0
        self.match_id_calls: list[tuple[str, int | None]] = []
        self.match_calls: list[str] = []

        class _Stats:
            requests_made = 0
            cache_hits = 0
            retries = 0
            rate_limited = 0

        self.stats = _Stats()

    def get_apex_league(self, tier: str, queue: str = "RANKED_SOLO_5x5") -> dict[str, Any]:
        entries = []
        for i in range(self.ladder_size):
            entry: dict[str, Any] = {
                "leaguePoints": 500 + i,
                "wins": 100 + i,
                "losses": 90,
                "hotStreak": i % 2 == 0,
                "rank": "I",
            }
            if self.with_puuid:
                entry["puuid"] = f"puuid-{i}"
            else:
                entry["summonerId"] = f"summoner-{i}"
            entries.append(entry)
        return {"tier": tier, "queue": queue, "entries": entries}

    def get_summoner_by_id(self, summoner_id: str) -> dict[str, Any]:
        self.summoner_lookups += 1
        return {"puuid": summoner_id.replace("summoner", "puuid")}

    def get_match_ids(self, puuid, *, count=20, queue=None, start_time=None, **kw):
        self.match_id_calls.append((puuid, start_time))
        return [f"EUW1_{puuid}_{i}" for i in range(2)]

    def get_match(self, match_id: str) -> dict[str, Any]:
        self.match_calls.append(match_id)
        participants = [
            {"puuid": f"{match_id}-p{i}", "teamId": 100 if i < 5 else 200, "championId": i,
             "teamPosition": "TOP", "kills": 5, "goldEarned": 9000}
            for i in range(10)
        ]
        return {
            "metadata": {"matchId": match_id},
            "info": {
                "platformId": "EUW1", "queueId": 420, "gameVersion": "14.18.1",
                "gameStartTimestamp": 1_726_000_000_000, "gameDuration": 1800,
                "participants": participants,
                "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
            },
        }


def test_seed_stores_the_whole_ladder_not_just_the_seed_subset(cache: Cache):
    """The extra entries came free in the same call and save phase-3 lookups."""
    client = StubClient(ladder_size=50)
    snapshot_id, puuids = seed_ladder(client, cache, tiers=["CHALLENGER"], max_summoners=3)

    assert len(puuids) == 3, "only the capped subset should seed match collection"
    stored = cache.conn.execute(
        "SELECT COUNT(*) n FROM league_entries WHERE snapshot_id = ?", (snapshot_id,)
    ).fetchone()["n"]
    assert stored == 50, "the full ladder should be recorded in the snapshot"


def test_seed_without_a_cap_uses_every_player(cache: Cache):
    client = StubClient(ladder_size=7)
    _, puuids = seed_ladder(client, cache, tiers=["CHALLENGER"], max_summoners=None)
    assert len(puuids) == 7


def test_seed_sampling_spreads_across_the_ladder(cache: Cache):
    """Slicing the first N would draw every seed from one tier/division page."""
    client = StubClient(ladder_size=200)
    _, puuids = seed_ladder(client, cache, tiers=["CHALLENGER"], max_summoners=20)

    positions = sorted(int(p.split("-")[1]) for p in puuids)
    assert len(positions) == 20
    assert max(positions) > 100, "sample is confined to the head of the ladder"


def test_seed_sampling_is_reproducible(cache: Cache, tmp_path):
    client = StubClient(ladder_size=200)
    _, first = seed_ladder(client, cache, tiers=["CHALLENGER"], max_summoners=15, random_state=7)
    with Cache(tmp_path / "second.sqlite") as other:
        _, second = seed_ladder(
            StubClient(ladder_size=200), other, tiers=["CHALLENGER"],
            max_summoners=15, random_state=7,
        )
    assert first == second


def test_seed_only_resolves_puuids_for_the_seed_subset(cache: Cache):
    """Resolving the whole ladder could cost hundreds of extra API calls."""
    client = StubClient(ladder_size=100, with_puuid=False)
    _, puuids = seed_ladder(client, cache, tiers=["CHALLENGER"], max_summoners=5)
    assert len(puuids) == 5
    assert client.summoner_lookups == 5, "resolved more summoners than needed"


def test_seed_records_a_timestamp(cache: Cache):
    client = StubClient(ladder_size=3)
    snapshot_id, _ = seed_ladder(client, cache, tiers=["CHALLENGER"])
    row = cache.conn.execute(
        "SELECT captured_at FROM league_snapshots WHERE snapshot_id = ?", (snapshot_id,)
    ).fetchone()
    assert row["captured_at"] > 0


def test_collect_matches_stores_matches_and_returns_participants(cache: Cache):
    client = StubClient()
    seen = collect_matches(client, cache, ["puuid-0"], matches_per_summoner=2)

    assert cache.match_count() == 2
    assert len(seen) == 20  # 2 matches x 10 players
    assert len(client.match_calls) == 2


def test_collect_matches_skips_matches_already_stored(cache: Cache):
    client = StubClient()
    collect_matches(client, cache, ["puuid-0"], matches_per_summoner=2)
    calls_after_first = len(client.match_calls)

    collect_matches(client, cache, ["puuid-0"], matches_per_summoner=2)
    assert len(client.match_calls) == calls_after_first, "re-fetched an already-stored match"
    assert cache.match_count() == 2


def test_collect_matches_still_returns_participants_for_cached_matches(cache: Cache):
    """A resumed run must still feed phase 3, not just a fresh one."""
    client = StubClient()
    collect_matches(client, cache, ["puuid-0"], matches_per_summoner=2)
    seen = collect_matches(client, cache, ["puuid-0"], matches_per_summoner=2)
    assert len(seen) == 20


def test_collect_matches_drops_in_game_statistics(cache: Cache):
    client = StubClient()
    collect_matches(client, cache, ["puuid-0"], matches_per_summoner=2)
    columns = {
        r["name"] for r in cache.conn.execute("PRAGMA table_info(match_participants)").fetchall()
    }
    assert "kills" not in columns and "gold_earned" not in columns


def test_forward_only_passes_start_time_through(cache: Cache):
    client = StubClient()
    collect_matches(client, cache, ["puuid-0"], start_time=1_726_000_000)
    assert client.match_id_calls[0][1] == 1_726_000_000


def _snapshot_at(cache: Cache, captured_at: float) -> int:
    snapshot_id = cache.start_league_snapshot("euw1", "RANKED_SOLO_5x5")
    cache.conn.execute(
        "UPDATE league_snapshots SET captured_at = ? WHERE snapshot_id = ?",
        (captured_at, snapshot_id),
    )
    cache.conn.commit()
    return snapshot_id


def test_forward_only_window_opens_at_the_previous_snapshot(cache: Cache):
    """Regression: anchoring on the new snapshot asks for matches after 'now'.

    That returned zero matches on every run, forever, rather than the matches
    played since the last run.
    """
    _snapshot_at(cache, 1_000.0)
    latest = _snapshot_at(cache, 5_000.0)
    assert forward_only_start_time(cache, latest) == 1_000.0


def test_forward_only_uses_the_most_recent_earlier_snapshot(cache: Cache):
    _snapshot_at(cache, 1_000.0)
    _snapshot_at(cache, 3_000.0)
    latest = _snapshot_at(cache, 5_000.0)
    assert forward_only_start_time(cache, latest) == 3_000.0


def test_forward_only_has_no_window_on_the_very_first_run(cache: Cache):
    first = _snapshot_at(cache, 1_000.0)
    assert forward_only_start_time(cache, first) is None


def test_forward_only_ignores_snapshots_taken_later(cache: Cache):
    """A snapshot written after this one is not a valid window opener."""
    target = _snapshot_at(cache, 2_000.0)
    _snapshot_at(cache, 9_000.0)
    assert forward_only_start_time(cache, target) is None


def test_missing_match_list_is_skipped_not_fatal(cache: Cache):
    class Missing(StubClient):
        def get_match_ids(self, puuid, **kw):
            raise RiotNotFoundError("no such player")

    seen = collect_matches(Missing(), cache, ["ghost"], matches_per_summoner=2)
    assert seen == set()
    assert cache.match_count() == 0
