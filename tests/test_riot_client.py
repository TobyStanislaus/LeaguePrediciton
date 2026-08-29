"""Tests for the Riot client: rate limiting, retries, error surfacing, cache-first reads.

No test touches the network -- every request goes through a fake session.
"""

from __future__ import annotations

import time

import pytest

from data.riot_client import (
    PLATFORM_TO_REGION,
    RateLimiter,
    RiotAPIError,
    RiotAuthError,
    RiotClient,
    RiotNotFoundError,
    RiotRateLimitError,
    RiotServerError,
    iter_apex_entries,
    load_api_key,
    normalise_league_entry,
    normalise_match,
)
from tests.conftest import FakeResponse, FakeSession


def make_client(cache, session, **kwargs) -> RiotClient:
    return RiotClient(
        platform=kwargs.pop("platform", "euw1"),
        cache=cache,
        api_key="test-key-not-real",
        session=session,
        max_retries=kwargs.pop("max_retries", 3),
        limits=kwargs.pop("limits", ((100, 1.0),)),
        **kwargs,
    )


# --------------------------------------------------------------------------
# Key handling
# --------------------------------------------------------------------------


def test_missing_key_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr("data.riot_client.load_dotenv", lambda **kw: None)
    monkeypatch.delenv("RIOT_API_KEY", raising=False)
    with pytest.raises(RiotAuthError) as excinfo:
        load_api_key()
    message = str(excinfo.value)
    assert "RIOT_API_KEY is not set" in message
    assert ".env.example" in message


def test_blank_key_is_treated_as_missing(monkeypatch):
    monkeypatch.setattr("data.riot_client.load_dotenv", lambda **kw: None)
    monkeypatch.setenv("RIOT_API_KEY", "   ")
    with pytest.raises(RiotAuthError):
        load_api_key()


def test_key_travels_in_the_header_only(cache):
    session = FakeSession()
    client = make_client(cache, session)
    assert session.headers["X-Riot-Token"] == "test-key-not-real"


def test_repr_does_not_leak_the_key(cache):
    client = make_client(cache, FakeSession())
    assert "test-key-not-real" not in repr(client)
    assert "euw1" in repr(client)


def test_unknown_platform_is_rejected(cache):
    with pytest.raises(ValueError, match="unknown platform"):
        RiotClient(platform="mars1", cache=cache, api_key="k", session=FakeSession())


def test_platform_maps_to_regional_route(cache):
    client = make_client(cache, FakeSession(), platform="kr")
    assert client.region == "asia"
    assert PLATFORM_TO_REGION["na1"] == "americas"


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


def test_limiter_allows_a_burst_up_to_the_limit():
    limiter = RateLimiter([(3, 10.0)])
    started = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    assert time.monotonic() - started < 0.05


def test_limiter_blocks_past_the_limit():
    limiter = RateLimiter([(2, 0.15)])
    started = time.monotonic()
    for _ in range(4):
        limiter.acquire()
    elapsed = time.monotonic() - started
    assert elapsed >= 0.15
    assert limiter.total_wait > 0


def test_limiter_enforces_the_tightest_of_several_windows():
    limiter = RateLimiter([(100, 1.0), (2, 0.15)])
    started = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    assert time.monotonic() - started >= 0.15


def test_limiter_requires_at_least_one_limit():
    with pytest.raises(ValueError):
        RateLimiter([])


# --------------------------------------------------------------------------
# Cache-first behaviour
# --------------------------------------------------------------------------


def test_second_call_is_served_from_cache(cache, match_payload):
    session = FakeSession([FakeResponse(200, match_payload)])
    client = make_client(cache, session)

    first = client.get_match("EUW1_1234567890")
    second = client.get_match("EUW1_1234567890")

    assert first == second
    assert len(session.calls) == 1, "cache did not prevent the second network call"
    assert client.stats.requests_made == 1
    assert client.stats.cache_hits == 1


def test_a_fresh_client_reuses_the_cache_on_disk(cache, match_payload):
    first_session = FakeSession([FakeResponse(200, match_payload)])
    make_client(cache, first_session).get_match("EUW1_1234567890")

    second_session = FakeSession()  # any request would raise
    client = make_client(cache, second_session)
    assert client.get_match("EUW1_1234567890")["metadata"]["matchId"] == "EUW1_1234567890"
    assert second_session.calls == []


def test_cache_key_distinguishes_query_parameters(cache):
    session = FakeSession([FakeResponse(200, ["a"]), FakeResponse(200, ["b"])])
    client = make_client(cache, session)
    assert client.get_match_ids("puuid-1", count=5) == ["a"]
    assert client.get_match_ids("puuid-1", count=10) == ["b"]
    assert len(session.calls) == 2


def test_none_valued_params_are_dropped(cache):
    session = FakeSession([FakeResponse(200, [])])
    client = make_client(cache, session)
    client.get_match_ids("puuid-1", queue=None, start_time=None)
    _, params = session.calls[0]
    assert "queue" not in params
    assert "startTime" not in params
    assert params["count"] == 20


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_are_explicit_and_not_retried(cache, status, no_sleep):
    session = FakeSession([FakeResponse(status, None)])
    client = make_client(cache, session)

    with pytest.raises(RiotAuthError) as excinfo:
        client.get_match("EUW1_1")

    message = str(excinfo.value)
    assert "expired" in message
    assert "RIOT_API_KEY" in message
    assert len(session.calls) == 1, "auth failures must not be retried"
    assert client.stats.errors[status] == 1


def test_not_found_raises_its_own_type(cache, no_sleep):
    session = FakeSession([FakeResponse(404, None)])
    client = make_client(cache, session)
    with pytest.raises(RiotNotFoundError):
        client.get_match("EUW1_missing")
    assert client.stats.not_found == 1


def test_429_is_retried_and_honours_retry_after(cache, match_payload, no_sleep):
    session = FakeSession([
        FakeResponse(429, None, headers={"Retry-After": "3", "X-Rate-Limit-Type": "application"}),
        FakeResponse(200, match_payload),
    ])
    client = make_client(cache, session)

    result = client.get_match("EUW1_1234567890")

    assert result["metadata"]["matchId"] == "EUW1_1234567890"
    assert client.stats.rate_limited == 1
    assert client.stats.retries == 1
    assert no_sleep and no_sleep[0] >= 3.0, "Retry-After was not respected"


def test_429_without_retry_after_uses_a_default(cache, match_payload, no_sleep):
    session = FakeSession([FakeResponse(429, None), FakeResponse(200, match_payload)])
    client = make_client(cache, session)
    client.get_match("EUW1_1234567890")
    assert no_sleep[0] == 5.0


def test_persistent_429_eventually_raises(cache, no_sleep):
    session = FakeSession([FakeResponse(429, None) for _ in range(6)])
    client = make_client(cache, session, max_retries=2)
    with pytest.raises(RiotRateLimitError):
        client.get_match("EUW1_1")
    assert len(session.calls) == 3  # initial attempt + 2 retries


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_server_errors_are_retried_then_succeed(cache, match_payload, status, no_sleep):
    session = FakeSession([FakeResponse(status, None), FakeResponse(200, match_payload)])
    client = make_client(cache, session)
    assert client.get_match("EUW1_1234567890")["info"]["queueId"] == 420
    assert client.stats.retries == 1


def test_persistent_server_error_raises(cache, no_sleep):
    session = FakeSession([FakeResponse(503, None) for _ in range(6)])
    client = make_client(cache, session, max_retries=2)
    with pytest.raises(RiotServerError):
        client.get_match("EUW1_1")


def test_backoff_grows_between_attempts(cache, no_sleep):
    session = FakeSession([FakeResponse(500, None) for _ in range(4)])
    client = make_client(cache, session, max_retries=3)
    with pytest.raises(RiotServerError):
        client.get_match("EUW1_1")
    assert len(no_sleep) == 3
    assert no_sleep[0] < no_sleep[-1], "backoff did not increase"


def test_unexpected_4xx_raises_the_base_error(cache, no_sleep):
    session = FakeSession([FakeResponse(415, None, text="Unsupported Media Type")])
    client = make_client(cache, session)
    with pytest.raises(RiotAPIError) as excinfo:
        client.get_match("EUW1_1")
    assert excinfo.value.status == 415


def test_network_exception_is_retried(cache, match_payload, no_sleep):
    import requests

    class FlakySession(FakeSession):
        def __init__(self):
            super().__init__([FakeResponse(200, match_payload)])
            self.failed = False

        def get(self, url, params=None, timeout=None):
            if not self.failed:
                self.failed = True
                raise requests.ConnectionError("connection reset")
            return super().get(url, params, timeout)

    client = make_client(cache, FlakySession())
    assert client.get_match("EUW1_1234567890")["info"]["queueId"] == 420
    assert client.stats.retries == 1


def test_failed_responses_are_not_cached(cache, no_sleep):
    session = FakeSession([FakeResponse(404, None), FakeResponse(404, None)])
    client = make_client(cache, session)
    for _ in range(2):
        with pytest.raises(RiotNotFoundError):
            client.get_match("EUW1_missing")
    assert len(session.calls) == 2, "an error response was cached"


# --------------------------------------------------------------------------
# Endpoint routing
# --------------------------------------------------------------------------


def test_match_endpoints_use_the_regional_host(cache, match_payload):
    session = FakeSession([FakeResponse(200, match_payload)])
    client = make_client(cache, session)
    client.get_match("EUW1_1234567890")
    url, _ = session.calls[0]
    assert url.startswith("https://europe.api.riotgames.com/lol/match/v5/matches/")


def test_league_endpoints_use_the_platform_host(cache):
    session = FakeSession([FakeResponse(200, {"tier": "CHALLENGER", "entries": []})])
    client = make_client(cache, session)
    client.get_apex_league("CHALLENGER")
    url, _ = session.calls[0]
    assert url.startswith("https://euw1.api.riotgames.com/lol/league/v4/challengerleagues/")


def test_apex_tier_is_validated(cache):
    client = make_client(cache, FakeSession())
    with pytest.raises(ValueError, match="not an apex tier"):
        client.get_apex_league("GOLD")


def test_match_id_count_is_capped_at_100(cache):
    session = FakeSession([FakeResponse(200, [])])
    client = make_client(cache, session)
    client.get_match_ids("p1", count=500)
    _, params = session.calls[0]
    assert params["count"] == 100


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def test_normalise_match_extracts_the_winner(match_payload):
    match, participants = normalise_match(match_payload)
    assert match["match_id"] == "EUW1_1234567890"
    assert match["winning_team"] == 100
    assert match["queue_id"] == 420
    assert match["game_start_ts"] == pytest.approx(1_726_000_060.0)
    assert len(participants) == 10


def test_normalise_match_drops_every_in_game_statistic(match_payload):
    """The core leakage guard for the collection layer."""
    _, participants = normalise_match(match_payload)
    allowed = {"puuid", "team_id", "champion_id", "team_position"}
    for participant in participants:
        assert set(participant) == allowed, f"unexpected keys: {set(participant) - allowed}"


def test_normalise_match_keeps_no_per_player_win_flag(match_payload):
    """Per-participant ``win`` would hand the label straight to the features."""
    _, participants = normalise_match(match_payload)
    assert all("win" not in p for p in participants)


def test_normalise_match_teams_are_five_a_side(match_payload):
    _, participants = normalise_match(match_payload)
    blue = [p for p in participants if p["team_id"] == 100]
    red = [p for p in participants if p["team_id"] == 200]
    assert len(blue) == len(red) == 5


def test_normalise_match_without_a_match_id_is_rejected():
    with pytest.raises(ValueError, match="matchId"):
        normalise_match({"metadata": {}, "info": {}})


def test_normalise_match_falls_back_to_game_creation():
    payload = {
        "metadata": {"matchId": "EUW1_1"},
        "info": {"gameCreation": 1_700_000_000_000, "teams": [], "participants": []},
    }
    match, _ = normalise_match(payload)
    assert match["game_start_ts"] == pytest.approx(1_700_000_000.0)


def test_normalise_league_entry_maps_riot_field_names():
    entry = normalise_league_entry({
        "puuid": "p1", "summonerId": "s1", "tier": "diamond", "rank": "II",
        "leaguePoints": 42, "wins": 120, "losses": 100, "hotStreak": True,
        "veteran": False, "freshBlood": True, "inactive": False,
    })
    assert entry["tier"] == "DIAMOND"
    assert entry["rank_division"] == "II"
    assert entry["league_points"] == 42
    assert entry["hot_streak"] is True
    assert entry["fresh_blood"] is True


def test_apex_entries_inherit_the_league_tier():
    """LeagueItemDTO carries no tier of its own -- it comes from the league."""
    league = {
        "tier": "CHALLENGER",
        "entries": [{"puuid": "p1", "leaguePoints": 1200, "wins": 300, "losses": 250}],
    }
    entries = list(iter_apex_entries(league))
    assert entries[0]["tier"] == "CHALLENGER"
    assert entries[0]["league_points"] == 1200
