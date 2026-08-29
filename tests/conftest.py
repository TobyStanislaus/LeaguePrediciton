"""Shared fixtures: a temp cache and a fake HTTP session (no network in tests)."""

from __future__ import annotations

from typing import Any

import pytest

from data.cache import Cache


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(
        self,
        status_code: int = 200,
        payload: Any = None,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self) -> Any:
        return self._payload


class FakeSession:
    """Replays a queued list of responses and records the calls made."""

    def __init__(self, responses: list[FakeResponse] | None = None) -> None:
        self.headers: dict[str, str] = {}
        self.responses = list(responses or [])
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self.closed = False

    def queue(self, *responses: FakeResponse) -> "FakeSession":
        self.responses.extend(responses)
        return self

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: float | None = None):
        self.calls.append((url, params))
        if not self.responses:
            raise AssertionError(f"unexpected extra request to {url}")
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def cache(tmp_path) -> Cache:
    store = Cache(tmp_path / "test.sqlite")
    yield store
    store.close()


@pytest.fixture
def no_sleep(monkeypatch):
    """Make retry/backoff sleeps instant, and record how long was asked for."""
    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("data.riot_client.time.sleep", fake_sleep)
    return slept


@pytest.fixture
def match_payload() -> dict[str, Any]:
    """A MATCH-V5 payload including the in-game stats we must NOT keep."""
    def participant(puuid: str, team_id: int, won: bool, champ: int, pos: str) -> dict[str, Any]:
        return {
            "puuid": puuid,
            "teamId": team_id,
            "championId": champ,
            "teamPosition": pos,
            # Everything below is post-kickoff and must be dropped:
            "win": won,
            "kills": 7,
            "deaths": 3,
            "assists": 11,
            "goldEarned": 12500,
            "totalDamageDealtToChampions": 21000,
            "visionScore": 34,
            "totalMinionsKilled": 180,
            "champLevel": 16,
        }

    positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    participants = [
        participant(f"blue-{i}", 100, True, 100 + i, positions[i]) for i in range(5)
    ] + [participant(f"red-{i}", 200, False, 200 + i, positions[i]) for i in range(5)]

    return {
        "metadata": {
            "matchId": "EUW1_1234567890",
            "participants": [p["puuid"] for p in participants],
        },
        "info": {
            "platformId": "EUW1",
            "queueId": 420,
            "gameVersion": "14.18.1",
            "gameCreation": 1_726_000_000_000,
            "gameStartTimestamp": 1_726_000_060_000,
            "gameEndTimestamp": 1_726_002_000_000,
            "gameDuration": 1940,
            "participants": participants,
            "teams": [
                {"teamId": 100, "win": True, "objectives": {"baron": {"kills": 2}}},
                {"teamId": 200, "win": False, "objectives": {"baron": {"kills": 0}}},
            ],
        },
    }
