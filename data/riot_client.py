"""Riot Games API client: auth, rate limiting, retries, and cache-first reads.

Design notes
------------
* The API key is read **only** from the ``RIOT_API_KEY`` environment variable
  (loaded from ``.env`` via python-dotenv). It is never written to a log line,
  a repr, an exception message, or the cache -- it travels in the
  ``X-Riot-Token`` header only.
* Every successful response is stored in the SQLite cache and every request
  checks the cache first, so re-runs and resumed jobs cost no rate limit.
* Riot splits routing between *platform* hosts (``euw1``, ``na1``, ...) used by
  LEAGUE-V4 and SUMMONER-V4, and *regional* hosts (``europe``, ``americas``,
  ...) used by MATCH-V5. ``PLATFORM_TO_REGION`` maps between them.
* Rate limits are enforced client-side before the call goes out, rather than
  only reacting to 429s -- reacting alone gets a key temporarily blacklisted.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
from urllib.parse import quote, urlencode

import requests
from dotenv import load_dotenv

from data.cache import Cache

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

PLATFORM_TO_REGION = {
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "na1": "americas",
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "me1": "europe",
    "jp1": "asia",
    "kr": "asia",
    "oc1": "sea",
    "ph2": "sea",
    "sg2": "sea",
    "th2": "sea",
    "tw2": "sea",
    "vn2": "sea",
}

APEX_TIERS = {"CHALLENGER", "GRANDMASTER", "MASTER"}

# Cache freshness. A completed match is immutable, so it never expires; ladder
# state and a player's recent match list obviously do.
TTL_IMMUTABLE = None
TTL_LADDER = 6 * 3600
TTL_MATCH_IDS = 3600
# An in-progress game changes by the second, and a player may finish one and
# start another; cache it only long enough to avoid hammering a retry loop.
TTL_ACTIVE_GAME = 60


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class RiotAPIError(RuntimeError):
    """Base class for Riot API failures."""

    def __init__(self, message: str, status: int | None = None, url: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.url = url


class RiotAuthError(RiotAPIError):
    """401/403 -- missing, invalid, or expired key. Never retried."""


class RiotNotFoundError(RiotAPIError):
    """404 -- the resource does not exist. Callers usually skip and continue."""


class RiotRateLimitError(RiotAPIError):
    """429 that survived every retry."""


class RiotServerError(RiotAPIError):
    """5xx that survived every retry."""


# --------------------------------------------------------------------------
# Key loading
# --------------------------------------------------------------------------


def load_api_key(env_var: str = "RIOT_API_KEY") -> str:
    """Load the Riot API key from the environment, via .env if present.

    Raises a clear error rather than letting an empty key produce a confusing
    403 several hundred requests into a run.
    """
    load_dotenv(override=False)
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RiotAuthError(
            f"{env_var} is not set. Copy .env.example to .env and paste your key from "
            "https://developer.riotgames.com/ . The key is read from the environment only "
            "and must never be hardcoded."
        )
    return key


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window limiter enforcing several (count, seconds) limits at once.

    Riot's personal-key defaults are 20 requests per 1s and 100 per 120s. The
    limiter blocks until every window has room, so bursts are shaped before
    they leave the process.
    """

    def __init__(self, limits: Sequence[tuple[int, float]] = ((20, 1.0), (100, 120.0))) -> None:
        if not limits:
            raise ValueError("at least one limit is required")
        self.limits = list(limits)
        self._hits: list[deque[float]] = [deque() for _ in self.limits]
        self._lock = threading.Lock()
        self.total_wait = 0.0

    def _sleep_needed(self, now: float) -> float:
        wait = 0.0
        for (count, window), hits in zip(self.limits, self._hits):
            while hits and hits[0] <= now - window:
                hits.popleft()
            if len(hits) >= count:
                wait = max(wait, hits[0] + window - now)
        return wait

    def acquire(self) -> None:
        """Block until a request may be sent, then record it."""
        while True:
            with self._lock:
                now = time.monotonic()
                wait = self._sleep_needed(now)
                if wait <= 0:
                    for hits in self._hits:
                        hits.append(now)
                    return
            self.total_wait += wait
            log.debug("rate limit: sleeping %.2fs", wait)
            time.sleep(wait)


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


@dataclass
class ClientStats:
    """Counters for monitoring a long scraping run."""

    requests_made: int = 0
    cache_hits: int = 0
    retries: int = 0
    rate_limited: int = 0
    not_found: int = 0
    errors: dict[int, int] = field(default_factory=dict)

    def record_error(self, status: int) -> None:
        self.errors[status] = self.errors.get(status, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests_made": self.requests_made,
            "cache_hits": self.cache_hits,
            "retries": self.retries,
            "rate_limited": self.rate_limited,
            "not_found": self.not_found,
            "errors": dict(self.errors),
        }


class RiotClient:
    """Cache-first, rate-limited Riot API client."""

    def __init__(
        self,
        platform: str = "euw1",
        cache: Cache | None = None,
        api_key: str | None = None,
        limits: Sequence[tuple[int, float]] = ((20, 1.0), (100, 120.0)),
        max_retries: int = 5,
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        platform = platform.lower()
        if platform not in PLATFORM_TO_REGION:
            raise ValueError(
                f"unknown platform {platform!r}; expected one of {sorted(PLATFORM_TO_REGION)}"
            )
        self.platform = platform
        self.region = PLATFORM_TO_REGION[platform]
        self._api_key = api_key if api_key is not None else load_api_key()
        self.cache = cache if cache is not None else Cache()
        self._owns_cache = cache is None
        self.limiter = RateLimiter(limits)
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"X-Riot-Token": self._api_key})
        self.stats = ClientStats()

    def __repr__(self) -> str:  # never leak the key through a repr
        return f"RiotClient(platform={self.platform!r}, region={self.region!r})"

    def close(self) -> None:
        self.session.close()
        if self._owns_cache:
            self.cache.close()

    def __enter__(self) -> "RiotClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- transport ---------------------------------------------------------

    @staticmethod
    def _cache_key(host: str, path: str, params: dict[str, Any] | None) -> str:
        query = urlencode(sorted((params or {}).items()))
        return f"{host}{path}?{query}" if query else f"{host}{path}"

    def _get(
        self,
        host: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        endpoint: str,
        max_age: float | None = TTL_IMMUTABLE,
    ) -> Any:
        """Cache-first GET with rate limiting and retries."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        cache_key = self._cache_key(host, path, params)

        cached = self.cache.get_response(cache_key, max_age=max_age)
        if cached is not None:
            self.stats.cache_hits += 1
            log.debug("cache hit %s", cache_key)
            return cached

        url = f"https://{host}{path}"
        attempt = 0
        while True:
            self.limiter.acquire()
            self.stats.requests_made += 1
            try:
                response = self.session.get(
                    url, params=params or None, timeout=self.timeout
                )
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise RiotAPIError(f"network error after {attempt} retries: {exc}", url=url)
                attempt += 1
                self.stats.retries += 1
                self._backoff(attempt, reason=f"network error: {exc}")
                continue

            status = response.status_code

            if status == 200:
                payload = response.json()
                self.cache.put_response(cache_key, endpoint, url, payload)
                return payload

            if status in (401, 403):
                self.stats.record_error(status)
                raise RiotAuthError(
                    f"HTTP {status} from {endpoint}. The Riot API key is missing, invalid, or "
                    "expired. Personal development keys expire every 24 hours -- regenerate "
                    "one at https://developer.riotgames.com/ and update RIOT_API_KEY in .env.",
                    status=status,
                    url=url,
                )

            if status == 404:
                self.stats.not_found += 1
                raise RiotNotFoundError(f"HTTP 404 from {endpoint}", status=404, url=url)

            if status == 429:
                self.stats.rate_limited += 1
                if attempt >= self.max_retries:
                    raise RiotRateLimitError(
                        f"still rate limited after {attempt} retries on {endpoint}",
                        status=429,
                        url=url,
                    )
                attempt += 1
                self.stats.retries += 1
                retry_after = self._retry_after(response)
                limit_type = response.headers.get("X-Rate-Limit-Type", "unknown")
                log.warning(
                    "429 (%s limit) on %s -- sleeping %.1fs [attempt %d/%d]",
                    limit_type,
                    endpoint,
                    retry_after,
                    attempt,
                    self.max_retries,
                )
                time.sleep(retry_after)
                continue

            if status >= 500:
                self.stats.record_error(status)
                if attempt >= self.max_retries:
                    raise RiotServerError(
                        f"HTTP {status} from {endpoint} after {attempt} retries",
                        status=status,
                        url=url,
                    )
                attempt += 1
                self.stats.retries += 1
                self._backoff(attempt, reason=f"HTTP {status}")
                continue

            self.stats.record_error(status)
            raise RiotAPIError(
                f"HTTP {status} from {endpoint}: {response.text[:200]}", status=status, url=url
            )

    @staticmethod
    def _retry_after(response: requests.Response) -> float:
        """Riot sends Retry-After on 429; fall back to a safe default."""
        raw = response.headers.get("Retry-After")
        try:
            return max(1.0, float(raw)) + 0.5
        except (TypeError, ValueError):
            return 5.0

    def _backoff(self, attempt: int, reason: str) -> None:
        delay = min(60.0, (2 ** (attempt - 1))) + random.uniform(0, 0.5)
        log.warning("%s -- backing off %.1fs [attempt %d/%d]", reason, delay, attempt, self.max_retries)
        time.sleep(delay)

    # -- LEAGUE-V4 ---------------------------------------------------------

    def get_apex_league(self, tier: str, queue: str = "RANKED_SOLO_5x5") -> dict[str, Any]:
        """Challenger / Grandmaster / Master ladder for a queue.

        Returns a LeagueListDTO whose ``entries`` are LeagueItemDTOs (these lack
        a ``tier`` field of their own -- the tier is the league's).
        """
        tier = tier.upper()
        if tier not in APEX_TIERS:
            raise ValueError(f"{tier} is not an apex tier; expected one of {sorted(APEX_TIERS)}")
        slug = {"CHALLENGER": "challengerleagues", "GRANDMASTER": "grandmasterleagues",
                "MASTER": "masterleagues"}[tier]
        return self._get(
            f"{self.platform}.api.riotgames.com",
            f"/lol/league/v4/{slug}/by-queue/{queue}",
            endpoint=f"league-v4.{slug}",
            max_age=TTL_LADDER,
        )

    def get_league_entries(
        self,
        tier: str,
        division: str,
        queue: str = "RANKED_SOLO_5x5",
        page: int = 1,
    ) -> list[dict[str, Any]]:
        """One page of a non-apex division (IRON..DIAMOND). 205 entries per page."""
        return self._get(
            f"{self.platform}.api.riotgames.com",
            f"/lol/league/v4/entries/{queue}/{tier.upper()}/{division.upper()}",
            {"page": page},
            endpoint="league-v4.entries",
            max_age=TTL_LADDER,
        )

    def get_league_entries_by_puuid(self, puuid: str) -> list[dict[str, Any]]:
        """Current ranked entries for one player, across queues."""
        return self._get(
            f"{self.platform}.api.riotgames.com",
            f"/lol/league/v4/entries/by-puuid/{puuid}",
            endpoint="league-v4.entries-by-puuid",
            max_age=TTL_LADDER,
        )

    # -- SUMMONER-V4 -------------------------------------------------------

    def get_summoner_by_id(self, summoner_id: str) -> dict[str, Any]:
        """Resolve an encrypted summoner ID to a summoner record (incl. PUUID).

        Only needed for older ladder payloads that carry ``summonerId`` but no
        ``puuid``; current responses include the PUUID directly.
        """
        return self._get(
            f"{self.platform}.api.riotgames.com",
            f"/lol/summoner/v4/summoners/{summoner_id}",
            endpoint="summoner-v4.by-id",
            max_age=TTL_LADDER,
        )

    # -- ACCOUNT-V1 --------------------------------------------------------

    def get_account_by_riot_id(self, game_name: str, tag_line: str) -> dict[str, Any]:
        """Resolve a Riot ID ("Name#TAG") to an account record containing the PUUID.

        Lives on the *regional* host, not the platform one. The tag is passed
        without its leading '#'.
        """
        return self._get(
            f"{self.region}.api.riotgames.com",
            f"/riot/account/v1/accounts/by-riot-id/{quote(game_name)}/{quote(tag_line)}",
            endpoint="account-v1.by-riot-id",
            max_age=TTL_LADDER,
        )

    # -- CHAMPION-MASTERY-V4 -----------------------------------------------

    def get_champion_masteries(self, puuid: str) -> list[dict[str, Any]]:
        """Every champion mastery entry for a player, in one call.

        Mastery points accrue with every game played on a champion, and a win
        awards more than a loss, so a mastery total read *after* a match carries
        a faint trace of that match's result. One game is a fraction of a
        percent of a typical total, but features built from this should prefer
        log-scaled or rank-based forms over raw points.
        """
        return self._get(
            f"{self.platform}.api.riotgames.com",
            f"/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}",
            endpoint="champion-mastery-v4.by-puuid",
            max_age=TTL_LADDER,
        )

    # -- SPECTATOR-V5 ------------------------------------------------------

    def get_active_game(self, puuid: str) -> dict[str, Any]:
        """The game this player is currently in.

        Raises :class:`RiotNotFoundError` when they are not in one -- Riot
        signals "not in game" with a 404, so that is an ordinary answer here
        rather than an error worth retrying.
        """
        return self._get(
            f"{self.platform}.api.riotgames.com",
            f"/lol/spectator/v5/active-games/by-summoner/{puuid}",
            endpoint="spectator-v5.active-game",
            max_age=TTL_ACTIVE_GAME,
        )

    # -- MATCH-V5 ----------------------------------------------------------

    def get_match_ids(
        self,
        puuid: str,
        *,
        count: int = 20,
        start: int = 0,
        queue: int | None = 420,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[str]:
        """Recent match IDs for a player, newest first.

        ``queue=420`` is ranked solo/duo. ``start_time``/``end_time`` are epoch
        *seconds* and are how forward-only collection is expressed.
        """
        return self._get(
            f"{self.region}.api.riotgames.com",
            f"/lol/match/v5/matches/by-puuid/{puuid}/ids",
            {
                "start": start,
                "count": min(count, 100),
                "queue": queue,
                "startTime": start_time,
                "endTime": end_time,
            },
            endpoint="match-v5.ids-by-puuid",
            max_age=TTL_MATCH_IDS,
        )

    def get_match(self, match_id: str) -> dict[str, Any]:
        """Full match detail. A finished match is immutable, so cached forever."""
        return self._get(
            f"{self.region}.api.riotgames.com",
            f"/lol/match/v5/matches/{match_id}",
            endpoint="match-v5.match",
            max_age=TTL_IMMUTABLE,
        )


# --------------------------------------------------------------------------
# Normalisation helpers (pure functions -- unit tested without any network)
# --------------------------------------------------------------------------


def normalise_league_entry(entry: dict[str, Any], tier: str | None = None) -> dict[str, Any]:
    """Flatten a LeagueEntryDTO / LeagueItemDTO into our column names.

    Apex ``LeagueItemDTO`` entries carry no ``tier`` of their own, so the
    caller passes the league's tier in.
    """
    return {
        "puuid": entry.get("puuid"),
        "summoner_id": entry.get("summonerId"),
        "tier": (entry.get("tier") or tier or "").upper() or None,
        "rank_division": entry.get("rank"),
        "league_points": entry.get("leaguePoints"),
        "wins": entry.get("wins"),
        "losses": entry.get("losses"),
        "hot_streak": bool(entry.get("hotStreak")),
        "veteran": bool(entry.get("veteran")),
        "fresh_blood": bool(entry.get("freshBlood")),
        "inactive": bool(entry.get("inactive")),
    }


def normalise_match(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Split a MATCH-V5 payload into (match row, participant rows).

    Only pre-game identity and the match outcome are extracted. Per-participant
    performance (kills, gold, damage, ...) is deliberately dropped here: it is
    outcome-correlated and must not reach a pre-game feature table. The full
    payload is still in ``api_cache`` if an in-game model is ever wanted.
    """
    info = payload.get("info", {})
    metadata = payload.get("metadata", {})

    match_id = metadata.get("matchId")
    if not match_id:
        raise ValueError("match payload has no metadata.matchId")

    winning_team: int | None = None
    for team in info.get("teams", []):
        if team.get("win"):
            winning_team = team.get("teamId")
            break

    # Riot reports these in milliseconds.
    game_start_ms = info.get("gameStartTimestamp") or info.get("gameCreation")
    game_start_ts = game_start_ms / 1000.0 if game_start_ms else None

    match_row = {
        "match_id": match_id,
        "platform_id": info.get("platformId"),
        "region": None,  # filled in by the collector, which knows its routing
        "queue_id": info.get("queueId"),
        "game_version": info.get("gameVersion"),
        "game_start_ts": game_start_ts,
        "game_duration": info.get("gameDuration"),
        "winning_team": winning_team,
    }

    participants = [
        {
            "puuid": p.get("puuid"),
            "team_id": p.get("teamId"),
            "champion_id": p.get("championId"),
            "team_position": p.get("teamPosition") or None,
        }
        for p in info.get("participants", [])
        if p.get("puuid")
    ]

    return match_row, participants


def iter_apex_entries(league: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield normalised entries from an apex LeagueListDTO."""
    tier = league.get("tier")
    for entry in league.get("entries", []):
        yield normalise_league_entry(entry, tier=tier)
