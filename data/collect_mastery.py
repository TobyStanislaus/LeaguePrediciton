"""Collect champion mastery for the players who appear in collected matches.

    python -m data.collect_mastery --platform euw1 --max-players 12000

One CHAMPION-MASTERY-V4 call returns a player's entire mastery list, so this
costs roughly one request per unique player. Players are visited in order of
their most recent match, so stopping early still leaves complete coverage of the
newest matches -- the ones the staleness guard keeps.

A caveat worth stating plainly: mastery points are read *now*, and every game
played adds to them, with a win awarding more than a loss. A mastery total
therefore carries a faint trace of the match being predicted, in the same way a
current ladder snapshot does. The effect is far smaller (one game is a fraction
of a percent of a typical total, against roughly 18 LP on a 0-100 scale), but it
is the same shape of problem, so features built on this should use log-scaled or
rank-based forms and be ablated rather than trusted.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Sequence

from data.cache import DEFAULT_CACHE_PATH, Cache
from data.riot_client import RiotAPIError, RiotAuthError, RiotClient, RiotNotFoundError

log = logging.getLogger("collect_mastery")


def players_by_recency(cache: Cache, limit: int | None = None) -> list[str]:
    """PUUIDs ordered by their most recent collected match, newest first."""
    rows = cache.conn.execute(
        "SELECT mp.puuid AS puuid, MAX(m.game_start_ts) AS latest"
        " FROM match_participants mp JOIN matches m ON m.match_id = mp.match_id"
        " WHERE m.game_start_ts IS NOT NULL"
        " GROUP BY mp.puuid ORDER BY latest DESC"
    ).fetchall()
    puuids = [r["puuid"] for r in rows]
    return puuids if limit is None else puuids[:limit]


def collect_mastery(
    client: RiotClient,
    cache: Cache,
    puuids: Sequence[str],
    progress_every: int = 100,
) -> tuple[int, int]:
    """Fetch and store masteries. Returns (players done, mastery rows written)."""
    todo = [p for p in puuids if not cache.has_mastery(p)]
    log.info("%d players in scope, %d still to fetch", len(puuids), len(todo))

    done = 0
    written = 0
    for index, puuid in enumerate(todo, start=1):
        try:
            entries = client.get_champion_masteries(puuid)
        except RiotNotFoundError:
            cache.add_champion_masteries(puuid, [])
            continue

        written += cache.add_champion_masteries(puuid, entries)
        done += 1

        if index % progress_every == 0 or index == len(todo):
            stats = client.stats
            log.info(
                "mastery %d/%d | rows=%d | api calls=%d cache hits=%d retries=%d 429s=%d",
                index, len(todo), written,
                stats.requests_made, stats.cache_hits, stats.retries, stats.rate_limited,
            )

    return done, written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", default="euw1")
    parser.add_argument("--db", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument(
        "--max-players", type=int, default=None,
        help="cap on players; they are visited newest-match-first, so a cap still"
        " yields complete coverage of the most recent matches",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    started = time.time()
    cache = Cache(args.db)
    try:
        client = RiotClient(platform=args.platform, cache=cache)
    except RiotAuthError as exc:
        log.error("%s", exc)
        cache.close()
        return 2

    try:
        puuids = players_by_recency(cache, args.max_players)
        collect_mastery(client, cache, puuids)
    except KeyboardInterrupt:
        log.warning("interrupted -- progress is saved, re-run to resume")
    except RiotAuthError as exc:
        log.error("%s", exc)
        return 2
    except RiotAPIError as exc:
        log.error("API error: %s", exc)
        return 1
    finally:
        log.info("players with mastery stored: %d", cache.mastery_player_count())
        log.info("client: %s", client.stats.as_dict())
        log.info("elapsed: %.1fs", time.time() - started)
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
