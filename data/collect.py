"""Collection driver: ladder seed -> match IDs -> match detail -> per-player league stats.

Run::

    python -m data.collect --platform euw1 --max-summoners 50 --matches-per-summoner 20

Three phases, each resumable and cache-backed:

1. **Seed** -- pull a ranked ladder page (apex tiers, or a tier/division) and
   record every entry as a point-in-time ``league_snapshots`` row.
2. **Matches** -- for each seed PUUID, list recent ranked match IDs (MATCH-V5)
   and fetch full detail for the ones we do not already hold.
3. **Participants** -- for every player appearing in a collected match but not
   yet in a snapshot, fetch their LEAGUE-V4 entry (tier, LP, wins/losses, hot
   streak). A pre-game team feature table needs rank for all ten players, not
   just the seed.

A note on timing and leakage
----------------------------
LEAGUE-V4 returns a player's rank *right now*, not their rank when some past
match started. For a match played before the snapshot, the LP and win/loss
counts we record already include that match's result -- which is leakage into a
pre-game prediction.

Every league row therefore carries ``captured_at`` and every match carries
``game_start_ts``, so feature building can do a proper point-in-time join
(snapshot strictly before kickoff). ``--forward-only`` enforces this at
collection time by asking Riot for matches played *after* the snapshot; it
returns little on a first run and fills up as you re-run over following days.
Without it you get usable data immediately, at the cost of rows the feature
stage will have to treat carefully.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any, Iterable, Sequence

from data.cache import DEFAULT_CACHE_PATH, Cache
from data.riot_client import (
    APEX_TIERS,
    RiotAPIError,
    RiotAuthError,
    RiotClient,
    RiotNotFoundError,
    iter_apex_entries,
    normalise_league_entry,
    normalise_match,
)

log = logging.getLogger("collect")

RANKED_SOLO_QUEUE_ID = 420
NON_APEX_DIVISIONS = ("I", "II", "III", "IV")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# --------------------------------------------------------------------------
# Phase 1 -- seed the ladder
# --------------------------------------------------------------------------


def seed_ladder(
    client: RiotClient,
    cache: Cache,
    tiers: Sequence[str],
    queue: str = "RANKED_SOLO_5x5",
    divisions: Sequence[str] = NON_APEX_DIVISIONS,
    pages: int = 1,
    max_summoners: int | None = None,
) -> tuple[int, list[str]]:
    """Record a ladder snapshot. Returns (snapshot_id, puuids)."""
    snapshot_id = cache.start_league_snapshot(
        client.platform, queue, note=f"seed tiers={','.join(tiers)}"
    )
    entries: list[dict[str, Any]] = []

    for tier in tiers:
        tier = tier.upper()
        if tier in APEX_TIERS:
            league = client.get_apex_league(tier, queue=queue)
            tier_entries = list(iter_apex_entries(league))
            log.info("seed %s: %d entries", tier, len(tier_entries))
            entries.extend(tier_entries)
        else:
            for division in divisions:
                for page in range(1, pages + 1):
                    page_entries = client.get_league_entries(
                        tier, division, queue=queue, page=page
                    )
                    if not page_entries:
                        break
                    normalised = [normalise_league_entry(e, tier=tier) for e in page_entries]
                    log.info(
                        "seed %s %s page %d: %d entries", tier, division, page, len(normalised)
                    )
                    entries.extend(normalised)

    # Store the whole ladder, not just the seed subset: it all arrived in the
    # same call, and every entry recorded here is one less phase-3 lookup later.
    written = cache.add_league_entries(snapshot_id, entries)

    # Only the seed subset drives match collection, so only it needs a PUUID
    # resolved -- resolving all of them could cost hundreds of extra calls.
    seeds = entries if max_summoners is None else entries[:max_summoners]
    seeds = resolve_missing_puuids(client, seeds)
    puuids = [e["puuid"] for e in seeds if e.get("puuid")]

    log.info(
        "snapshot %d: wrote %d league entries; seeding matches from %d players",
        snapshot_id,
        written,
        len(puuids),
    )
    return snapshot_id, puuids


def resolve_missing_puuids(
    client: RiotClient, entries: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Fill in PUUIDs for ladder entries that only carry a summoner ID.

    Current LEAGUE-V4 responses include ``puuid`` directly; this is a fallback
    for older payloads, and costs one SUMMONER-V4 call per unresolved entry.
    """
    resolved: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("puuid"):
            resolved.append(entry)
            continue
        summoner_id = entry.get("summoner_id")
        if not summoner_id:
            continue
        try:
            summoner = client.get_summoner_by_id(summoner_id)
        except RiotNotFoundError:
            log.debug("summoner %s not found, skipping", summoner_id)
            continue
        entry = dict(entry, puuid=summoner.get("puuid"))
        if entry["puuid"]:
            resolved.append(entry)
    return resolved


# --------------------------------------------------------------------------
# Phase 2 -- matches
# --------------------------------------------------------------------------


def collect_matches(
    client: RiotClient,
    cache: Cache,
    puuids: Sequence[str],
    matches_per_summoner: int = 20,
    queue_id: int | None = RANKED_SOLO_QUEUE_ID,
    start_time: int | None = None,
    progress_every: int = 10,
) -> set[str]:
    """Fetch match detail for each seed player's recent matches.

    Returns the set of PUUIDs seen across all collected matches (input to
    phase 3). Already-stored matches are skipped without a network call.
    """
    seen_puuids: set[str] = set()
    new_matches = 0
    skipped = 0

    for index, puuid in enumerate(puuids, start=1):
        progress_key = f"matchids:{puuid}:{start_time or 0}"
        try:
            match_ids = client.get_match_ids(
                puuid, count=matches_per_summoner, queue=queue_id, start_time=start_time
            )
        except RiotNotFoundError:
            log.debug("no match list for %s", puuid[:12])
            continue

        for match_id in match_ids:
            if cache.has_match(match_id):
                skipped += 1
                row = cache.conn.execute(
                    "SELECT puuid FROM match_participants WHERE match_id = ?", (match_id,)
                ).fetchall()
                seen_puuids.update(r["puuid"] for r in row)
                continue
            try:
                payload = client.get_match(match_id)
            except RiotNotFoundError:
                log.debug("match %s not found", match_id)
                continue

            match_row, participants = normalise_match(payload)
            match_row["region"] = client.region
            cache.add_match(match_row, participants)
            seen_puuids.update(p["puuid"] for p in participants)
            new_matches += 1

        cache.mark_done(progress_key)

        if index % progress_every == 0 or index == len(puuids):
            stats = client.stats
            log.info(
                "summoners %d/%d | matches new=%d cached=%d | api calls=%d cache hits=%d "
                "retries=%d 429s=%d",
                index,
                len(puuids),
                new_matches,
                skipped,
                stats.requests_made,
                stats.cache_hits,
                stats.retries,
                stats.rate_limited,
            )

    return seen_puuids


# --------------------------------------------------------------------------
# Phase 3 -- league stats for every participant
# --------------------------------------------------------------------------


def collect_participant_league_entries(
    client: RiotClient,
    cache: Cache,
    puuids: Iterable[str],
    queue: str = "RANKED_SOLO_5x5",
    max_lookups: int | None = None,
    progress_every: int = 50,
) -> int:
    """Fetch LEAGUE-V4 entries for players we have matches for but no rank.

    Recorded as its own snapshot so ``captured_at`` reflects when these rows
    were actually read, not when phase 1 started.
    """
    known = set(cache.known_puuids())
    todo = [p for p in dict.fromkeys(puuids) if p not in known]
    if max_lookups is not None:
        todo = todo[:max_lookups]
    if not todo:
        log.info("phase 3: every participant already has a league entry")
        return 0

    snapshot_id = cache.start_league_snapshot(
        client.platform, queue, note="match participants"
    )
    log.info("phase 3: looking up %d participants (snapshot %d)", len(todo), snapshot_id)

    written = 0
    for index, puuid in enumerate(todo, start=1):
        try:
            entries = client.get_league_entries_by_puuid(puuid)
        except RiotNotFoundError:
            continue

        for raw in entries:
            if raw.get("queueType") != queue:
                continue
            entry = normalise_league_entry(raw)
            entry["puuid"] = entry.get("puuid") or puuid
            written += cache.add_league_entries(snapshot_id, [entry])

        if index % progress_every == 0 or index == len(todo):
            log.info(
                "phase 3: %d/%d looked up, %d entries written (api calls=%d cache hits=%d)",
                index,
                len(todo),
                written,
                client.stats.requests_made,
                client.stats.cache_hits,
            )

    return written


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Riot ranked ladder and match data into the local cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--platform", default="euw1", help="platform routing value, e.g. euw1")
    parser.add_argument("--queue", default="RANKED_SOLO_5x5", help="ranked queue name")
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["DIAMOND"],
        help="tiers to seed from, e.g. CHALLENGER GRANDMASTER, or DIAMOND EMERALD",
    )
    parser.add_argument(
        "--divisions", nargs="+", default=list(NON_APEX_DIVISIONS),
        help="divisions to seed for non-apex tiers",
    )
    parser.add_argument("--pages", type=int, default=1, help="ladder pages per tier/division")
    parser.add_argument(
        "--max-summoners", type=int, default=50,
        help="cap on seed players (each costs 1 match-list call plus match detail calls)",
    )
    parser.add_argument(
        "--matches-per-summoner", type=int, default=20, help="recent matches to pull per player"
    )
    parser.add_argument(
        "--max-participant-lookups", type=int, default=500,
        help="cap on phase 3 LEAGUE-V4 lookups; None-equivalent is a very large number",
    )
    parser.add_argument(
        "--forward-only", action="store_true",
        help="only collect matches that started after the ladder snapshot "
             "(point-in-time correct, but returns little until you re-run later)",
    )
    parser.add_argument(
        "--skip-participants", action="store_true", help="skip phase 3"
    )
    parser.add_argument("--db", default=str(DEFAULT_CACHE_PATH), help="SQLite cache path")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    started = time.time()
    cache = Cache(args.db)

    try:
        client = RiotClient(platform=args.platform, cache=cache)
    except RiotAuthError as exc:
        log.error("%s", exc)
        cache.close()
        return 2

    try:
        snapshot_id, puuids = seed_ladder(
            client,
            cache,
            tiers=args.tiers,
            queue=args.queue,
            divisions=args.divisions,
            pages=args.pages,
            max_summoners=args.max_summoners,
        )
        if not puuids:
            log.error("seed produced no PUUIDs -- nothing to collect")
            return 1

        start_time = None
        if args.forward_only:
            row = cache.conn.execute(
                "SELECT captured_at FROM league_snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
            start_time = int(row["captured_at"])
            log.info(
                "forward-only: requesting matches started after %s",
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            )
        else:
            log.warning(
                "collecting matches played BEFORE the ladder snapshot. Their recorded LP and "
                "win/loss counts already include those results -- feature building must do a "
                "point-in-time join or this leaks. Use --forward-only for clean rows."
            )

        seen = collect_matches(
            client,
            cache,
            puuids,
            matches_per_summoner=args.matches_per_summoner,
            queue_id=RANKED_SOLO_QUEUE_ID,
            start_time=start_time,
        )

        if not args.skip_participants:
            collect_participant_league_entries(
                client, cache, seen, queue=args.queue,
                max_lookups=args.max_participant_lookups,
            )

    except RiotAuthError as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.warning("interrupted -- progress is saved, re-run the same command to resume")
    except RiotAPIError as exc:
        log.error("API error: %s", exc)
        return 1
    finally:
        elapsed = time.time() - started
        log.info("cache: %s", cache.summary())
        log.info("client: %s", client.stats.as_dict())
        log.info("elapsed: %.1fs (rate-limit sleep %.1fs)", elapsed, client.limiter.total_wait)
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
