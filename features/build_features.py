"""Turn collected matches and ladder snapshots into a team-level, pre-game feature table.

One row per match. Every feature is derived from state knowable **before**
kickoff: rank, LP, winrate, hot-streak flag, and the spread of rank within a
team. Nothing from in-game performance is used, or even loaded -- the collector
never stored it.

The joining problem
-------------------
LEAGUE-V4 gives a player's rank *now*, not their rank at kickoff. Joining a
current snapshot onto a past match leaks that match's own result into its
features. Three modes handle this differently:

``naive``
    Join the latest snapshot regardless of time. **Contaminated** -- the
    snapshot includes the outcome of the very match being predicted. Useful
    only as a deliberate demonstration of what leakage does to a score.

``point_in_time``
    Use only the most recent snapshot captured strictly *before* kickoff.
    Correct by construction, but yields nothing until snapshots predate
    matches (see ``--forward-only`` collection).

``reconstructed``
    Start from the snapshot and undo what happened between kickoff and capture.
    Wins and losses are integer counters, so they reconstruct **exactly**: if a
    player won the match being predicted, their pre-game win count was one
    lower. Any other collected matches in the interval are undone too.

    This is legitimate rather than circular: the label is used only to reverse
    its own effect on a counter, recovering the true pre-game value. LP cannot
    be recovered exactly (gains vary ~10-30 per game), so it is adjusted by a
    fixed ``lp_delta`` per undone match. That removes the systematic component
    of the leak; a small unbiased residual remains, which is why LP-derived
    features are the ones to drop first if results still look too good.

    The reconstruction is only as complete as the match history collected. An
    uncollected game in the interval stays baked into the counters -- but its
    result is uncorrelated with the match being predicted, so it is noise, not
    leakage.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from data.cache import DEFAULT_CACHE_PATH, Cache

log = logging.getLogger(__name__)

Mode = Literal["naive", "point_in_time", "reconstructed"]
MODES: tuple[Mode, ...] = ("naive", "point_in_time", "reconstructed")

BLUE, RED = 100, 200

# Riot's tiers, lowest first. Master/Grandmaster/Challenger share one continuous
# LP pool, so they all sit on the Master base and are separated by LP alone.
TIER_ORDER = (
    "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND",
    "MASTER", "GRANDMASTER", "CHALLENGER",
)
TIER_ORDINAL = {tier: i for i, tier in enumerate(TIER_ORDER)}
APEX_TIERS = {"MASTER", "GRANDMASTER", "CHALLENGER"}
DIVISION_ORDINAL = {"IV": 0, "III": 1, "II": 2, "I": 3}
TIER_SPAN = 400  # four divisions of 100 LP
MASTER_BASE = TIER_ORDINAL["MASTER"] * TIER_SPAN

# Average LP swing per ranked game, used to undo LP in ``reconstructed`` mode.
DEFAULT_LP_DELTA = 18.0

# Reconstruction is only valid while the snapshot still reflects the match.
#
# Undoing a result assumes the snapshot's counters contain it. Days later that
# holds. Months later it does not: LP is bounded per division and resets between
# splits, and the player has since played hundreds of games. Subtracting
# ``lp_delta * outcome`` from a number that no longer contains this match's
# result does not remove the outcome -- it stamps it in, inverted, and a model
# reads it straight back off. Observed for real: a mixed-tier set with a median
# snapshot age of 33 days scored 0.83 accuracy purely on that artefact.
#
# Beyond this age a (match, player) pair is simply not reconstructable, so it is
# dropped rather than silently fabricated.
DEFAULT_MAX_SNAPSHOT_AGE_DAYS = 7.0

# Per-team aggregates. The team-level feature names are built from these.
TEAM_STATS = (
    "rank_points_mean",
    "rank_points_median",
    "rank_points_std",
    "rank_points_min",
    "rank_points_max",
    "lp_mean",
    "winrate_mean",
    "games_mean",
    "hot_streak_count",
)

# Columns that are metadata, not model inputs.
METADATA_COLUMNS = ("match_id", "game_start_ts", "queue_id", "game_version", "leakage_mode")
LABEL_COLUMN = "blue_win"


def feature_columns() -> list[str]:
    """The exact set of model input columns, in a stable order."""
    columns = [f"blue_{stat}" for stat in TEAM_STATS]
    columns += [f"red_{stat}" for stat in TEAM_STATS]
    columns += [f"diff_{stat}" for stat in TEAM_STATS]
    return columns


# --------------------------------------------------------------------------
# Rank encoding
# --------------------------------------------------------------------------


def rank_points(tier: str | None, division: str | None, league_points: float | None) -> float:
    """Map (tier, division, LP) onto one monotone ladder scale.

    Non-apex: ``tier * 400 + division * 100 + LP``. Apex tiers share the Master
    base and are separated by their (unbounded) LP, which is how Riot actually
    ranks them.
    """
    if tier is None or (isinstance(tier, float) and np.isnan(tier)):
        return float("nan")
    tier = str(tier).upper()
    if tier not in TIER_ORDINAL:
        return float("nan")
    lp = 0.0 if league_points is None or pd.isna(league_points) else float(league_points)

    if tier in APEX_TIERS:
        return MASTER_BASE + lp

    division_index = DIVISION_ORDINAL.get(str(division).upper(), 0)
    return TIER_ORDINAL[tier] * TIER_SPAN + division_index * 100.0 + lp


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class RawFrames:
    matches: pd.DataFrame
    participants: pd.DataFrame
    entries: pd.DataFrame


def load_frames(cache: Cache) -> RawFrames:
    matches = pd.read_sql_query(
        "SELECT match_id, queue_id, game_version, game_start_ts, winning_team FROM matches"
        " WHERE game_start_ts IS NOT NULL AND winning_team IS NOT NULL",
        cache.conn,
    )
    participants = pd.read_sql_query(
        "SELECT match_id, puuid, team_id FROM match_participants", cache.conn
    )
    entries = pd.read_sql_query(
        "SELECT puuid, tier, rank_division, league_points, wins, losses, hot_streak,"
        " captured_at FROM league_entries",
        cache.conn,
    )
    return RawFrames(matches=matches, participants=participants, entries=entries)


# --------------------------------------------------------------------------
# Player state at kickoff
# --------------------------------------------------------------------------


def _select_entries(entries: pd.DataFrame, pairs: pd.DataFrame, mode: Mode) -> pd.DataFrame:
    """Attach one league entry to each (match, player) pair.

    ``point_in_time`` keeps only entries captured before kickoff; the other
    modes take the most recent entry available.
    """
    merged = pairs.merge(entries, on="puuid", how="inner")

    if mode == "point_in_time":
        merged = merged[merged["captured_at"] < merged["game_start_ts"]]
        if merged.empty:
            return merged
        # Most recent snapshot that still predates kickoff.
        merged = merged.sort_values("captured_at").groupby(
            ["match_id", "puuid"], as_index=False
        ).last()
    else:
        merged = merged.sort_values("captured_at").groupby(
            ["match_id", "puuid"], as_index=False
        ).last()

    return merged


def _reconstruct_counters(
    state: pd.DataFrame, player_matches: pd.DataFrame, lp_delta: float
) -> pd.DataFrame:
    """Undo results recorded between kickoff and snapshot capture.

    For each (match, player), subtract every collected match of that player at
    or after this kickoff and before the snapshot -- the match being predicted
    included, since its result is already in the snapshot's counters.
    """
    state = state.copy()

    # (puuid -> sorted arrays of that player's match times and outcomes)
    history: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for puuid, group in player_matches.groupby("puuid", sort=False):
        group = group.sort_values("game_start_ts")
        history[puuid] = (
            group["game_start_ts"].to_numpy(dtype=float),
            group["player_won"].to_numpy(dtype=bool),
        )

    wins_undone = np.zeros(len(state), dtype=float)
    losses_undone = np.zeros(len(state), dtype=float)

    start_ts = state["game_start_ts"].to_numpy(dtype=float)
    captured = state["captured_at"].to_numpy(dtype=float)
    puuids = state["puuid"].to_numpy()

    for i in range(len(state)):
        times, won = history.get(puuids[i], (None, None))
        if times is None:
            continue
        # Matches from this kickoff (inclusive) up to the snapshot (exclusive).
        lo = np.searchsorted(times, start_ts[i], side="left")
        hi = np.searchsorted(times, captured[i], side="left")
        if hi <= lo:
            continue
        window = won[lo:hi]
        wins_undone[i] = float(window.sum())
        losses_undone[i] = float(len(window) - window.sum())

    state["wins"] = state["wins"] - wins_undone
    state["losses"] = state["losses"] - losses_undone
    # LP cannot be recovered exactly; remove the systematic component.
    state["league_points"] = state["league_points"] - lp_delta * (wins_undone - losses_undone)
    state["matches_undone"] = wins_undone + losses_undone

    # Counters can only go negative if history is inconsistent; clamp defensively.
    state["wins"] = state["wins"].clip(lower=0)
    state["losses"] = state["losses"].clip(lower=0)
    state["league_points"] = state["league_points"].clip(lower=0)
    return state


def build_player_state(
    frames: RawFrames,
    mode: Mode,
    lp_delta: float = DEFAULT_LP_DELTA,
    max_snapshot_age_days: float | None = DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
) -> pd.DataFrame:
    """One row per (match, player) carrying that player's pre-game state."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    pairs = frames.participants.merge(
        frames.matches[["match_id", "game_start_ts", "winning_team"]], on="match_id", how="inner"
    )
    pairs["player_won"] = pairs["team_id"] == pairs["winning_team"]

    state = _select_entries(frames.entries, pairs, mode)
    if state.empty:
        return state

    if mode == "reconstructed":
        if max_snapshot_age_days is not None:
            age_days = (state["captured_at"] - state["game_start_ts"]) / 86400.0
            keep = age_days <= max_snapshot_age_days
            dropped = int((~keep).sum())
            if dropped:
                log.warning(
                    "dropping %d of %d (match, player) rows whose snapshot is more than "
                    "%.1f days after kickoff -- those results can no longer be undone "
                    "reliably, and adjusting them injects the outcome instead of removing it",
                    dropped,
                    len(state),
                    max_snapshot_age_days,
                )
            state = state[keep]
            if state.empty:
                return state

        state = _reconstruct_counters(
            state, pairs[["puuid", "game_start_ts", "player_won"]], lp_delta
        )

    state["rank_points"] = [
        rank_points(t, d, lp)
        for t, d, lp in zip(state["tier"], state["rank_division"], state["league_points"])
    ]
    games = state["wins"] + state["losses"]
    state["games"] = games
    state["winrate"] = np.where(games > 0, state["wins"] / games.replace(0, np.nan), np.nan)
    return state


# --------------------------------------------------------------------------
# Team aggregation
# --------------------------------------------------------------------------


def aggregate_teams(state: pd.DataFrame, require_full_teams: bool = True) -> pd.DataFrame:
    """Aggregate player state into one row per (match, team)."""
    grouped = state.groupby(["match_id", "team_id"])
    teams = grouped.agg(
        rank_points_mean=("rank_points", "mean"),
        rank_points_median=("rank_points", "median"),
        rank_points_std=("rank_points", "std"),
        rank_points_min=("rank_points", "min"),
        rank_points_max=("rank_points", "max"),
        lp_mean=("league_points", "mean"),
        winrate_mean=("winrate", "mean"),
        games_mean=("games", "mean"),
        hot_streak_count=("hot_streak", "sum"),
        players=("puuid", "count"),
    ).reset_index()

    # A single-player team has no spread; std is NaN rather than 0.
    teams["rank_points_std"] = teams["rank_points_std"].fillna(0.0)

    if require_full_teams:
        teams = teams[teams["players"] == 5]
    return teams


def build_feature_table(
    cache: Cache,
    mode: Mode = "reconstructed",
    lp_delta: float = DEFAULT_LP_DELTA,
    require_full_teams: bool = True,
    max_snapshot_age_days: float | None = DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
) -> pd.DataFrame:
    """Build the model-ready table: one row per match, blue/red/diff features."""
    frames = load_frames(cache)
    state = build_player_state(
        frames, mode=mode, lp_delta=lp_delta, max_snapshot_age_days=max_snapshot_age_days
    )

    empty = pd.DataFrame(columns=list(METADATA_COLUMNS) + feature_columns() + [LABEL_COLUMN])
    if state.empty:
        log.warning("mode %r produced no usable (match, player) rows", mode)
        return empty

    teams = aggregate_teams(state, require_full_teams=require_full_teams)
    if teams.empty:
        return empty

    blue = teams[teams["team_id"] == BLUE].set_index("match_id")
    red = teams[teams["team_id"] == RED].set_index("match_id")
    both = blue.index.intersection(red.index)
    if len(both) == 0:
        log.warning("no match has both teams fully resolved")
        return empty

    blue, red = blue.loc[both], red.loc[both]

    table = pd.DataFrame(index=both)
    for stat in TEAM_STATS:
        table[f"blue_{stat}"] = blue[stat]
        table[f"red_{stat}"] = red[stat]
        table[f"diff_{stat}"] = blue[stat] - red[stat]

    meta = frames.matches.set_index("match_id").loc[both]
    table["game_start_ts"] = meta["game_start_ts"]
    table["queue_id"] = meta["queue_id"]
    table["game_version"] = meta["game_version"]
    table[LABEL_COLUMN] = (meta["winning_team"] == BLUE).astype(int)
    table["leakage_mode"] = mode

    table = table.reset_index().rename(columns={"index": "match_id"})
    table = table.sort_values("game_start_ts").reset_index(drop=True)

    ordered = list(METADATA_COLUMNS) + feature_columns() + [LABEL_COLUMN]
    return table[ordered]


def save_table(table: pd.DataFrame, path: Path) -> Path:
    """Write the feature table, falling back to CSV if Parquet is unavailable.

    Parquet needs pyarrow's compiled extension, which some Windows Application
    Control policies block. Returns the path actually written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        table.to_csv(path, index=False)
        return path
    try:
        table.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError) as exc:
        fallback = path.with_suffix(".csv")
        log.warning("parquet unavailable (%s) -- writing %s instead", exc, fallback)
        table.to_csv(fallback, index=False)
        return fallback


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=MODES, default="reconstructed")
    parser.add_argument("--db", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--out", default="data/processed/features.parquet")
    parser.add_argument("--lp-delta", type=float, default=DEFAULT_LP_DELTA)
    parser.add_argument(
        "--max-snapshot-age-days", type=float, default=DEFAULT_MAX_SNAPSHOT_AGE_DAYS,
        help="in reconstructed mode, drop pairs whose snapshot is older than this;"
        " beyond it the result can no longer be undone without injecting the outcome",
    )
    parser.add_argument(
        "--allow-partial-teams", action="store_true",
        help="keep matches where fewer than 5 players per side have a rank",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)-7s %(name)s | %(message)s",
    )

    with Cache(args.db) as cache:
        table = build_feature_table(
            cache,
            mode=args.mode,
            lp_delta=args.lp_delta,
            require_full_teams=not args.allow_partial_teams,
            max_snapshot_age_days=args.max_snapshot_age_days,
        )

    if table.empty:
        log.error("no rows produced in mode %r", args.mode)
        return 1

    written = save_table(table, Path(args.out))

    log.info("mode=%s rows=%d features=%d", args.mode, len(table), len(feature_columns()))
    log.info(
        "label balance: blue wins %.1f%%", 100.0 * table[LABEL_COLUMN].mean()
    )
    log.info(
        "time span: %s -> %s",
        pd.to_datetime(table["game_start_ts"].min(), unit="s"),
        pd.to_datetime(table["game_start_ts"].max(), unit="s"),
    )
    log.info("wrote %s", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
