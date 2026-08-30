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
from dataclasses import dataclass, field
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


# Role-aware features. Team averages hide *where* a rank gap sits, and a two
# hundred point edge in the jungle is not the same game as the same edge in
# support. Riot assigns these at champion select, so they are pre-game.
ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
ROLE_FEATURES = tuple(f"role_diff_{role.lower()}" for role in ROLES) + ("role_diff_spread",)

# Measured, not assumed: a rolling-origin ablation over six sequential folds on
# 2,113 matches gave a mean accuracy gain of -0.011 (improving in 1 fold of 6),
# with log loss and AUC flat. A single split had suggested +0.022, which turned
# out to be split luck.
#
# They also cost something at prediction time: SPECTATOR-V5 reports no assigned
# roles, so a live game leaves all six columns to the imputer.
#
# The code and tests are kept so the result is reproducible and the switch is
# one line if better data changes the answer.
INCLUDE_ROLE_FEATURES = False

# Champion-select features: how strong each side's five champions have been,
# from *earlier* matches only. Champions are locked before the game starts, so
# this is pre-game -- provided the winrate is built strictly from the past.
CHAMPION_FEATURES = ("blue_champ_winrate", "red_champ_winrate", "diff_champ_winrate")

# Also measured, also negative: mean accuracy gain -0.0024 (se 0.0047) over the
# same six folds, AUC -0.0007, and diff_champ_winrate correlates with the label
# at 0.004.
#
# The reason is structural rather than a bug. Riot balances champions to roughly
# even winrates, the spread across 173 of them is small, and averaging five per
# side cancels most of what remains: the resulting diff has a standard deviation
# of 0.019. Champion *identity* is not the signal. Player proficiency on that
# champion plausibly is, and that needs CHAMPION-MASTERY-V4 -- about 15,600
# calls for this dataset, so it is a deliberate decision rather than a free one.
INCLUDE_CHAMPION_FEATURES = False

# Champion mastery: how practised each player is on the champion they locked.
# This is the proficiency signal champion identity failed to carry.
#
# Mastery points are read now and grow with every game, a win awarding more than
# a loss, so a raw total carries a faint trace of the match being predicted --
# the same shape of problem as a current ladder snapshot, roughly two orders of
# magnitude smaller. Both features below are chosen to blunt it: a log scale
# makes one game's points negligible, and a champion's rank within a player's
# own pool almost never moves on a single game.
MASTERY_FEATURES = (
    "blue_mastery_log_mean",
    "red_mastery_log_mean",
    "diff_mastery_log_mean",
    "blue_mastery_rank_mean",
    "red_mastery_rank_mean",
    "diff_mastery_rank_mean",
)
#
# Status: promising, not yet established. On the 464 matches with full coverage
# (22% of the table, 6,000 of 15,595 players fetched), a rolling-origin ablation
# gave a mean accuracy gain of +0.0215 with 4 of 5 folds improving -- but a
# standard error of 0.0207, so about one sigma. For contrast, role features
# improved 1 fold of 6 and champion winrate 2 of 6.
#
# The covered subset is also not a fair sample: base accuracy on it is ~0.70
# against ~0.63 overall, because a match only qualifies when all ten players are
# among the most recently active. Fuller coverage is being collected to settle
# this; revisit the switch when it lands.
INCLUDE_MASTERY_FEATURES = True

# Beta prior strength for champion winrates. A champion seen five times should
# not be credited with an 80% winrate, so estimates shrink toward 0.5 until the
# sample earns otherwise.
CHAMPION_PRIOR_STRENGTH = 25.0


def feature_columns() -> list[str]:
    """The exact set of model input columns, in a stable order."""
    columns = [f"blue_{stat}" for stat in TEAM_STATS]
    columns += [f"red_{stat}" for stat in TEAM_STATS]
    columns += [f"diff_{stat}" for stat in TEAM_STATS]
    if INCLUDE_ROLE_FEATURES:
        columns += list(ROLE_FEATURES)
    if INCLUDE_CHAMPION_FEATURES:
        columns += list(CHAMPION_FEATURES)
    if INCLUDE_MASTERY_FEATURES:
        columns += list(MASTERY_FEATURES)
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
    mastery: pd.DataFrame = field(default_factory=pd.DataFrame)
    mastery_players: frozenset[str] = frozenset()


def load_frames(cache: Cache) -> RawFrames:
    matches = pd.read_sql_query(
        "SELECT match_id, queue_id, game_version, game_start_ts, winning_team FROM matches"
        " WHERE game_start_ts IS NOT NULL AND winning_team IS NOT NULL",
        cache.conn,
    )
    participants = pd.read_sql_query(
        "SELECT match_id, puuid, team_id, team_position, champion_id FROM match_participants",
        cache.conn,
    )
    entries = pd.read_sql_query(
        "SELECT puuid, tier, rank_division, league_points, wins, losses, hot_streak,"
        " captured_at FROM league_entries",
        cache.conn,
    )
    mastery = pd.read_sql_query(
        "SELECT puuid, champion_id, mastery_points FROM champion_mastery", cache.conn
    )
    # Players we actually asked about. A player who was asked but has no entry
    # for a champion has genuinely never played it (0 points); a player we never
    # asked about is simply unknown, and the two must not be confused.
    asked = pd.read_sql_query(
        "SELECT SUBSTR(key, 9) AS puuid FROM collection_progress WHERE key LIKE 'mastery:%'",
        cache.conn,
    )

    return RawFrames(
        matches=matches,
        participants=participants,
        entries=entries,
        mastery=mastery,
        mastery_players=frozenset(asked["puuid"]),
    )


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

    return derive_player_metrics(state)


def derive_player_metrics(players: pd.DataFrame) -> pd.DataFrame:
    """Add ``rank_points``, ``games`` and ``winrate`` to raw player rows.

    Split out from ``build_player_state`` so prediction can reuse it for
    players who are not part of any stored match.
    """
    players = players.copy()
    players["rank_points"] = [
        rank_points(t, d, lp)
        for t, d, lp in zip(players["tier"], players["rank_division"], players["league_points"])
    ]
    games = players["wins"] + players["losses"]
    players["games"] = games
    players["winrate"] = np.where(
        games > 0, players["wins"] / games.replace(0, np.nan), np.nan
    )
    return players


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


def role_features(state: pd.DataFrame) -> pd.DataFrame:
    """Per-role rank gaps: blue's player minus red's, lane by lane.

    Indexed by match_id, one column per role plus the spread between the
    largest and smallest gap -- a team ahead everywhere plays differently from
    one ahead in two lanes and behind in two.

    Roles Riot did not label come back as NaN rather than 0: a missing gap is
    unknown, and zero would assert the lanes were even.
    """
    empty = pd.DataFrame(columns=list(ROLE_FEATURES))
    if "team_position" not in state.columns or state.empty:
        return empty

    labelled = state[state["team_position"].isin(ROLES)]
    if labelled.empty:
        return empty

    # Mean guards against the rare duplicate role within one team.
    per_side = labelled.groupby(["match_id", "team_position", "team_id"])[
        "rank_points"
    ].mean()
    wide = per_side.unstack("team_id")
    if BLUE not in wide.columns or RED not in wide.columns:
        return empty

    gaps = (wide[BLUE] - wide[RED]).unstack("team_position")
    gaps = gaps.reindex(columns=list(ROLES))
    gaps.columns = [f"role_diff_{role.lower()}" for role in gaps.columns]

    gaps["role_diff_spread"] = gaps.max(axis=1) - gaps.min(axis=1)
    return gaps


def champion_prior_winrates(
    matches: pd.DataFrame,
    participants: pd.DataFrame,
    alpha: float = CHAMPION_PRIOR_STRENGTH,
) -> pd.DataFrame:
    """Mean champion winrate per side, computed from strictly earlier matches.

    Walks matches in kickoff order, reads each champion's record *before*
    scoring the match, and only then folds that match's result in. A winrate
    computed over the whole dataset would include the outcome being predicted --
    the same mistake as joining a current ladder snapshot onto a past game, in a
    different costume.

    Estimates are shrunk toward 0.5 by a Beta(alpha, alpha) prior so a champion
    with four games does not arrive claiming a 75% winrate.
    """
    empty = pd.DataFrame(columns=list(CHAMPION_FEATURES))
    needed = {"match_id", "game_start_ts", "winning_team"}
    if matches.empty or participants.empty or not needed <= set(matches.columns):
        return empty
    if "champion_id" not in participants.columns:
        return empty

    ordered = matches.dropna(subset=["game_start_ts", "winning_team"]).sort_values(
        "game_start_ts"
    )
    by_match: dict[str, list[tuple[int, int]]] = {}
    for match_id, team_id, champion_id in zip(
        participants["match_id"], participants["team_id"], participants["champion_id"]
    ):
        if champion_id is None or pd.isna(champion_id):
            continue
        by_match.setdefault(match_id, []).append((int(team_id), int(champion_id)))

    wins: dict[int, float] = {}
    games: dict[int, float] = {}
    rows = []

    for match_id, winning_team in zip(ordered["match_id"], ordered["winning_team"]):
        squad = by_match.get(match_id)
        if not squad:
            continue

        sides: dict[int, list[float]] = {BLUE: [], RED: []}
        for team_id, champion_id in squad:
            if team_id not in sides:
                continue
            prior = (wins.get(champion_id, 0.0) + alpha) / (
                games.get(champion_id, 0.0) + 2 * alpha
            )
            sides[team_id].append(prior)

        if sides[BLUE] and sides[RED]:
            blue_mean = float(np.mean(sides[BLUE]))
            red_mean = float(np.mean(sides[RED]))
            rows.append(
                {
                    "match_id": match_id,
                    "blue_champ_winrate": blue_mean,
                    "red_champ_winrate": red_mean,
                    "diff_champ_winrate": blue_mean - red_mean,
                }
            )

        # Only now does this match count toward the running record.
        for team_id, champion_id in squad:
            games[champion_id] = games.get(champion_id, 0.0) + 1.0
            if team_id == winning_team:
                wins[champion_id] = wins.get(champion_id, 0.0) + 1.0

    if not rows:
        return empty
    return pd.DataFrame(rows).set_index("match_id")


def mastery_features(frames: RawFrames) -> pd.DataFrame:
    """Per-side champion-mastery summaries, indexed by match_id.

    Two views of the same thing, both blunt to a single game's contribution:

    * ``mastery_log_mean`` -- mean of ``log1p(points)`` on the champion played.
    * ``mastery_rank_mean`` -- mean rank of that champion inside the player's
      own pool, 1 being their most-played. Lower is more comfortable.

    A player we asked about who has no row for a champion has never played it,
    which is 0 points and a rank past the end of their pool. A player we never
    asked about is NaN -- unknown is not the same as zero.
    """
    empty = pd.DataFrame(columns=list(MASTERY_FEATURES))
    if frames.mastery.empty or not frames.mastery_players:
        return empty

    mastery = frames.mastery.copy()
    mastery["champion_id"] = mastery["champion_id"].astype("int64")
    mastery["mastery_points"] = mastery["mastery_points"].fillna(0).astype("float64")
    mastery["mastery_rank"] = (
        mastery.groupby("puuid")["mastery_points"].rank(ascending=False, method="min")
    )
    pool_size = mastery.groupby("puuid")["champion_id"].size().rename("pool_size")

    played = frames.participants.dropna(subset=["champion_id"]).copy()
    played["champion_id"] = played["champion_id"].astype("int64")
    played = played[played["puuid"].isin(frames.mastery_players)]
    if played.empty:
        return empty

    played = played.merge(
        mastery[["puuid", "champion_id", "mastery_points", "mastery_rank"]],
        on=["puuid", "champion_id"],
        how="left",
    ).merge(pool_size, on="puuid", how="left")

    # Asked-about players with no entry: never played it.
    played["mastery_points"] = played["mastery_points"].fillna(0.0)
    played["mastery_rank"] = played["mastery_rank"].fillna(played["pool_size"] + 1)
    played["mastery_log"] = np.log1p(played["mastery_points"])

    grouped = played.groupby(["match_id", "team_id"]).agg(
        mastery_log_mean=("mastery_log", "mean"),
        mastery_rank_mean=("mastery_rank", "mean"),
        players=("puuid", "count"),
    ).reset_index()
    # Only summarise a side we saw all five of; a three-player average is a
    # different quantity wearing the same column name.
    grouped = grouped[grouped["players"] == 5]

    blue = grouped[grouped["team_id"] == BLUE].set_index("match_id")
    red = grouped[grouped["team_id"] == RED].set_index("match_id")
    both = blue.index.intersection(red.index)
    if len(both) == 0:
        return empty

    blue, red = blue.loc[both], red.loc[both]
    out = pd.DataFrame(index=both)
    for stat in ("mastery_log_mean", "mastery_rank_mean"):
        out[f"blue_{stat}"] = blue[stat]
        out[f"red_{stat}"] = red[stat]
        out[f"diff_{stat}"] = blue[stat] - red[stat]
    return out


def pivot_team_features(
    teams: pd.DataFrame,
    roles: pd.DataFrame | None = None,
    champions: pd.DataFrame | None = None,
    mastery: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Turn per-(match, team) aggregates into one row per match.

    Produces ``blue_*``, ``red_*``, ``diff_*`` and ``role_diff_*`` columns.
    Matches missing either side are dropped.
    """
    blue = teams[teams["team_id"] == BLUE].set_index("match_id")
    red = teams[teams["team_id"] == RED].set_index("match_id")
    both = blue.index.intersection(red.index)
    if len(both) == 0:
        return pd.DataFrame(columns=feature_columns())

    blue, red = blue.loc[both], red.loc[both]
    table = pd.DataFrame(index=both)
    for stat in TEAM_STATS:
        table[f"blue_{stat}"] = blue[stat]
        table[f"red_{stat}"] = red[stat]
        table[f"diff_{stat}"] = blue[stat] - red[stat]

    if INCLUDE_ROLE_FEATURES:
        for column in ROLE_FEATURES:
            if roles is not None and column in roles:
                table[column] = roles.reindex(both)[column]
            else:
                # Live prediction has no role labels (see features_for_players).
                table[column] = np.nan

    if INCLUDE_CHAMPION_FEATURES:
        for column in CHAMPION_FEATURES:
            if champions is not None and column in champions:
                table[column] = champions.reindex(both)[column]
            else:
                table[column] = np.nan

    if INCLUDE_MASTERY_FEATURES:
        for column in MASTERY_FEATURES:
            if mastery is not None and column in mastery:
                table[column] = mastery.reindex(both)[column]
            else:
                table[column] = np.nan

    # Return in the declared order, not construction order. A model fed these
    # columns in a different order than it was fitted on produces confident
    # nonsense, and only sklearn's feature-name check stands between the two.
    return table[feature_columns()]


def features_for_players(
    players: pd.DataFrame, match_id: str = "live", require_full_teams: bool = True
) -> pd.DataFrame:
    """Build one feature row from ten players' current ranked state.

    This is the prediction path: no stored match, no label, no reconstruction
    -- the ranks *are* pre-game because the game has not been played yet.

    ``players`` needs ``puuid``, ``team_id`` (100/200), ``tier``,
    ``rank_division``, ``league_points``, ``wins``, ``losses``, ``hot_streak``.
    An optional ``team_position`` enables the role features.

    SPECTATOR-V5 does not report assigned roles, so a live prediction leaves
    ``role_diff_*`` as NaN for the imputer to fill. Those predictions are
    therefore made on fewer effective features than the model was trained with.
    """
    required = {
        "puuid", "team_id", "tier", "rank_division", "league_points",
        "wins", "losses", "hot_streak",
    }
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"players is missing columns: {sorted(missing)}")

    players = players.copy()
    players["match_id"] = match_id
    if "team_position" not in players.columns:
        players["team_position"] = None

    state = derive_player_metrics(players)
    teams = aggregate_teams(state, require_full_teams=require_full_teams)
    return pivot_team_features(teams, roles=role_features(state))


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

    table = pivot_team_features(
        teams,
        roles=role_features(state),
        # Built from every collected match, not only the usable ones: more
        # history behind each winrate, and still strictly past-only.
        champions=champion_prior_winrates(frames.matches, frames.participants),
        mastery=mastery_features(frames),
    )
    if table.empty:
        log.warning("no match has both teams fully resolved")
        return empty

    both = table.index
    meta = frames.matches.set_index("match_id").loc[both]
    table["game_start_ts"] = meta["game_start_ts"]
    table["queue_id"] = meta["queue_id"]
    table["game_version"] = meta["game_version"]
    table[LABEL_COLUMN] = (meta["winning_team"] == BLUE).astype(int)
    table["leakage_mode"] = mode

    table = table.reset_index().rename(columns={"index": "match_id"})
    table = table.sort_values("game_start_ts").reset_index(drop=True)

    # An all-NaN feature is imputed to a constant and contributes nothing, but
    # it looks like a working column: an ablation against it reports an exactly
    # zero effect rather than a bug. Caught exactly that way once, when a source
    # query was missing champion_id.
    dead = [c for c in feature_columns() if table[c].isna().all()]
    if dead:
        log.warning(
            "these feature columns are entirely NaN and carry no information: %s",
            ", ".join(dead),
        )

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
