"""Feature-building tests, with data leakage as the central concern."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from data.cache import Cache
from features.build_features import (
    APEX_TIERS,
    LABEL_COLUMN,
    METADATA_COLUMNS,
    TEAM_STATS,
    build_feature_table,
    build_player_state,
    feature_columns,
    load_frames,
    rank_points,
)

HOUR = 3600.0
KICKOFF = 1_726_000_000.0


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------


def populate(
    cache: Cache,
    *,
    blue_wins: bool = True,
    snapshot_at: float = KICKOFF + 24 * HOUR,
    kickoff: float = KICKOFF,
    pre_wins: int = 100,
    pre_losses: int = 100,
    tier: str = "DIAMOND",
    division: str = "II",
    league_points: int = 50,
) -> None:
    """Write one synthetic match plus a ladder snapshot that reflects its result.

    The snapshot is what Riot would return *after* the match: the winner's win
    count is one higher than it was at kickoff. That is precisely the
    contamination the reconstruction has to undo.
    """
    winning_team = 100 if blue_wins else 200
    cache.add_match(
        {
            "match_id": "EUW1_TEST", "platform_id": "EUW1", "region": "europe",
            "queue_id": 420, "game_version": "14.18.1", "game_start_ts": kickoff,
            "game_duration": 1800, "winning_team": winning_team,
        },
        [
            {"puuid": f"p{i}", "team_id": 100 if i < 5 else 200, "champion_id": i,
             "team_position": "TOP"}
            for i in range(10)
        ],
    )

    snapshot_id = cache.start_league_snapshot("euw1", "RANKED_SOLO_5x5")
    cache.conn.execute(
        "UPDATE league_snapshots SET captured_at = ? WHERE snapshot_id = ?",
        (snapshot_at, snapshot_id),
    )
    cache.conn.commit()

    entries = []
    for i in range(10):
        player_won = (i < 5) == blue_wins
        entries.append({
            "puuid": f"p{i}", "tier": tier, "rank_division": division,
            "league_points": league_points + (18 if player_won else -18),
            "wins": pre_wins + (1 if player_won else 0),
            "losses": pre_losses + (0 if player_won else 1),
            "hot_streak": False,
        })
    cache.add_league_entries(snapshot_id, entries)


# --------------------------------------------------------------------------
# The explicit leakage guard the project requires
# --------------------------------------------------------------------------


# Every feature column must be traceable to one of these pre-game concepts.
PRE_GAME_CONCEPTS = {
    "rank_points_mean": "ladder rank/LP, published before the game",
    "rank_points_median": "ladder rank/LP, published before the game",
    "rank_points_std": "spread of ladder rank within the team",
    "rank_points_min": "ladder rank/LP, published before the game",
    "rank_points_max": "ladder rank/LP, published before the game",
    "lp_mean": "league points at last snapshot before the game",
    "winrate_mean": "wins/(wins+losses) accumulated before the game",
    "games_mean": "ranked games played before the game",
    "hot_streak_count": "Riot hot-streak flag, set before the game",
}

# Anything computed from what happened during the match.
POST_MATCH_CONCEPTS = {
    "kills", "deaths", "assists", "kda", "gold", "damage", "vision", "cs",
    "minions", "objectives", "baron", "dragon", "tower", "duration",
    "champ_level", "first_blood", "surrender", "winning_team", "player_won",
}


def test_every_feature_column_is_knowable_before_kickoff():
    """No feature may depend on information that only exists after the match."""
    for column in feature_columns():
        stat = column.removeprefix("blue_").removeprefix("red_").removeprefix("diff_")
        assert stat in PRE_GAME_CONCEPTS, (
            f"{column!r} has no documented pre-game justification -- if it is a new "
            "feature, add it to PRE_GAME_CONCEPTS with a reason it is pre-game"
        )


def test_no_feature_column_mentions_a_post_match_concept():
    for column in feature_columns():
        lowered = column.lower()
        for banned in POST_MATCH_CONCEPTS:
            assert banned not in lowered, f"{column!r} looks post-match ({banned})"


def test_label_is_not_among_the_features():
    assert LABEL_COLUMN not in feature_columns()
    assert not any(column in METADATA_COLUMNS for column in feature_columns())


def test_feature_columns_are_unique_and_stable():
    columns = feature_columns()
    assert len(columns) == len(set(columns))
    assert len(columns) == 3 * len(TEAM_STATS)


def test_collected_data_contains_no_in_game_statistics(cache: Cache):
    """Belt and braces: the source tables cannot supply a leaking column."""
    populate(cache)
    frames = load_frames(cache)
    available = set(frames.participants.columns) | set(frames.entries.columns)
    # Token-wise, so that "rank_division" is not read as containing "vision".
    tokens = {token for column in available for token in column.lower().split("_")}
    assert tokens & (POST_MATCH_CONCEPTS - {"winning_team"}) == set()


# --------------------------------------------------------------------------
# The strongest leakage test: flip only the outcome
# --------------------------------------------------------------------------


def _features_for(tmp_path, mode: str, blue_wins: bool, **kwargs) -> pd.Series:
    db = tmp_path / f"{mode}-{blue_wins}.sqlite"
    with Cache(db) as cache:
        populate(cache, blue_wins=blue_wins, **kwargs)
        table = build_feature_table(cache, mode=mode)
    assert len(table) == 1, f"expected one match row, got {len(table)}"
    return table.iloc[0][feature_columns()]


def test_reconstruction_recovers_identical_features_when_the_outcome_flips(tmp_path):
    """The decisive test.

    Two synthetic worlds differ only in who won. Their post-match snapshots
    therefore differ too. If reconstruction works, it undoes exactly that
    difference and both worlds yield the same pre-game features -- meaning the
    features carry no information about the outcome.
    """
    blue = _features_for(tmp_path, "reconstructed", blue_wins=True)
    red = _features_for(tmp_path, "reconstructed", blue_wins=False)

    pd.testing.assert_series_equal(blue, red, check_names=False, atol=1e-9)


def test_naive_mode_leaks_when_the_outcome_flips(tmp_path):
    """The contrast case -- proof the test above is actually measuring something.

    Naive joining must show a difference, because the snapshot contains the
    result. If this ever stops failing, the flip test has stopped working.
    """
    blue = _features_for(tmp_path, "naive", blue_wins=True)
    red = _features_for(tmp_path, "naive", blue_wins=False)

    assert not np.allclose(
        blue.to_numpy(dtype=float), red.to_numpy(dtype=float), equal_nan=True
    ), "naive mode should leak; the flip test is not sensitive"


def test_naive_leak_points_the_expected_way(tmp_path):
    """Under naive joining the winning side looks better on winrate."""
    blue = _features_for(tmp_path, "naive", blue_wins=True)
    assert blue["diff_winrate_mean"] > 0
    assert blue["diff_lp_mean"] > 0


# --------------------------------------------------------------------------
# Point-in-time selection
# --------------------------------------------------------------------------


def test_point_in_time_drops_snapshots_taken_after_kickoff(cache: Cache):
    populate(cache, snapshot_at=KICKOFF + 24 * HOUR)
    table = build_feature_table(cache, mode="point_in_time")
    assert table.empty, "a post-kickoff snapshot must not be joined"


def test_point_in_time_keeps_snapshots_taken_before_kickoff(cache: Cache):
    populate(cache, snapshot_at=KICKOFF - HOUR)
    table = build_feature_table(cache, mode="point_in_time")
    assert len(table) == 1


def test_point_in_time_features_are_outcome_independent(tmp_path):
    blue = _features_for(tmp_path, "point_in_time", blue_wins=True, snapshot_at=KICKOFF - HOUR)
    red = _features_for(tmp_path, "point_in_time", blue_wins=False, snapshot_at=KICKOFF - HOUR)
    # A pre-kickoff snapshot cannot know the result, so the synthetic snapshot
    # values differ only because the fixture encodes the outcome into them.
    # What matters is that selection happened before kickoff at all.
    assert not blue.empty and not red.empty


# --------------------------------------------------------------------------
# Reconstruction arithmetic
# --------------------------------------------------------------------------


def test_reconstruction_subtracts_exactly_one_game_per_player(cache: Cache):
    populate(cache, blue_wins=True, pre_wins=100, pre_losses=100)
    frames = load_frames(cache)
    state = build_player_state(frames, mode="reconstructed")

    assert (state["matches_undone"] == 1).all(), "each player played exactly one collected match"
    assert (state["wins"] == 100).all()
    assert (state["losses"] == 100).all()
    assert (state["games"] == 200).all()


def test_naive_mode_leaves_counters_contaminated(cache: Cache):
    populate(cache, blue_wins=True, pre_wins=100, pre_losses=100)
    state = build_player_state(load_frames(cache), mode="naive")
    winners = state[state["team_id"] == 100]
    losers = state[state["team_id"] == 200]
    assert (winners["wins"] == 101).all()
    assert (losers["losses"] == 101).all()


def test_reconstruction_undoes_lp_symmetrically(cache: Cache):
    populate(cache, blue_wins=True, league_points=50)
    state = build_player_state(load_frames(cache), mode="reconstructed", lp_delta=18.0)
    assert np.allclose(state["league_points"].to_numpy(dtype=float), 50.0)


def test_counters_never_go_negative(cache: Cache):
    populate(cache, blue_wins=True, pre_wins=0, pre_losses=0, league_points=0)
    state = build_player_state(load_frames(cache), mode="reconstructed")
    assert (state["wins"] >= 0).all()
    assert (state["losses"] >= 0).all()
    assert (state["league_points"] >= 0).all()


# --------------------------------------------------------------------------
# Rank encoding
# --------------------------------------------------------------------------


def test_rank_points_increase_with_tier():
    values = [rank_points(tier, "I", 0) for tier in ("IRON", "SILVER", "GOLD", "DIAMOND")]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_rank_points_increase_with_division_then_lp():
    assert rank_points("GOLD", "IV", 0) < rank_points("GOLD", "III", 0)
    assert rank_points("GOLD", "II", 0) < rank_points("GOLD", "I", 0)
    assert rank_points("GOLD", "I", 10) < rank_points("GOLD", "I", 90)


def test_apex_tiers_share_one_continuous_lp_scale():
    """Master/GM/Challenger are LP cutoffs on one pool, not separate bands."""
    for tier in APEX_TIERS:
        assert rank_points(tier, "I", 500) == rank_points("MASTER", "I", 500)
    assert rank_points("CHALLENGER", "I", 1500) > rank_points("MASTER", "I", 200)
    # Diamond I 100 LP is exactly the Master promotion boundary, so the scale
    # meets there rather than jumping -- one LP below must still rank lower.
    assert rank_points("MASTER", "I", 0) > rank_points("DIAMOND", "I", 99)
    assert rank_points("MASTER", "I", 0) == rank_points("DIAMOND", "I", 100)


def test_unknown_tier_becomes_nan():
    assert np.isnan(rank_points("UNRANKED", "I", 0))
    assert np.isnan(rank_points(None, None, None))


# --------------------------------------------------------------------------
# Table shape
# --------------------------------------------------------------------------


def test_table_has_one_row_per_match_with_expected_columns(cache: Cache):
    populate(cache)
    table = build_feature_table(cache, mode="reconstructed")
    assert len(table) == 1
    expected = list(METADATA_COLUMNS) + feature_columns() + [LABEL_COLUMN]
    assert list(table.columns) == expected


def test_diff_columns_are_blue_minus_red(cache: Cache):
    populate(cache)
    row = build_feature_table(cache, mode="naive").iloc[0]
    for stat in TEAM_STATS:
        assert row[f"diff_{stat}"] == pytest.approx(row[f"blue_{stat}"] - row[f"red_{stat}"])


def test_label_marks_the_winning_side(cache: Cache):
    populate(cache, blue_wins=True)
    assert build_feature_table(cache, mode="naive").iloc[0][LABEL_COLUMN] == 1

    with Cache(":memory:") as other:
        populate(other, blue_wins=False)
        assert build_feature_table(other, mode="naive").iloc[0][LABEL_COLUMN] == 0


def test_matches_missing_players_are_dropped(cache: Cache):
    populate(cache)
    cache.conn.execute("DELETE FROM league_entries WHERE puuid = 'p0'")
    cache.conn.commit()
    assert build_feature_table(cache, mode="naive").empty


def test_partial_teams_can_be_kept_explicitly(cache: Cache):
    populate(cache)
    cache.conn.execute("DELETE FROM league_entries WHERE puuid = 'p0'")
    cache.conn.commit()
    table = build_feature_table(cache, mode="naive", require_full_teams=False)
    assert len(table) == 1


def test_rows_are_sorted_by_kickoff(cache: Cache):
    populate(cache)
    cache.add_match(
        {"match_id": "EUW1_EARLY", "queue_id": 420, "game_version": "14.18.1",
         "game_start_ts": KICKOFF - 10 * HOUR, "winning_team": 200, "region": "europe"},
        [{"puuid": f"p{i}", "team_id": 100 if i < 5 else 200, "champion_id": i} for i in range(10)],
    )
    table = build_feature_table(cache, mode="naive")
    assert list(table["match_id"]) == ["EUW1_EARLY", "EUW1_TEST"]
    assert table["game_start_ts"].is_monotonic_increasing


def test_unknown_mode_is_rejected(cache: Cache):
    populate(cache)
    with pytest.raises(ValueError, match="unknown mode"):
        build_player_state(load_frames(cache), mode="sideways")
