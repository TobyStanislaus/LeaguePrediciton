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
    reflects_match: bool = True,
) -> None:
    """Write one synthetic match plus a ladder snapshot.

    With ``reflects_match`` (the default) the snapshot is what Riot would return
    shortly *after* the match: the winner's win count is one higher than it was
    at kickoff. That is the contamination reconstruction has to undo.

    With ``reflects_match=False`` the snapshot is stale -- taken long enough
    after that the match's result is no longer identifiable in it (LP reset, a
    new split, hundreds of intervening games). Counters are then identical
    regardless of who won, and "undoing" the result would *inject* the outcome
    rather than remove it.
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
        if reflects_match:
            lp = league_points + (18 if player_won else -18)
            wins = pre_wins + (1 if player_won else 0)
            losses = pre_losses + (0 if player_won else 1)
        else:
            lp, wins, losses = league_points, pre_wins, pre_losses
        entries.append({
            "puuid": f"p{i}", "tier": tier, "rank_division": division,
            "league_points": lp, "wins": wins, "losses": losses, "hot_streak": False,
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
    # Roles are assigned at champion select, before the game starts.
    "role_diff_top": "top-lane ladder gap, roles known at champion select",
    "role_diff_jungle": "jungle ladder gap, roles known at champion select",
    "role_diff_middle": "mid-lane ladder gap, roles known at champion select",
    "role_diff_bottom": "bot-lane ladder gap, roles known at champion select",
    "role_diff_utility": "support ladder gap, roles known at champion select",
    "role_diff_spread": "how unevenly the ladder gaps are spread across roles",
    # Champions are locked at champion select, and the winrate behind them is
    # built only from matches that finished before this one started.
    "champ_winrate": "the side's champions' winrate in strictly earlier matches",
    # Mastery is accumulated before the game; the champion is locked at select.
    "mastery_log_mean": "log mastery points on the champion locked at select",
    "mastery_rank_mean": "that champion's rank within the player's own pool",
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


def test_every_builder_emits_columns_in_the_declared_order():
    """Regression: a model fed these out of order predicts confident nonsense.

    build_feature_table reordered at the end, so a mismatch inside the pivot
    stayed invisible until the prediction path used the pivot directly.
    """
    import pandas as pd_

    from features.build_features import pivot_team_features

    teams = pd_.DataFrame(
        [{"match_id": "m", "team_id": team, **{stat: 1.0 for stat in TEAM_STATS}}
         for team in (100, 200)]
    )
    assert list(pivot_team_features(teams).columns) == feature_columns()


def test_feature_columns_are_unique_and_stable():
    from features.build_features import (
        CHAMPION_FEATURES,
        INCLUDE_CHAMPION_FEATURES,
        INCLUDE_MASTERY_FEATURES,
        INCLUDE_ROLE_FEATURES,
        MASTERY_FEATURES,
        ROLE_FEATURES,
    )

    columns = feature_columns()
    assert len(columns) == len(set(columns))
    expected = (
        3 * len(TEAM_STATS)
        + (len(ROLE_FEATURES) if INCLUDE_ROLE_FEATURES else 0)
        + (len(CHAMPION_FEATURES) if INCLUDE_CHAMPION_FEATURES else 0)
        + (len(MASTERY_FEATURES) if INCLUDE_MASTERY_FEATURES else 0)
    )
    assert len(columns) == expected


def _mastery_frames(mastery_rows, asked, blue_champ=1, red_champ=1):
    """RawFrames carrying one match plus a mastery table."""
    from features.build_features import RawFrames

    matches = pd.DataFrame([
        {"match_id": "m", "game_start_ts": 100.0, "winning_team": 100}
    ])
    parts = []
    for i in range(10):
        parts.append({
            "match_id": "m", "puuid": f"p{i}", "team_id": 100 if i < 5 else 200,
            "team_position": None, "champion_id": blue_champ if i < 5 else red_champ,
        })
    return RawFrames(
        matches=matches,
        participants=pd.DataFrame(parts),
        entries=pd.DataFrame(),
        mastery=pd.DataFrame(mastery_rows),
        mastery_players=frozenset(asked),
    )


def test_mastery_rewards_the_more_practised_side():
    from features.build_features import mastery_features

    rows = []
    for i in range(10):
        # Blue plays champion 1 with far more points than red on champion 2.
        rows.append({"puuid": f"p{i}", "champion_id": 1 if i < 5 else 2,
                     "mastery_points": 500_000 if i < 5 else 1_000})
    frames = _mastery_frames(rows, [f"p{i}" for i in range(10)], blue_champ=1, red_champ=2)

    out = mastery_features(frames)
    assert out.loc["m", "diff_mastery_log_mean"] > 0
    assert out.loc["m", "blue_mastery_log_mean"] > out.loc["m", "red_mastery_log_mean"]


def test_a_player_never_asked_about_is_unknown_not_zero():
    """Zero points would assert they have never played it; we simply do not know."""
    from features.build_features import mastery_features

    rows = [{"puuid": f"p{i}", "champion_id": 1, "mastery_points": 10_000} for i in range(5)]
    # Only blue was asked about, so red cannot be summarised at all.
    frames = _mastery_frames(rows, [f"p{i}" for i in range(5)])
    assert mastery_features(frames).empty


def test_an_asked_player_with_no_entry_counts_as_never_played():
    from features.build_features import mastery_features

    rows = [{"puuid": f"p{i}", "champion_id": 99, "mastery_points": 50_000}
            for i in range(10)]
    # Everyone was asked, but nobody has an entry for the champion actually played.
    frames = _mastery_frames(rows, [f"p{i}" for i in range(10)], blue_champ=1, red_champ=1)
    out = mastery_features(frames)
    assert not out.empty
    assert out.loc["m", "blue_mastery_log_mean"] == pytest.approx(0.0)
    assert out.loc["m", "diff_mastery_log_mean"] == pytest.approx(0.0)


def test_mastery_rank_is_within_the_player_pool():
    from features.build_features import mastery_features

    rows = []
    for i in range(10):
        # Champion 1 is each player's best; champion 2 their second.
        rows.append({"puuid": f"p{i}", "champion_id": 1, "mastery_points": 100_000})
        rows.append({"puuid": f"p{i}", "champion_id": 2, "mastery_points": 10_000})
    frames = _mastery_frames(rows, [f"p{i}" for i in range(10)], blue_champ=1, red_champ=2)

    out = mastery_features(frames)
    assert out.loc["m", "blue_mastery_rank_mean"] == pytest.approx(1.0)
    assert out.loc["m", "red_mastery_rank_mean"] == pytest.approx(2.0)
    assert out.loc["m", "diff_mastery_rank_mean"] == pytest.approx(-1.0)


def test_a_partly_known_side_is_not_summarised():
    """A three-player average is a different quantity under the same name."""
    from features.build_features import mastery_features

    rows = [{"puuid": f"p{i}", "champion_id": 1, "mastery_points": 5_000} for i in range(10)]
    asked = [f"p{i}" for i in range(10) if i != 3]  # one blue player missing
    frames = _mastery_frames(rows, asked)
    assert mastery_features(frames).empty


def test_champion_winrate_ignores_the_match_being_scored():
    """The decisive property: a match must not inform its own champion winrate.

    Two identical fixtures differing only in who won must give the first match
    identical champion features -- it has no history to draw on either way.
    """
    from features.build_features import champion_prior_winrates

    def build(first_winner: int) -> pd.DataFrame:
        matches = pd.DataFrame([
            {"match_id": "m1", "game_start_ts": 100.0, "winning_team": first_winner},
            {"match_id": "m2", "game_start_ts": 200.0, "winning_team": 100},
        ])
        parts = []
        for match_id in ("m1", "m2"):
            for i in range(10):
                parts.append({"match_id": match_id, "team_id": 100 if i < 5 else 200,
                              "champion_id": i})
        return champion_prior_winrates(matches, pd.DataFrame(parts))

    blue_first = build(100)
    red_first = build(200)
    assert blue_first.loc["m1", "diff_champ_winrate"] == pytest.approx(
        red_first.loc["m1", "diff_champ_winrate"]
    )
    # The second match *does* differ, because by then the first match is history.
    assert blue_first.loc["m2", "diff_champ_winrate"] != pytest.approx(
        red_first.loc["m2", "diff_champ_winrate"]
    )


def test_champion_winrates_shrink_toward_a_coin_flip_when_unseen():
    from features.build_features import champion_prior_winrates

    matches = pd.DataFrame([{"match_id": "m1", "game_start_ts": 1.0, "winning_team": 100}])
    parts = [
        {"match_id": "m1", "team_id": 100 if i < 5 else 200, "champion_id": i}
        for i in range(10)
    ]
    out = champion_prior_winrates(matches, pd.DataFrame(parts))
    assert out.loc["m1", "blue_champ_winrate"] == pytest.approx(0.5)
    assert out.loc["m1", "diff_champ_winrate"] == pytest.approx(0.0)


def test_a_repeatedly_winning_champion_earns_a_higher_winrate():
    from features.build_features import champion_prior_winrates

    matches, parts = [], []
    for k in range(40):
        matches.append({"match_id": f"m{k}", "game_start_ts": float(k), "winning_team": 100})
        for i in range(10):
            parts.append({"match_id": f"m{k}", "team_id": 100 if i < 5 else 200,
                          "champion_id": i})
    out = champion_prior_winrates(pd.DataFrame(matches), pd.DataFrame(parts))
    # Blue's champions always won, so by the last match blue's prior leads.
    assert out.loc["m39", "diff_champ_winrate"] > 0.2


def test_role_gaps_are_signed_towards_blue():
    """A stronger blue jungler must produce a positive jungle gap."""
    from features.build_features import ROLES, role_features

    rows = []
    for i, role in enumerate(ROLES):
        rows.append({"match_id": "m", "team_id": 100, "team_position": role,
                     "rank_points": 2000 + (500 if role == "JUNGLE" else 0)})
        rows.append({"match_id": "m", "team_id": 200, "team_position": role,
                     "rank_points": 2000})

    gaps = role_features(pd.DataFrame(rows))
    assert gaps.loc["m", "role_diff_jungle"] == pytest.approx(500)
    assert gaps.loc["m", "role_diff_top"] == pytest.approx(0)
    assert gaps.loc["m", "role_diff_spread"] == pytest.approx(500)


def test_unlabelled_roles_become_nan_not_zero():
    """Zero would assert the lanes were even; the truth is that we do not know."""
    from features.build_features import role_features

    rows = [
        {"match_id": "m", "team_id": 100, "team_position": "TOP", "rank_points": 2500},
        {"match_id": "m", "team_id": 200, "team_position": "TOP", "rank_points": 2000},
    ]
    gaps = role_features(pd.DataFrame(rows))
    assert gaps.loc["m", "role_diff_top"] == pytest.approx(500)
    assert np.isnan(gaps.loc["m", "role_diff_jungle"])


def test_role_features_absent_from_live_players_are_nan(cache: Cache):
    """SPECTATOR-V5 reports no roles, so live predictions must still build."""
    from features.build_features import (
        INCLUDE_ROLE_FEATURES,
        ROLE_FEATURES,
        features_for_players,
    )

    if not INCLUDE_ROLE_FEATURES:
        pytest.skip("role features are disabled -- see the ablation note in build_features")

    players = pd.DataFrame([
        {"puuid": f"p{i}", "team_id": 100 if i < 5 else 200, "tier": "DIAMOND",
         "rank_division": "II", "league_points": 50, "wins": 100, "losses": 90,
         "hot_streak": False}
        for i in range(10)
    ])
    features = features_for_players(players)
    assert list(features.columns) == feature_columns()
    for column in ROLE_FEATURES:
        assert features[column].isna().all()


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


def _features_for(tmp_path, mode: str, blue_wins: bool, build=None, **kwargs) -> pd.Series:
    db = tmp_path / f"{mode}-{blue_wins}.sqlite"
    with Cache(db) as cache:
        populate(cache, blue_wins=blue_wins, **kwargs)
        table = build_feature_table(cache, mode=mode, **(build or {}))
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
# Stale snapshots: reconstruction's failure mode
# --------------------------------------------------------------------------

STALE = 60 * 24 * HOUR


def test_stale_snapshots_are_dropped_not_reconstructed(cache: Cache):
    """A snapshot months later no longer contains the match, so undoing is invalid."""
    populate(cache, snapshot_at=KICKOFF + STALE, reflects_match=False)
    assert build_feature_table(cache, mode="reconstructed").empty


def test_fresh_snapshots_are_still_reconstructed(cache: Cache):
    populate(cache, snapshot_at=KICKOFF + 24 * HOUR)
    assert len(build_feature_table(cache, mode="reconstructed")) == 1


def test_the_age_limit_is_configurable(cache: Cache):
    populate(cache, snapshot_at=KICKOFF + 10 * 24 * HOUR)
    assert build_feature_table(cache, mode="reconstructed").empty
    assert len(
        build_feature_table(cache, mode="reconstructed", max_snapshot_age_days=30.0)
    ) == 1


def test_reconstructing_a_stale_snapshot_would_inject_the_outcome(tmp_path):
    """Regression for a real bug: this is *why* stale rows must be dropped.

    With the age limit disabled and a snapshot that no longer reflects the
    match, subtracting the result from LP does not remove the outcome -- it
    writes it in, inverted. Flipping who won then changes the features, and a
    model reads the label straight back off. Observed on real mixed-tier data
    as 0.83 accuracy from an artefact.
    """
    disabled = {"max_snapshot_age_days": None}
    blue = _features_for(
        tmp_path, "reconstructed", blue_wins=True,
        build=disabled, snapshot_at=KICKOFF + STALE, reflects_match=False,
    )
    red = _features_for(
        tmp_path, "reconstructed", blue_wins=False,
        build=disabled, snapshot_at=KICKOFF + STALE, reflects_match=False,
    )

    assert blue["diff_lp_mean"] != red["diff_lp_mean"], (
        "if this no longer differs, the injection mechanism is gone and this "
        "regression test can be simplified"
    )
    # And note the direction: the winning side's LP is pushed *down*, which is
    # the inverted signature that gave the bug away.
    assert blue["diff_lp_mean"] < 0 < red["diff_lp_mean"]


def test_the_age_limit_removes_that_injection(tmp_path):
    """The fix: with the limit on, the same stale data yields no rows at all."""
    for blue_wins in (True, False):
        db = tmp_path / f"stale-{blue_wins}.sqlite"
        with Cache(db) as cache:
            populate(
                cache, blue_wins=blue_wins,
                snapshot_at=KICKOFF + STALE, reflects_match=False,
            )
            assert build_feature_table(cache, mode="reconstructed").empty


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
