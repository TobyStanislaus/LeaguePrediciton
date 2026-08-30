"""Tests for the prediction path: live-game parsing, player gathering, persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.riot_client import RiotNotFoundError
from features.build_features import BLUE, RED, features_for_players, feature_columns
from models.evaluate import load_model, save_model
from models.predict import (
    describe_team,
    gather_players,
    ranked_entry,
    resolve_riot_id,
    teams_from_active_game,
)


class StubClient:
    """Enough of RiotClient for the prediction path, with no network."""

    def __init__(self, ranked: bool = True, missing: set[str] | None = None) -> None:
        self.platform = "euw1"
        self.region = "europe"
        self.ranked = ranked
        self.missing = missing or set()
        self.lookups: list[str] = []

    def get_league_entries_by_puuid(self, puuid: str):
        self.lookups.append(puuid)
        if puuid in self.missing:
            raise RiotNotFoundError("no entries")
        if not self.ranked:
            return [{"queueType": "RANKED_FLEX_SR", "tier": "GOLD", "rank": "I"}]
        return [
            {
                "queueType": "RANKED_SOLO_5x5", "puuid": puuid, "tier": "DIAMOND",
                "rank": "II", "leaguePoints": 55, "wins": 120, "losses": 100,
                "hotStreak": False,
            }
        ]

    def get_account_by_riot_id(self, game_name: str, tag_line: str):
        return {"puuid": f"puuid-of-{game_name}-{tag_line}", "gameName": game_name}


def active_game(queue_id: int = 420) -> dict:
    return {
        "gameId": 123456,
        "platformId": "EUW1",
        "gameQueueConfigId": queue_id,
        "gameLength": 720,
        "participants": [
            {"puuid": f"b{i}", "teamId": BLUE, "championId": i} for i in range(5)
        ] + [
            {"puuid": f"r{i}", "teamId": RED, "championId": 100 + i} for i in range(5)
        ],
    }


# --------------------------------------------------------------------------
# Live game parsing
# --------------------------------------------------------------------------


def test_active_game_splits_into_two_fives():
    teams = teams_from_active_game(active_game())
    assert teams.blue == ["b0", "b1", "b2", "b3", "b4"]
    assert teams.red == ["r0", "r1", "r2", "r3", "r4"]
    assert teams.hidden == 0


def test_active_game_with_no_participants_yields_empty_sides():
    teams = teams_from_active_game({"participants": []})
    assert teams.blue == [] and teams.red == []


def test_players_riot_hides_are_counted_not_dropped_silently():
    """Real behaviour: Riot returns puuid=null for some spectator participants,
    with a riotId that is only the champion name. Their rank is unknowable."""
    game = active_game()
    for participant in game["participants"][5:8]:
        participant["puuid"] = None
        participant["riotId"] = "Xayah"

    teams = teams_from_active_game(game)
    assert teams.blue == ["b0", "b1", "b2", "b3", "b4"]
    assert teams.red == ["r3", "r4"]
    assert teams.hidden_red == 3
    assert teams.hidden_blue == 0
    assert teams.hidden == 3


def test_empty_string_puuid_counts_as_hidden():
    game = active_game()
    game["participants"][0]["puuid"] = ""
    assert teams_from_active_game(game).hidden_blue == 1


def test_unknown_team_ids_are_ignored():
    game = active_game()
    game["participants"].append({"puuid": "spectator", "teamId": 300})
    teams = teams_from_active_game(game)
    assert len(teams.blue) == 5 and len(teams.red) == 5


def test_riot_id_is_split_on_the_last_hash():
    client = StubClient()
    assert resolve_riot_id(client, "Faker#KR1") == "puuid-of-Faker-KR1"
    # Names may themselves contain a hash; the tag is the final segment.
    assert resolve_riot_id(client, "od#d#EUW") == "puuid-of-od#d-EUW"


def test_riot_id_without_a_tag_is_rejected():
    with pytest.raises(ValueError, match="Name#TAG"):
        resolve_riot_id(StubClient(), "JustAName")


def test_riot_id_whitespace_is_trimmed():
    assert resolve_riot_id(StubClient(), " Faker # KR1 ") == "puuid-of-Faker-KR1"


# --------------------------------------------------------------------------
# Gathering ranked state
# --------------------------------------------------------------------------


def test_ranked_entry_picks_the_solo_queue_row():
    entry = ranked_entry(StubClient(), "p1")
    assert entry["tier"] == "DIAMOND"
    assert entry["league_points"] == 55


def test_a_player_with_only_flex_is_treated_as_unranked():
    """Flex rank is not solo-queue skill; guessing from it would be worse than NaN."""
    entry = ranked_entry(StubClient(ranked=False), "p1")
    assert entry["tier"] is None
    assert np.isnan(entry["league_points"])


def test_a_missing_player_does_not_raise():
    entry = ranked_entry(StubClient(missing={"ghost"}), "ghost")
    assert entry["tier"] is None


def test_gather_players_labels_both_sides():
    players = gather_players(StubClient(), [f"b{i}" for i in range(5)],
                             [f"r{i}" for i in range(5)])
    assert len(players) == 10
    assert (players["team_id"] == BLUE).sum() == 5
    assert (players["team_id"] == RED).sum() == 5


def test_gather_players_asks_about_each_player_once():
    client = StubClient()
    gather_players(client, [f"b{i}" for i in range(5)], [f"r{i}" for i in range(5)])
    assert len(client.lookups) == 10


# --------------------------------------------------------------------------
# Feature construction for an unplayed game
# --------------------------------------------------------------------------


def test_features_for_players_produces_exactly_the_model_columns():
    players = gather_players(StubClient(), [f"b{i}" for i in range(5)],
                             [f"r{i}" for i in range(5)])
    features = features_for_players(players)
    assert list(features.columns) == feature_columns()
    assert len(features) == 1


def test_features_for_players_rejects_a_missing_column():
    players = pd.DataFrame({"puuid": ["a"], "team_id": [BLUE]})
    with pytest.raises(ValueError, match="missing columns"):
        features_for_players(players)


def test_live_features_contain_no_label_or_metadata():
    """The prediction path must not require anything only a finished game has."""
    players = gather_players(StubClient(), [f"b{i}" for i in range(5)],
                             [f"r{i}" for i in range(5)])
    features = features_for_players(players)
    assert "blue_win" not in features.columns
    assert "game_start_ts" not in features.columns


def test_stronger_blue_team_moves_the_features_in_blue_favour():
    rows = []
    for i in range(5):
        rows.append({"puuid": f"b{i}", "team_id": BLUE, "tier": "MASTER",
                     "rank_division": "I", "league_points": 500, "wins": 200,
                     "losses": 100, "hot_streak": True})
    for i in range(5):
        rows.append({"puuid": f"r{i}", "team_id": RED, "tier": "GOLD",
                     "rank_division": "IV", "league_points": 10, "wins": 50,
                     "losses": 80, "hot_streak": False})
    features = features_for_players(pd.DataFrame(rows))
    assert features["diff_rank_points_mean"].iloc[0] > 0
    assert features["diff_winrate_mean"].iloc[0] > 0


def test_unranked_players_become_nan_not_zero():
    """Zero would read as Iron IV -- a confident wrong answer."""
    players = gather_players(StubClient(missing={"b0"}), [f"b{i}" for i in range(5)],
                             [f"r{i}" for i in range(5)])
    features = features_for_players(players, require_full_teams=False)
    assert not features.empty


def test_describe_team_handles_a_fully_unranked_side():
    players = gather_players(StubClient(ranked=False), [f"b{i}" for i in range(5)],
                             [f"r{i}" for i in range(5)])
    assert "no ranked players" in describe_team(players, BLUE)


# --------------------------------------------------------------------------
# Model persistence
# --------------------------------------------------------------------------


def _toy_model():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, len(feature_columns()))), columns=feature_columns())
    y = rng.integers(0, 2, size=60)
    model = Pipeline([("impute", SimpleImputer()), ("clf", LogisticRegression())])
    model.fit(X, y)
    return model


def test_saved_model_round_trips(tmp_path):
    path = save_model(_toy_model(), tmp_path / "m.joblib", {"name": "toy", "mode": "reconstructed"})
    model, meta = load_model(path)
    assert meta["name"] == "toy"
    assert meta["mode"] == "reconstructed"
    assert meta["feature_columns"] == feature_columns()
    assert "saved_at" in meta


def test_saved_model_can_predict_a_live_feature_row(tmp_path):
    path = save_model(_toy_model(), tmp_path / "m.joblib", {"name": "toy"})
    model, _ = load_model(path)

    players = gather_players(StubClient(), [f"b{i}" for i in range(5)],
                             [f"r{i}" for i in range(5)])
    features = features_for_players(players)
    probability = float(model.predict_proba(features)[0, 1])
    assert 0.0 <= probability <= 1.0


def test_loading_a_model_with_stale_feature_columns_is_refused(tmp_path):
    import joblib

    path = tmp_path / "stale.joblib"
    joblib.dump({"model": _toy_model(), "feature_columns": ["only_one"]}, path)
    with pytest.raises(ValueError, match="different feature columns"):
        load_model(path)


def test_missing_model_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_baseline"):
        load_model(tmp_path / "absent.joblib")
