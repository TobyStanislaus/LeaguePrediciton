"""Predict a game with a trained model.

Two ways in:

    # the game you are in right now
    python -m models.predict --riot-id "Faker#KR1" --platform kr

    # any ten players, blue first
    python -m models.predict --blue p1 p2 p3 p4 p5 --red p6 p7 p8 p9 p10

This is the one place where no leakage precautions are needed: the game has not
been played, so a player's current rank *is* their pre-game rank. That is the
whole point of restricting the feature set to pre-game state -- the model can
run on a game whose outcome does not exist yet.

What it cannot do is be confident. On rank/LP alone a well-behaved model sits
near 59% accuracy, so a single prediction is a lean, not a call. The probability
is the useful output: "61% blue" is a meaningful statement about a near-coin-flip
game in a way that a bare winner label is not.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import pandas as pd

from data.cache import Cache
from data.riot_client import (
    RiotAPIError,
    RiotAuthError,
    RiotClient,
    RiotNotFoundError,
    normalise_league_entry,
)
from features.build_features import (
    BLUE,
    RED,
    features_for_players,
    rank_points,
)
from models.evaluate import load_model

log = logging.getLogger(__name__)

RANKED_SOLO = "RANKED_SOLO_5x5"
RANKED_SOLO_QUEUE_ID = 420


# --------------------------------------------------------------------------
# Gathering pre-game state
# --------------------------------------------------------------------------


def ranked_entry(client: RiotClient, puuid: str, queue: str = RANKED_SOLO) -> dict[str, Any]:
    """Current ranked entry for one player, or an unranked placeholder.

    An unranked player is left as NaN rather than guessed at; the model's
    imputer fills it with the median, and the caller is told how many.
    """
    blank = {
        "puuid": puuid, "tier": None, "rank_division": None, "league_points": float("nan"),
        "wins": float("nan"), "losses": float("nan"), "hot_streak": False,
    }
    try:
        entries = client.get_league_entries_by_puuid(puuid)
    except RiotNotFoundError:
        return blank

    for raw in entries:
        if raw.get("queueType") == queue:
            entry = normalise_league_entry(raw)
            entry["puuid"] = entry.get("puuid") or puuid
            return entry
    return blank


def gather_players(
    client: RiotClient, blue: Sequence[str], red: Sequence[str]
) -> pd.DataFrame:
    """Build the ten-row player frame the feature builder expects."""
    rows = []
    for team_id, puuids in ((BLUE, blue), (RED, red)):
        for puuid in puuids:
            entry = ranked_entry(client, puuid)
            entry["team_id"] = team_id
            rows.append(entry)
    return pd.DataFrame(rows)


def resolve_riot_id(client: RiotClient, riot_id: str) -> str:
    """Turn "Name#TAG" into a PUUID."""
    if "#" not in riot_id:
        raise ValueError(
            f"{riot_id!r} is not a Riot ID. Use the Name#TAG form, e.g. \"Faker#KR1\"."
        )
    game_name, _, tag_line = riot_id.rpartition("#")
    account = client.get_account_by_riot_id(game_name.strip(), tag_line.strip())
    return account["puuid"]


@dataclass
class LiveTeams:
    """Who we can actually look up in a live game.

    Riot withholds the PUUID of some spectator participants -- they come back
    as ``null`` with a placeholder ``riotId`` that is just the champion name.
    Those players cannot be rank-checked at all, so they are counted as hidden
    rather than quietly dropped: a prediction built on three of five players is
    much weaker than one built on five, and the caller must be told.
    """

    blue: list[str]
    red: list[str]
    hidden_blue: int
    hidden_red: int

    @property
    def hidden(self) -> int:
        return self.hidden_blue + self.hidden_red


def teams_from_active_game(game: dict[str, Any]) -> LiveTeams:
    """Split a SPECTATOR-V5 payload into blue and red PUUIDs, counting hidden ones."""
    sides: dict[int, list[str]] = {BLUE: [], RED: []}
    hidden = {BLUE: 0, RED: 0}

    for participant in game.get("participants", []):
        team = participant.get("teamId")
        if team not in sides:
            continue
        puuid = participant.get("puuid")
        if puuid:
            sides[team].append(puuid)
        else:
            hidden[team] += 1

    return LiveTeams(sides[BLUE], sides[RED], hidden[BLUE], hidden[RED])


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def describe_team(players: pd.DataFrame, team_id: int) -> str:
    side = players[players["team_id"] == team_id]
    ranked = side[side["tier"].notna()]
    if ranked.empty:
        return "no ranked players"

    points = [
        rank_points(t, d, lp)
        for t, d, lp in zip(ranked["tier"], ranked["rank_division"], ranked["league_points"])
    ]
    games = ranked["wins"] + ranked["losses"]
    winrate = (ranked["wins"] / games.replace(0, pd.NA)).astype(float).mean()
    tiers = ", ".join(sorted({str(t).title() for t in ranked["tier"]}))
    return (
        f"{len(ranked)}/5 ranked | ladder points {pd.Series(points).mean():.0f} "
        f"| winrate {winrate:.1%} | {tiers}"
    )


def report_prediction(players: pd.DataFrame, probability: float, meta: dict[str, Any]) -> None:
    print("\n  blue:", describe_team(players, BLUE))
    print("  red: ", describe_team(players, RED))

    favoured, chance = ("BLUE", probability) if probability >= 0.5 else ("RED", 1 - probability)
    print(f"\n  prediction: {favoured} {chance:.1%}   (blue {probability:.1%} / "
          f"red {1 - probability:.1%})")

    reported = (meta.get("metrics") or {}).get("accuracy")
    if reported:
        print(
            f"\n  the model this came from scored {float(reported):.1%} accuracy on held-out "
            "matches,\n  so treat this as a lean rather than a call -- the probability is the "
            "useful part."
        )
    if abs(probability - 0.5) < 0.03:
        print("  this one is a coin flip; the model has no real opinion.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="artifacts/model_baseline.joblib")
    parser.add_argument("--platform", default="euw1")
    parser.add_argument("--riot-id", help='predict the live game of "Name#TAG"')
    parser.add_argument("--blue", nargs=5, metavar="PUUID", help="five blue-side PUUIDs")
    parser.add_argument("--red", nargs=5, metavar="PUUID", help="five red-side PUUIDs")
    parser.add_argument("--db", default="data/cache/riot.sqlite")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)-7s %(name)s | %(message)s",
    )

    if not args.riot_id and not (args.blue and args.red):
        parser.error("give either --riot-id, or both --blue and --red")

    model, meta = load_model(args.model)

    cache = Cache(args.db)
    try:
        client = RiotClient(platform=args.platform, cache=cache)
    except RiotAuthError as exc:
        print(f"error: {exc}")
        cache.close()
        return 2

    try:
        if args.riot_id:
            puuid = resolve_riot_id(client, args.riot_id)
            try:
                game = client.get_active_game(puuid)
            except RiotNotFoundError:
                print(f"{args.riot_id} is not in a game right now.")
                return 1

            queue_id = game.get("gameQueueConfigId")
            print(f"live game {game.get('gameId')} on {game.get('platformId')} "
                  f"(queue {queue_id}, {int(game.get('gameLength', 0) // 60)}m in)")
            if queue_id != RANKED_SOLO_QUEUE_ID:
                print(f"  [!] this is queue {queue_id}, not ranked solo/duo ({RANKED_SOLO_QUEUE_ID}). "
                      "The model was trained on ranked solo only, so this prediction is "
                      "out of distribution.")
            teams = teams_from_active_game(game)
            blue, red = teams.blue, teams.red
            if teams.hidden:
                print(
                    f"  [!] Riot withheld the identity of {teams.hidden} of 10 players "
                    f"(blue {teams.hidden_blue}, red {teams.hidden_red}). Their rank cannot "
                    "be looked up, so this prediction rests on the rest."
                )
            if not blue or not red:
                print("error: one side is entirely hidden; nothing to predict from")
                return 1
        else:
            blue, red = list(args.blue), list(args.red)
            if len(blue) != 5 or len(red) != 5:
                print(f"error: expected 5 players per side, got {len(blue)} and {len(red)}")
                return 1

        players = gather_players(client, blue, red)
        unranked = int(players["tier"].isna().sum())
        if unranked:
            print(f"  [!] {unranked} of 10 players have no ranked entry; "
                  "their values are imputed and the prediction is weaker for it.")

        features = features_for_players(players, require_full_teams=False)
        if features.empty:
            print("error: could not build features for this game")
            return 1

        probability = float(model.predict_proba(features)[0, 1])
        report_prediction(players, probability, meta)

    except RiotAuthError as exc:
        print(f"error: {exc}")
        return 2
    except RiotAPIError as exc:
        print(f"error: {exc}")
        return 1
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
