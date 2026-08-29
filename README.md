# LeaguePrediciton
# "Vibe Coded" with Claude
Predict the winning team of a League of Legends match from **pre-game state only** —
team rank, LP, winrate, win streak and rank spread. A supervised binary
classification problem, in the spirit of the Titanic survival project.

No in-game statistics (KDA, gold, damage, objectives) are used as features: those
are only knowable after the match has started, and including them would leak the
outcome into a pre-game prediction task.

---

## Status

| Stage | State |
| --- | --- |
| 1. Scaffold | done |
| 2. API key handling + secret scanning | done |
| 3. Data collection (`data/riot_client.py`, `data/collect.py`, `data/cache.py`) | done |
| 4. Feature engineering (`features/build_features.py`) | done |
| 5. Modelling & evaluation (`models/`) | done |

---

## Setup

Requires Python 3.11+.

```bash
git clone <your-fork-url>
cd LeaguePrediciton

python -m venv .venv
source .venv/bin/activate      # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### API key

The Riot API key is read **only** from the `RIOT_API_KEY` environment variable,
loaded from a local `.env` via `python-dotenv`. It is never hardcoded in source,
notebooks, config or comments.

```bash
cp .env.example .env           # Windows PowerShell: Copy-Item .env.example .env
```

Then open `.env` and paste your key from
<https://developer.riotgames.com/>:

```
RIOT_API_KEY=<paste your key here>
```

`.env` is listed in `.gitignore` and must never be committed. Personal
development keys expire every 24 hours; regenerate and update `.env` when calls
start returning 403.

---

## Secret scanning

Run before every commit — it fails with exit code 1 if a key-shaped string, a
non-placeholder `RIOT_API_KEY=` assignment, `.env`, or a `data/cache/` file has
been staged:

```bash
python scripts/check_secrets.py
```

Scan every committable file instead of just the staged diff — tracked *and*
untracked, excluding anything gitignored:

```bash
python scripts/check_secrets.py --all
```

Untracked files matter here: a brand-new `.env.example` holding a real key is
invisible to `git ls-files` right up until it gets staged.

A line that must legitimately look like a key (the scanner's own test fixtures)
can be exempted with a trailing `# pragma: allowlist secret`.

Optionally wire it in as a pre-commit hook so it cannot be forgotten:

```bash
printf '#!/bin/sh\nexec python scripts/check_secrets.py\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```

---

## What is and isn't version-controlled

Committed: source, tests, `requirements.txt`, `.env.example`, `README.md`.

Ignored (`.gitignore`): `.env` and any other dotenv variant, `data/cache/`
(raw API responses — re-fetchable, bulky, and reveals nothing useful in
history), SQLite/Parquet files, derived feature tables, trained model
artefacts, virtualenvs, and the usual Python/editor/OS noise.

`data/cache/` is created at runtime by the collector; it is not tracked, so a
fresh clone will not contain it.

---

## Collecting data

```bash
python -m data.collect --platform euw1 --tiers DIAMOND --max-summoners 50 --matches-per-summoner 20
```

Three phases, all resumable — re-running the same command picks up where it
stopped, and anything already cached costs no rate limit:

1. **Seed** — pull a ranked ladder (apex tiers, or a tier plus divisions) and
   record every entry as a timestamped `league_snapshots` row.
2. **Matches** — for each seed player, list recent ranked match IDs (MATCH-V5)
   and fetch detail for the ones not already stored.
3. **Participants** — fetch LEAGUE-V4 entries for every player who appears in a
   collected match but has no rank yet. A team feature table needs rank for all
   ten players, not just the seed.

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--tiers CHALLENGER GRANDMASTER` | seed from apex tiers instead of a division |
| `--max-summoners N` | cap seed players (each costs 1 match-list call + detail calls) |
| `--max-participant-lookups N` | cap phase 3, the most call-hungry phase |
| `--forward-only` | only collect matches that started *after* the snapshot (see below) |
| `--skip-participants` | stop after phase 2 |
| `--db PATH` | alternative SQLite location |

Progress, API call counts, cache hits, retries and 429s are logged as it runs;
Ctrl-C is safe.

### Timing, and why it matters for leakage

LEAGUE-V4 returns a player's rank **right now**, not their rank when some past
match started. For a match played before the snapshot, the LP and win/loss
counts already include that match's own result — which leaks the outcome into a
supposedly pre-game feature.

Both timestamps are therefore stored (`league_entries.captured_at`,
`matches.game_start_ts`) so stage 4 can do a proper point-in-time join: use the
most recent snapshot captured *strictly before* kickoff.

* Default run — collects recent past matches. Usable immediately, but those rows
  need the point-in-time join to be trustworthy, and on a first run there is no
  earlier snapshot to join to.
* `--forward-only` — asks Riot only for matches started after the snapshot.
  Clean by construction, but returns almost nothing on a first run; the dataset
  fills up as you re-run over subsequent days.

The practical pattern is to run it once to seed, then re-run daily.

---

## Building features

```bash
python -m features.build_features --mode reconstructed --out data/processed/features.parquet
```

One row per match. Team-level aggregates of rank, LP, winrate, games played and
hot-streak count, for blue and red, plus their differences — 27 features, all
knowable before kickoff. Rank, division and LP collapse onto one monotone
ladder scale (`rank_points`), with Master/GM/Challenger sharing a continuous LP
pool the way Riot actually ranks them.

`--mode` picks how the ladder snapshot is joined to the match:

| Mode | Meaning |
| --- | --- |
| `point_in_time` | only snapshots captured strictly before kickoff. Correct by construction; empty until snapshots predate matches |
| `reconstructed` | start from the snapshot and undo what happened since kickoff. Wins/losses reconstruct **exactly**; LP approximately |
| `naive` | join the latest snapshot regardless of time. **Contaminated** — for demonstrating what leakage does, not for real results |

### How reconstruction avoids being circular

Wins and losses are integer counters. If a player won the match being
predicted, their pre-game win count was exactly one lower — so the label is used
only to reverse its own effect on a counter, recovering the true pre-game value.
That is reconstruction, not leakage.

LP cannot be recovered exactly (gains vary roughly 10–30 per game), so it is
adjusted by a fixed `--lp-delta` per undone match. That removes the systematic
part of the leak and leaves a small unbiased residual — which is why LP-derived
features are the first thing to drop if results still look too good.

The reconstruction is only as complete as the match history collected. A game in
the interval that was never collected stays baked into the counters, but its
result is uncorrelated with the match being predicted, so it is noise rather
than leakage.

Parquet is used when available and CSV otherwise; pyarrow's Parquet extension is
a compiled DLL that Windows Application Control policies sometimes block.

---

## Training and evaluation

```bash
python -m models.train_baseline --features data/processed/features.parquet
```

```bash
python -m models.train_boosted --features data/processed/features.parquet
```

Both are scored identically, so the comparison is fair:

* **Time-based split** — earliest matches train, latest test. A random shuffle
  would let the model see the future; rank and LP distributions drift across a
  season, so a shuffle overstates accuracy. A guard asserts the actual kickoff
  times on each side of the split, and fails if training data reaches into the
  test period.
* **Metrics** — accuracy, log loss, Brier score, ROC AUC, plus the
  always-predict-blue baseline that any useful model has to beat.
* **Calibration** — a quantile-binned reliability table and a plot in
  `artifacts/`. For a near-coin-flip problem, a model that says 58% and is right
  58% of the time is worth more than a confidently wrong one at equal accuracy.
* **Leakage sanity check** — rank/LP-only features should land around 55–65%.
  Anything at or above 70% is flagged, and 80%+ is reported as a suspected
  leakage bug rather than a good result. Both scripts exit non-zero above 80%,
  so this fails a CI run rather than quietly looking impressive.

### Daily forward-only run

```bash
python -m data.collect --platform euw1 --tiers CHALLENGER GRANDMASTER MASTER --max-summoners 40 --matches-per-summoner 15 --forward-only
```

Each run records a fresh ladder snapshot and collects matches played since the
previous one. Those rows have a snapshot that genuinely predates kickoff, so
`--mode point_in_time` starts producing data after the second run. Until then it
correctly returns nothing.

---

## Results so far

493 matches from the EUW apex ladders (Master/GM/Challenger), 369 train / 124
test on a chronological split.

| Mode | Model | Accuracy | Log loss | Brier | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| naive (leaky) | logistic | 0.605 | 0.633 | 0.220 | 0.705 |
| naive (leaky) | boosted | 0.621 | 0.647 | 0.228 | 0.697 |
| **reconstructed** | **logistic** | **0.581** | **0.677** | **0.242** | **0.627** |
| reconstructed | boosted | 0.597 | 0.685 | 0.245 | 0.644 |

Always-predict-blue scores 0.516, and a constant 0.5 prediction scores 0.693 on
log loss — both models beat both bars.

**The measured cost of leakage: +2.4 points of accuracy and +0.077 ROC AUC.**

The instructive part is that the leaky number is *not* absurd. Naive joining
gives 60.5%, comfortably inside the 55–65% band that looks reasonable for this
problem. Nothing about it screams "bug". Leakage here does not announce itself
as 95% accuracy; it quietly adds a couple of points and leaves a result you
would happily publish. Comparing modes is what makes it visible.

Two caveats on these numbers:

* **The sample is nearly all apex players** (400 Master, 397 Grandmaster, 264
  Challenger, 2 Diamond). Between-team rank spread there is only ~4.9% of the
  rank level, because matchmaking has already equalised the teams. A mixed-tier
  sample should have more to predict from.
* **Regularisation matters more than model choice.** An unregularised logistic
  fit scored 0.532 with log loss 0.711 — worse than a coin flip — on exactly
  the same features. Hence the cross-validated `C`.

---

## Tests

```bash
pytest
```

129 tests, none touching the network:

| File | Covers |
| --- | --- |
| `test_no_secrets.py` | key/cache commit guards, scanner behaviour |
| `test_cache.py` | SQLite cache, TTL, schema-level leakage guard |
| `test_riot_client.py` | rate limiting, retry/backoff, error surfacing, cache-first reads |
| `test_collect.py` | three-phase collection, resume, snapshot storage |
| `test_features.py` | rank encoding, join modes, reconstruction, **leakage** |
| `test_evaluate.py` | time split, metrics, calibration, sanity thresholds |

The central leakage test flips *only* the match outcome between two otherwise
identical synthetic worlds and asserts the reconstructed features come out
**identical** — if the features moved, they would be encoding the result. Its
partner test asserts that naive mode *does* move under the same flip, so the
first test cannot pass by being insensitive.

A further test enumerates every feature column and requires each to map to a
documented pre-game concept, so adding a new feature forces an explicit
justification that it is knowable before kickoff.

---

## Licence

See [LICENSE](LICENSE).
