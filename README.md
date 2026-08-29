# LeaguePrediciton

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
| 4. Feature engineering (`features/build_features.py`) | not started |
| 5. Modelling & evaluation (`models/`) | not started |

Placeholder modules exist for stages 4–5 so the package layout is stable; they
contain docstrings only.

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

## Tests

```bash
pytest
```

Covers the secret and cache-file guards (`tests/test_no_secrets.py`), the SQLite
cache including its TTL and schema-level leakage guard (`tests/test_cache.py`),
and the API client's rate limiting, retry/backoff, error surfacing and
cache-first reads (`tests/test_riot_client.py`). No test touches the network.

Feature-leakage and evaluation tests arrive with stages 4 and 5.

---

## Licence

See [LICENSE](LICENSE).
