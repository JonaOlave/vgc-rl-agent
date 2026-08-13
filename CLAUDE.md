# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reinforcement learning agent for Pokémon VGC doubles battles: PPO (Stable-Baselines3) with an optional
behavioral-cloning warm-start from human replay logs, trained against a **local Pokémon Showdown server**
via `poke-env`. A Vue 3 + FastAPI dashboard visualizes training/evaluation results on top of it.

Code comments and docstrings in `pokemon_rl/vgc/` and `pokemon_rl/imitation/` are written in Spanish; match
that convention when editing those files.

## Setup & running

> **Where things actually run**: in practice, the working venv, trained checkpoints (`models/vgc/`), checkpoint
> backups (`models/vgc_history/`), and evaluation data (`data/`) live in a sibling directory,
> `../pokemon-showdown-client/`, whose `pokemon_rl/` is an **untracked, manually-synced copy** of this repo's
> `pokemon_rl/` — not this repo itself. After editing code here, `cp` the changed file(s) into
> `../pokemon-showdown-client/pokemon_rl/...` before training will see the change. The actual PS **server**
> (distinct from `pokemon-showdown-client`, which is only the web client) is another sibling, `../pokemon-showdown/`.
> `gen9championsvgc2026regmb` is a fan-made Showdown mod (`data/mods/champions/` in that server repo), not the
> real official VGC regulation — it has its own species/item whitelist and a 32-point-per-stat/66-total EV
> system (not standard 0–252/510), and locks IVs at 31. Validate any new team with
> `node pokemon-showdown validate-team gen9championsvgc2026regmb < team.txt` (run from the server repo) before
> adding it to `vgc/team.py`.

Requires Python 3.11+, Node.js v20+, and a local Pokémon Showdown server:

```bash
python -m venv pokemon_rl/.venv
source pokemon_rl/.venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-api.txt
```

The PS server must be running before any training/evaluation/battle command (all of them connect over
websocket to `localhost:8000`):

```bash
cd /path/to/pokemon-showdown
node pokemon-showdown start --no-security --port 8000
```

### Training pipeline (run in order)

```bash
# 1. Behavioral cloning from downloaded human replays
python -m pokemon_rl.imitation.train_bc --download --n-download 500 \
    --format gen9vgc2025regg --epochs 30 --save-path models/bc_pretrained

# 2. PPO fine-tuning, optionally warm-started from the BC checkpoint
python -m pokemon_rl.vgc.train --format gen9championsvgc2026regmb \
    --champions-team --opponent random --bc-model models/bc_pretrained --timesteps 500000

# 3. Evaluate a trained checkpoint
python -m pokemon_rl.vgc.evaluate --champions-team --opponent random --n-battles 50
```

`vgc/train.py` and `vgc/evaluate.py` both take `--vgc-team` (VGC 2025 Reg G) or `--champions-team`
(Champions VGC 2026 Reg M-B) to select a sample team from `pokemon_rl/vgc/team.py`; omitting both trains/
evaluates on `gen9randomdoublesbattle`, which needs no team. `--opponent` is `random` or `heuristic`.

Checkpoints land in `models/vgc/` as `vgc_ppo_<steps>_steps.zip` plus a final `final.zip`; these paths are
read directly by the API's `CheckpointRepository` and `EvaluationRunner`, so keep training output there
unless the repositories are also updated.

### Dashboard (backend + frontend, two terminals)

```bash
# Backend — serves checkpoints/results, can launch evaluations on demand
source pokemon_rl/.venv/bin/activate
uvicorn pokemon_rl.api.main:app --port 8080 --reload

# Frontend
cd dashboard
npm install   # first time only
npm run dev   # http://localhost:5173, proxies /api/* to localhost:8080 (see vite.config.ts)
```

Dashboard build/typecheck: `npm run build` (runs `vue-tsc` then `vite build`) from `dashboard/`. There is
no lint script and no test suite (Python or JS) in this repo currently.

## Architecture

### Two generations of Python code — use `pokemon_rl/vgc/`, not the top-level singles files

`pokemon_rl/env.py`, `train.py`, `player.py`, `test_battle.py` are an earlier **singles**-format
implementation and are not part of the active pipeline. `pokemon_rl/constants.py` and `pokemon_rl/embedding.py`
are the exception — they hold the base per-Pokémon/per-move embedding logic and are imported and extended by
the doubles code. All active development happens in `pokemon_rl/vgc/`:

- `vgc/constants.py` — re-exports the base enums/dims from `pokemon_rl/constants.py` and adds doubles-specific
  layout constants (2 active slots/side, the 925-float observation size, the 107×107 action space).
- `vgc/embedding.py` — builds the 925-dim observation vector; wraps `pokemon_rl/embedding.py`'s
  `_embed_pokemon`/`_embed_move` helpers for both active slots and the bench.
- `vgc/env.py` — `VGCEnv(DoublesEnv)` (poke-env). Implements `embed_battle` and `calc_reward`, and overrides
  `action_to_order` for two known PPO-ignores-the-action-mask failure modes: (1) bypasses the RL agent's
  action entirely during forced-switch turns, picking a random valid combination directly; (2)
  `_normalize_targets` fixes moves that don't target a specific Pokémon (Trick Room, Protect, screens,
  Substitute, etc.) — the model regularly samples a bogus target for these, which used to silently invalidate
  the whole order and fall back to a random move even when the move-index choice was "correct". Both are
  commented in place — read them before touching action decoding.
  `make_vgc_env(...)` wraps this in poke-env's `SingleAgentWrapper` against a `RandomPlayer` or
  `SimpleHeuristicsPlayer` opponent, both connecting to `localhost:8000`. It also accepts
  `opponent_team_pool: list[str]` — when given, the **opponent only** picks a random team per battle from the
  pool (see `vgc/teambuilder.py`'s `RandomTeamPool`, a poke-env `Teambuilder`). This requires patching
  `agent2._team` after `super().__init__()` inside `VGCEnv.__init__`, because `PokeEnv` (poke-env's base env)
  constructs both internal battling agents (`agent1` = us, `agent2` = opponent) from the *same* `team` kwarg —
  passing a team only to the standalone `opponent` Player object passed to `SingleAgentWrapper` does **nothing**,
  since that object is only used for its `choose_move`/`teampreview` logic, not for team selection.
- `vgc/team.py` — Showdown-format team strings. `SAMPLE_TEAM_CHAMPIONS_REGMB` is our own fixed team (used by
  `--champions-team`); `SAMPLE_TEAM_TRICKROOM` / `_TAILWIND` / `_RAIN` / `_SAND` / `_SUN` are opponent-pool
  archetypes, collected in `OPPONENT_TEAM_POOL_CHAMPIONS_REGMB`. `SAMPLE_TEAM_NO_RESTRICTED` is **not legal**
  for `gen9championsvgc2026regmb` (kept only as a negative example — see the note above it) — don't add it to
  the pool without fixing it first.
- `vgc/train.py` / `vgc/evaluate.py` — CLI entry points described above. `train.py` also takes
  `--vary-opponent-team` to enable the opponent team pool above. `evaluate()` also accepts an
  `opponent_team_pool` param (not yet exposed as a CLI flag) — pass a single-element list to pin the
  opponent to one archetype for a whole eval run, which is how the per-matchup win-rate table below is
  produced (see `eval_round6_matchups.py` in `pokemon-showdown-client/`, not committed to this repo).
- `vgc/damage_calc.py` — `estimate_damage(battle, attacker, defender, move)`, a thin wrapper around
  poke-env's own bundled Gen9 damage calculator (`poke_env.calc.damage_calc_gen9`, a Python port of Smogon's
  calc) returning `{pct_min, pct_max, possible_ko, guaranteed_ko}` instead of a raw HP range. Needs
  `defender.stats` populated, which poke-env only knows automatically for **our own** team — for a genuinely
  unknown live opponent it raises `AssertionError('defender stats not defined')` until enough is revealed
  (no default-EV-spread fallback exists yet). For self-diagnostic use against a known pinned opponent
  archetype, pre-populate stats via `Teambuilder.parse_showdown_team()` + `compute_raw_stats()` (see
  `populate_known_stats()` in `pokemon-showdown-client/diag_trickroom_round6.py`, not committed to this repo).

`pokemon_rl/imitation/` (downloader → parser → dataset → train_bc) is the BC pipeline: it downloads replays
from the PS API, parses them into `(observation, action)` pairs using `vgc/embedding.py`'s `embed_battle`
(so BC and PPO observations stay identical), and produces an SB3-compatible policy checkpoint that
`vgc/train.py --bc-model` can load and continue training with PPO.

### Observation / action space (must stay in sync across embedding, env, and any loaded checkpoint)

- Observation: `Box(925,)` float32 — active Pokémon ×2, bench ×4, mirrored for the opponent, plus field/weather.
  Exact layout and byte budget are documented in `vgc/constants.py`.
- Action: `MultiDiscrete([107, 107])`, one slot per active Pokémon: 0 = pass, 1–6 = switch, 7–106 = move ×
  target × gimmick (mega/Z/dynamax/tera). Decoding logic is commented in `vgc/constants.py`.

A checkpoint trained against one observation/action layout is not loadable against a changed one — if you
touch `vgc/embedding.py` or `vgc/constants.py`, treat existing `models/vgc/*.zip` checkpoints as invalidated.

### API — DDD layering (`pokemon_rl/api/`)

`domain/` (entities: `Checkpoint`, `Evaluation`, `BattleResult`) → `application/` (queries + the
`RunEvaluationCommand` use case) → `infrastructure/` (repositories) → `presentation/` (FastAPI routers),
wired together in `main.py`. Two verticals, `training/` and `evaluation/`, each with their own slice through
all four layers — follow the existing vertical's shape when adding a new one rather than introducing a
different pattern.

Storage is filesystem-based, not a database:
- `CheckpointRepository` globs `models/vgc/*.zip`, parsing step count from the `vgc_ppo_<steps>_steps` filename
  pattern (or treating `final.zip` as the newest, sentinel step `-1`).
- `ResultsRepository` persists to `data/evaluation_results.json`, seeding it with sample history
  (`_SEED_DATA`) on first run if the file doesn't exist.
- `EvaluationRunner` (in `infrastructure/evaluation/`) drives real battles: loads a PPO model via
  `vgc/env.make_vgc_env` + `PPO.load`, runs `n_battles`, and returns an `Evaluation` domain entity —
  invoked from the `POST /api/evaluation/run` endpoint, so triggering an evaluation from the dashboard
  requires the PS server to be running, same as CLI training/eval.

### Dashboard — mirrors the same DDD layering in TypeScript (`dashboard/src/`)

`domain/` (TS interfaces for `Checkpoint`/`Evaluation`) → `infrastructure/api/` (Axios adapters per vertical,
sharing `client.ts`'s `apiClient` which is baseURL `/api`, proxied to `localhost:8080` by Vite) →
`application/` (Pinia stores: `trainingStore`, `evaluationStore`) → `presentation/` (views + components,
organized by vertical under `components/training/` and `components/evaluation/`). Follow this same
domain→infrastructure→application→presentation flow when adding a feature, and keep the vertical slice
(training vs. evaluation) consistent with the API side.

Some dashboard UI copy (button/label text, e.g. "Iniciar evaluación", "Modelo") is in Spanish — match the
existing language when editing a given screen rather than mixing languages within one component.

## Project status (updated 2026-08-12)

The agent has gone through 6 rounds of 500K-step PPO fine-tuning (warm-started each round from the previous
round's `final.zip`), against `SimpleHeuristicsPlayer`, with an opponent team pool that grew each round.
Win rate does **not** improve monotonically round-over-round — it's noisy, and drops when opponent variety
grows faster than training time compensates:

| Round | Opponent pool size | vs Champions (mirror) | vs Trick Room | vs Tailwind | vs Rain | vs Sand | vs Sun |
|---|---|---|---|---|---|---|---|
| 3 | 3 | 66.0% | 78.0% | 68.0% | — | — | — |
| 4 (post action-target fix) | 3 | 68.0% | 70.0% | 32.0% | — | — | — |
| 5 | 6 | 52.0% | **94.0%** | 48.0% | 36.0% | 20.0% | 34.0% |
| 6 (revised Sun/Sand teams) | 6 | **66.0%** | 72.0% | 40.0% | 36.0% | 20.0% | 42.0% |

(50 battles/matchup, deterministic policy.) Diagnosed the round-4 Tailwind dip with a turn-by-turn battle
trace (log active Pokémon/HP/chosen order every turn via `VGCEnv.action_to_order(..., fake=True)`, no need
for PS replays) rather than guessing — found the opponent bot never actually sets up Tailwind at all (so it
wasn't a speed-control issue), but our Garchomp's Earthquake is walled by 3 of that team's 6 Flying-types, a
real fixed-team coverage gap. Use this trace approach before spending another ~3.5h round chasing a
regression blind. **Round 6's Trick Room matchup dropped 94.0%→72.0% (−22pp) on an unchanged opponent
team**, the biggest single-round swing in the table — traced with `vgc/damage_calc.py` (see below) across 15
battles: the opponent bot never set up Trick Room either (0/130 turns, same root cause as Tailwind — see
next paragraph), there's no coverage wall (every opposing Pokémon had a high-damage or guaranteed-KO answer
on paper), but ~22/130 turns had a guaranteed KO available that didn't convert that turn — read as round 6
trading some matchup-specific sharpness for gains elsewhere (mirror +14pp, Sun +8pp), not a discrete bug.

**Known opponent-bot fact**: `SimpleHeuristicsPlayer` never Mega Evolves (confirmed by reading
`poke_env/player/baselines.py` — `Player.create_order` there is called with `dynamax=`/`terastallize=` but
never `mega=True`). Any opponent-team strategy that depends on Mega Evolution to activate its gimmick (e.g.
Charizard → Mega Y for Drought) won't work when the bot pilots that team. Abilities identical between base
and mega form (e.g. Tyranitar's Sand Stream) are unaffected. As of round 6, `SAMPLE_TEAM_SUN` and
`SAMPLE_TEAM_SAND` in `vgc/team.py` have been revised to stop relying on this: Sun's Charizard (mega-Y for
Drought) was replaced with Ninetales (native Drought), and Sand's Tyranitar swapped its now-inert mega stone
for a Chople Berry (Sand Stream was always Tyranitar's base ability, so the mega itself was never the
problem there — just a wasted item slot). Round-5 numbers for these two matchups predate the swap and
aren't a clean baseline for future comparisons; round 6 is the first clean measurement.

**Known opponent-bot fact #2 (root cause of "never sets up Trick Room/Tailwind")**:
`SimpleHeuristicsPlayer.choose_singles_move` (`poke_env/player/baselines.py:322-340`) scores each available
move as `base_power * STAB * atk-or-spa-ratio * accuracy * expected_hits * type_effectiveness` — a product
that starts with `base_power`. Every Status-category move has `base_power == 0`, so its score is always
exactly `0` and it structurally loses `max()` to any damaging move, unless it's the only legal option. A
separate special-cased branch (lines 306-320) lets the bot use self-stat-boost setup moves (`move.boosts` +
`target == "self"`, e.g. Swords Dance) — but field-effect status moves like Trick Room and Tailwind don't set
`move.boosts`, so they never qualify for that carve-out either. **This generalizes to every base_power-0
support move on every opponent-pool team** (screens, Thunder Wave, Helping Hand, Rage Powder, Will-O-Wisp,
Encore, etc.) — none of it ever fires against this bot. Practically: the "vs Trick Room / vs Tailwind"
matchup labels in the table above are really "vs that team's raw stat/type profile, played straight" — the
pool's actual field-effect game plans have never been tested. Would need patching
`SimpleHeuristicsPlayer` (or a different opponent policy) to test/train against real support-move usage —
not a team-composition or env fix.

**Known model gap**: even after fixing the action-target-encoding bug above, the model still essentially
never uses Trick Room in real battles (0/25 in a post-fix diagnostic) — it was trained for hundreds of
thousands of steps under the broken encoding where selecting it almost never worked, so the policy doesn't
spontaneously rediscover the option just because execution is now fixed. Would need extra exploration
(entropy) or explicit encouragement, not just more steps at the current settings.

**Real end goal (as of 2026-08-11, not yet started)**: the user's actual intent is a general assistant that
can read *any* Reg M-B battle (any team on either side) and recommend the best play each turn — not just a
strong bot for our one fixed team. This is a bigger pivot than the current setup can reach by training
longer: the model has only ever controlled our fixed 6-Pokémon roster (varying the *opponent's* team, rounds
3-5, never taught it to pilot a different team itself), and PPO currently returns a single action, not a
ranked/valued list of options an advisor use case would need. The user deliberately asked to let the current
line of work finish and serve "as a base" before redesigning — don't start implementing self-play or an
advisor-style output format unprompted; that conversation needs to be reopened explicitly first.
