# vgc-rl-agent

Reinforcement learning agent for Pokémon VGC doubles battles using PPO + behavioral cloning from human replays, trained on a local Pokémon Showdown server.

## Results

The agent has gone through 15 rounds of PPO fine-tuning (each warm-started from an earlier round's checkpoint)
against increasingly realistic opponents, with an opponent team pool that grew over time. Win rate is **noisy
round-over-round rather than monotonically improving** — see `CLAUDE.md` for the full round-by-round history
and diagnostics, and `data/results.sqlite3` (below) for the queryable version of the same history.

Rounds 8-11 experimented with **self-play** (our own side also picks a random team per battle, not just the
opponent's) to see whether the agent could learn to pilot *any* team in the pool, not just its original
Champions roster — it didn't work well in one or two rounds (the 6×6 own-team × opponent-team combinations
diluted practice too much for any one pairing), so rounds 12+ pivoted to **dedicated single-team immersion**
rounds instead: warm-start from the self-play checkpoint, then give one team a full round of undiluted
practice. That approach worked — Rain and Champions both reached ~40-48% average win rate this way; Trick
Room did not (~9%, a much bigger behavioral departure — inverted turn order vs. normal-speed play — that a
single immersion round couldn't fix). Round 15 (Champions, closing that line) result, 50 deterministic-policy
battles per matchup in `[Gen 9 Champions] VGC 2026 Reg M-B`, opponent `support_heuristic`:

| Matchup | Win rate |
|---------|----------|
| Champions (mirror) | 48.0% |
| Trick Room | 80.0% |
| Tailwind | 46.0% |
| Rain | 40.0% |
| Sand | 28.0% |
| Sun | 48.0% |

## Architecture

```
Human replays (PS API)
        │
        ▼
Behavioral Cloning (BC)      ← supervised pre-training
        │
        ▼
PPO fine-tuning              ← reinforcement learning
        │
        ▼
Trained agent (MultiInputPolicy)
```

### Observation space — `Box(925,) float32`

| Segment | Size | Description |
|---------|------|-------------|
| Active Pokémon (×2) | 2 × 85 | Species, HP%, status, stats, moves, type |
| Bench (×4) | 4 × 60 | HP%, fainted, type |
| Opponent active (×2) | 2 × 85 | Same as ours |
| Opponent bench (×4) | 4 × 25 | Reduced info |
| Field / weather | 45 | Weather, terrain, trick room, screens |

### Action space — `MultiDiscrete([107, 107])`

One action per active slot, encoded as:

| Range | Action |
|-------|--------|
| 0 | Pass |
| 1–6 | Switch to team slot N |
| 7–26 | Move 1–4 × target (5 targets) |
| 27–106 | Same with Mega / Z-move / Dynamax / Tera |

## Project structure

```
pokemon_rl/
├── vgc/
│   ├── constants.py      # Shared constants
│   ├── embedding.py      # Doubles battle embedding (925-dim)
│   ├── env.py            # VGC Gymnasium environment + action-decoding fixes
│   ├── team.py           # Sample teams (Champions Reg M-B, VGC Reg G) + opponent-pool archetypes
│   ├── teambuilder.py    # RandomTeamPool — opponent picks a random team per battle
│   ├── opponents.py      # SupportAwareHeuristicsPlayer — opponent that actually uses status/field moves
│   ├── damage_calc.py    # Damage estimator (wraps poke-env's Gen9 calc) for matchup diagnostics
│   ├── advisor.py        # rank_actions()/estimate_state_value() — ranked-action inference, no retrain needed
│   ├── results_db.py     # SQLite schema + tracking helpers for training_runs/evaluations/battle_results
│   ├── train.py          # VGC PPO training with optional BC warm-start
│   └── evaluate.py       # Win-rate evaluation script (CLI)
│
├── api/                  # FastAPI backend (DDD)
│   ├── domain/           # Entities: Checkpoint, Evaluation, BattleResult
│   ├── application/      # Use cases: queries + RunEvaluation command
│   ├── infrastructure/   # Repositories: disk checkpoints, results SQLite (data/results.sqlite3)
│   ├── presentation/     # Routers: /api/training, /api/evaluation
│   └── main.py           # FastAPI entry point
│
└── imitation/
    ├── downloader.py     # Download replays from PS API
    ├── parser.py         # Parse replay logs → (obs, action) pairs
    ├── dataset.py        # PyTorch Dataset wrapper
    └── train_bc.py       # Behavioral cloning pre-training

dashboard/                # Vue 3 training dashboard (DDD)
├── src/
│   ├── domain/           # TypeScript interfaces: Checkpoint, Evaluation
│   ├── application/      # Pinia stores: trainingStore, evaluationStore
│   ├── infrastructure/   # Axios API adapters
│   └── presentation/     # Views + components (chart, table, launcher)
└── package.json
```

## Requirements

- Python 3.11+
- Node.js v20+ (for the local PS server and dashboard)
- A local [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) server running on `localhost:8000`

## Setup

```bash
# Clone and create virtual environment
git clone https://github.com/JonaOlave/vgc-rl-agent.git
cd vgc-rl-agent
python -m venv pokemon_rl/.venv
source pokemon_rl/.venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-api.txt
```

Start the local PS server (in a separate terminal):
```bash
cd /path/to/pokemon-showdown
node pokemon-showdown start --no-security --port 8000
```

## Dashboard

The dashboard lets you visualize training history, compare win rates across opponents, browse per-battle results, and trigger new evaluations — all from a browser.

### 1 — Start the FastAPI backend

The backend serves training checkpoints and evaluation results from `data/results.sqlite3` (see
[Results tracking](#results-tracking) below), and can run new evaluations on demand.

```bash
# From the project root, with the virtual environment active
source pokemon_rl/.venv/bin/activate
uvicorn pokemon_rl.api.main:app --port 8080 --reload
```

The API will be available at `http://localhost:8080`. Interactive docs at `http://localhost:8080/docs`.

| Endpoint | Description |
|----------|-------------|
| `GET /api/training/checkpoints` | List all saved model checkpoints |
| `GET /api/evaluation/results` | List all evaluation runs |
| `GET /api/evaluation/results/{id}` | Get a specific evaluation with per-battle detail |
| `POST /api/evaluation/run` | Run a new evaluation against random/heuristic/support_heuristic |

### 2 — Start the Vue 3 dashboard

```bash
cd dashboard
npm install      # only needed the first time
npm run dev
```

Open **http://localhost:5173** in your browser.

![Dashboard overview](https://i.imgur.com/placeholder.png)

#### What you can see

| Panel | Description |
|-------|-------------|
| **Stat cards** | Best win rate, latest vs random, latest vs heuristic, total evaluations |
| **Win rate chart** | Historical win rate over time — blue line = vs random, orange = vs heuristic |
| **Checkpoints** | All saved `.zip` checkpoints with step count and timestamp |
| **Evaluations** | Sortable list of all evaluation runs with win rate indicator |
| **Battle table** | Per-battle breakdown (result, reward, fainted counts) for the selected evaluation |
| **New evaluation** | Form to launch an evaluation directly from the browser |

### 3 — Run a new evaluation from the dashboard

1. Select a model from the **Modelo** dropdown (defaults to `final`)
2. Choose opponent: **Random** or **Heurístico**
3. Set the number of battles
4. Click **Iniciar evaluación** — results appear automatically in the chart and table once complete

## Training pipeline

### Step 1 — Behavioral cloning from human replays

```bash
python -m pokemon_rl.imitation.train_bc \
    --download \
    --n-download 500 \
    --format gen9vgc2025regg \
    --epochs 30 \
    --save-path models/bc_pretrained
```

### Step 2 — PPO fine-tuning with BC warm-start

```bash
python -m pokemon_rl.vgc.train \
    --format gen9championsvgc2026regmb \
    --champions-team \
    --opponent heuristic \
    --bc-model models/bc_pretrained \
    --vary-opponent-team \
    --timesteps 500000
```

- `--opponent` accepts `random`, `heuristic` (poke-env's `SimpleHeuristicsPlayer`), or
  `support_heuristic` (our own `SupportAwareHeuristicsPlayer` — same heuristics, but actually uses
  Trick Room / Tailwind / screens / weather-setting moves; see [Notable fixes](#notable-fixes--additions)).
- `--vary-opponent-team` makes the opponent pick a random team per battle from
  `OPPONENT_TEAM_POOL_CHAMPIONS_REGMB` (`vgc/team.py`) instead of always mirroring our own team.
- `--vary-own-team` does the same for **our own** side (self-play) — pass both flags together to have
  both sides sample independently from the pool each battle. `train()` also accepts `opponent_team_pool`/
  `own_team_pool` directly (a list of team strings) to use a custom-sized pool instead of the full 6.

### Step 3 — Evaluate

```bash
python -m pokemon_rl.vgc.evaluate \
    --champions-team \
    --opponent random \
    --n-battles 50
```

### Results tracking

Every `train()` call registers itself in `data/results.sqlite3` automatically — no extra step needed. It
inserts a `training_runs` row when a round starts (auto-incrementing round number, and resolving
`warm_start_run_id` by comparing the starting checkpoint's MD5 against already-recorded checkpoints, so it
works regardless of which path alias was used to load it) and updates it with actual timesteps, duration,
and the final checkpoint's path/MD5 when the round finishes.

`evaluate()` doesn't auto-persist (it's also used standalone and by the API's evaluation runner), but returns
a `"battles"` key with full per-battle detail alongside the aggregate stats, ready to hand to
`results_db.record_evaluation()`:

```python
from pokemon_rl.vgc.results_db import connect, record_evaluation

conn = connect()
result = evaluate(team=SAMPLE_TEAM_CHAMPIONS_REGMB, opponent_team_pool=[SAMPLE_TEAM_TRICKROOM], ...)
record_evaluation(conn, training_run_id=16, our_team="champions",
                   opponent_archetype="trickroom", opponent_behavior="support_heuristic", result=result)
```

The schema (`training_runs` → `evaluations` → `battle_results`, plus a self-referencing `warm_start_run_id`
on `training_runs` for tracing a checkpoint's full lineage) is defined in `pokemon_rl/vgc/results_db.py`. The
dashboard's `/api/evaluation/*` endpoints read from this same database via `SqliteResultsRepository`.

## Team — Champions VGC 2026 Reg M-B

| Pokémon | Item | Role |
|---------|------|------|
| Incineroar | Sitrus Berry | Fake Out + Intimidate support |
| Grimmsnarl | Light Clay | Prankster screens |
| Garchomp | Garchompite | Mega physical sweeper |
| Annihilape | Focus Sash | Rage Fist + Trick Room counter |
| Gholdengo | White Herb | Special sweeper |
| Hatterene | Colbur Berry | Trick Room setter |

### Opponent team pool

Training and evaluation vary the *opponent's* team per battle across 6 archetypes (`vgc/team.py`), so the
agent doesn't just learn to beat a mirror match: Champions (mirror), Trick Room, Tailwind, Rain, Sand, and
Sun. The Sun and Sand teams were revised to stop relying on Mega Evolution (Charizard → Mega Y for Drought,
Tyranitar's now-unused mega stone) once we confirmed `SimpleHeuristicsPlayer` never mega-evolves — see below.

## Notable fixes & additions

A few non-obvious issues turned up while getting PPO to reliably pilot a doubles team, worth knowing before
touching action decoding, the opponent setup, or the reward/observation pipeline:

- **Force-switch bypass** (`VGCEnv.action_to_order`) — standard PPO ignores the action mask, so on
  forced-switch turns (a Pokémon faints mid-turn) the agent regularly emitted invalid orders that fell back
  to a random move instead of a random *valid switch*. Fixed by detecting the forced-switch case and picking
  directly from `battle.valid_orders`, bypassing the model's action for that turn only.
- **Target-encoding fix** (`VGCEnv._normalize_targets`) — for moves that don't target a specific Pokémon
  (Trick Room, Protect, Tailwind, screens, Substitute, etc.), PPO's sampled target axis was almost always
  wrong, silently invalidating the whole order even when the move choice itself was correct. Fixed by
  normalizing the target from the move's `deduced_target` classification.
- **Opponent team pool** (`vgc/teambuilder.py`'s `RandomTeamPool`) — poke-env's `SingleAgentWrapper` shares
  one team object between both internal battling agents by construction, so training was silently always a
  mirror match until this was patched to let the opponent side pick a random team per battle.
- **`SimpleHeuristicsPlayer` never uses status/field moves** — its move-scoring formula is a product that
  starts with `base_power`, so any Status-category move (Trick Room, Tailwind, screens, weather-setters, all
  `base_power == 0`) scores exactly `0` and structurally never wins against a damaging move. This meant none
  of the opponent pool's named archetypes ever actually executed their game plan. `vgc/opponents.py`'s
  `SupportAwareHeuristicsPlayer` patches this with speed-aware conditions (Trick Room if our team is slower
  on average, Tailwind if we don't have a speed edge, screens/weather early-game) — opt in via
  `--opponent support_heuristic`.
- **Damage estimator for diagnostics** (`vgc/damage_calc.py`) — wraps poke-env's own bundled Gen9 damage
  calculator (a Python port of Smogon's calc) to answer "what's the expected damage/KO chance here" during
  turn-by-turn battle traces, instead of guessing whether a win-rate regression is a coverage gap or an
  execution issue.
- **PPO never learned to respect its own action mask** — the model's raw output distribution assigns
  non-trivial probability to illegal actions (the same underlying issue behind the force-switch/target fixes
  above). `vgc/advisor.py`'s `rank_actions()` reads SB3's per-head action-probability distribution directly
  and filters it against `DoublesEnv.get_action_mask_individual` before ranking, instead of trusting the raw
  top-k. Doesn't require retraining — works against any existing checkpoint. Its probabilities reflect what
  the policy tends to pick (π(a|s)), not an estimate of outcome value — PPO only computes V(s) for the whole
  state, not a value per action.
- **Self-play generalizes far more slowly than expected** — training both sides' team selection at once (6×6
  own-team × opponent-team combinations) diluted practice per pairing badly; a single dedicated round for one
  team (own side fixed, opponent pool varies) reached competence in one shot where self-play plateaued for
  several rounds. See the Results section above and `CLAUDE.md` for the full investigation, including why
  Trick Room specifically never recovered even with a clean immersion round.

## License

MIT License — © 2026 [JonaOlave](https://github.com/JonaOlave)
