# vgc-rl-agent

Reinforcement learning agent for Pokémon VGC doubles battles using PPO + behavioral cloning from human replays, trained on a local Pokémon Showdown server.

## Results

Evaluated on **50 battles** against a random opponent in the `[Gen 9 Champions] VGC 2026 Reg M-B` format after 100K training steps:

| Metric | Value |
|--------|-------|
| Win rate | **78%** |
| Average fainted (ours) | 1.82 / 6 |
| Average fainted (opponent) | 3.54 / 6 |
| Mean reward | +1.54 |

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
│   ├── env.py            # VGC Gymnasium environment + force-switch fix
│   ├── team.py           # Sample teams (Champions Reg M-B, VGC Reg G)
│   ├── train.py          # VGC PPO training with optional BC warm-start
│   └── evaluate.py       # Win-rate evaluation script (CLI)
│
├── api/                  # FastAPI backend (DDD)
│   ├── domain/           # Entities: Checkpoint, Evaluation, BattleResult
│   ├── application/      # Use cases: queries + RunEvaluation command
│   ├── infrastructure/   # Repositories: disk checkpoints, results JSON
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

The backend serves training checkpoints and evaluation results, and can run new evaluations on demand.

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
| `POST /api/evaluation/run` | Run a new evaluation against random or heuristic |

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
    --opponent random \
    --bc-model models/bc_pretrained \
    --timesteps 500000
```

### Step 3 — Evaluate

```bash
python -m pokemon_rl.vgc.evaluate \
    --champions-team \
    --opponent random \
    --n-battles 50
```

## Team — Champions VGC 2026 Reg M-B

| Pokémon | Item | Role |
|---------|------|------|
| Incineroar | Sitrus Berry | Fake Out + Intimidate support |
| Grimmsnarl | Light Clay | Prankster screens |
| Garchomp | Garchompite | Mega physical sweeper |
| Annihilape | Focus Sash | Rage Fist + Trick Room counter |
| Gholdengo | White Herb | Special sweeper |
| Hatterene | Colbur Berry | Trick Room setter |

## License

MIT License — © 2026 [JonaOlave](https://github.com/JonaOlave)
