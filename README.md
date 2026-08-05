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
├── embedding.py          # Singles battle embedding (825-dim)
├── env.py                # Singles Gymnasium environment
├── train.py              # Singles PPO training
│
├── vgc/
│   ├── constants.py      # Shared constants
│   ├── embedding.py      # Doubles battle embedding (925-dim)
│   ├── env.py            # VGC Gymnasium environment + force-switch fix
│   ├── team.py           # Sample teams (Champions Reg M-B, VGC Reg G)
│   ├── train.py          # VGC PPO training with optional BC warm-start
│   └── evaluate.py       # Win-rate evaluation script
│
└── imitation/
    ├── downloader.py     # Download replays from PS API
    ├── parser.py         # Parse replay logs → (obs, action) pairs
    ├── dataset.py        # PyTorch Dataset wrapper
    └── train_bc.py       # Behavioral cloning pre-training
```

## Requirements

- Python 3.11+
- Node.js v20+ (for the local PS server)
- A local [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) server running on `localhost:8000`

## Setup

```bash
# Clone and create virtual environment
git clone https://github.com/JonaOlave/vgc-rl-agent.git
cd vgc-rl-agent
python -m venv pokemon_rl/.venv
source pokemon_rl/.venv/bin/activate
pip install -r requirements.txt
```

Start the local PS server (in a separate terminal):
```bash
cd /path/to/pokemon-showdown
node pokemon-showdown start --no-security --port 8000
```

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
