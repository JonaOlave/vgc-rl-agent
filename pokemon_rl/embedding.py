"""
embed_battle(battle) → np.ndarray[float32, (825,)]

Converts a poke-env Battle object into a flat numerical vector that can
be fed directly to a neural network.

Observation layout
------------------
[0   : 708]  12 Pokemon × 59 features  (6 ally + 6 opponent, active first)
[708 : 804]   4 moves  × 24 features   (moves of active Pokemon)
[804 : 825]  21 field features          (weather, terrain, hazards, screens)
"""

import numpy as np
from poke_env.battle import STACKABLE_CONDITIONS

from .constants import (
    BOOST_KEYS,
    CATEGORY_INDEX,
    FIELD_DIM,
    HAZARD_MAX_STACKS,
    HAZARDS,
    MAX_BASE_POWER,
    MAX_BASE_STAT,
    MOVE_BLOCK,
    MOVE_DIM,
    N_BOOSTS,
    N_CATEGORIES,
    N_MOVES,
    N_POKEMON_PER_SIDE,
    N_SIDES,
    N_STATUSES,
    N_STATS,
    N_TERRAINS,
    N_TYPES,
    N_WEATHERS,
    OBS_SIZE,
    POKEMON_BLOCK,
    POKEMON_DIM,
    SCREENS,
    STAT_KEYS,
    STATUS_INDEX,
    TERRAIN_INDEX,
    TERRAINS,
    TYPE_INDEX,
    WEATHER_INDEX,
    WEATHERS,
)


# ---------------------------------------------------------------------------
# Per-Pokemon embedding
# ---------------------------------------------------------------------------

def _embed_pokemon(pokemon, *, is_active: bool, is_unknown: bool = False) -> np.ndarray:
    """
    Return a float32 vector of shape (POKEMON_DIM,) = (59,) for one Pokemon.

    is_unknown=True is used for opponent slots that haven't been revealed yet.
    In that case we only know the slot exists; we assume full HP and set the
    is_unknown flag so the model can learn to handle hidden information.
    """
    vec = np.zeros(POKEMON_DIM, dtype=np.float32)
    cursor = 0

    # --- hp_fraction (1) ---
    if is_unknown:
        vec[cursor] = 1.0          # assume full HP
    elif pokemon.fainted:
        vec[cursor] = 0.0
    else:
        vec[cursor] = float(pokemon.current_hp_fraction)
    cursor += 1

    if not is_unknown:
        # --- type1 one-hot (N_TYPES) ---
        if pokemon.types[0] is not None:
            t1 = TYPE_INDEX.get(pokemon.types[0], -1)
            if t1 >= 0:
                vec[cursor + t1] = 1.0
        cursor += N_TYPES

        # --- type2 one-hot (N_TYPES) ---
        if len(pokemon.types) > 1 and pokemon.types[1] is not None:
            t2 = TYPE_INDEX.get(pokemon.types[1], -1)
            if t2 >= 0:
                vec[cursor + t2] = 1.0
        cursor += N_TYPES

        # --- status one-hot (N_STATUSES) ---
        if pokemon.status is not None:
            s = STATUS_INDEX.get(pokemon.status, -1)
            if s >= 0:
                vec[cursor + s] = 1.0
        cursor += N_STATUSES

        # --- base stats normalized (N_STATS) ---
        for i, key in enumerate(STAT_KEYS):
            vec[cursor + i] = pokemon.base_stats.get(key, 0) / MAX_BASE_STAT
        cursor += N_STATS

        # --- boosts normalized to [-1, 1] (N_BOOSTS) ---
        # Boosts are only meaningful while the Pokemon is active.
        if is_active:
            for i, key in enumerate(BOOST_KEYS):
                vec[cursor + i] = pokemon.boosts.get(key, 0) / 6.0
        cursor += N_BOOSTS

    else:
        # Skip all inner fields for unknown pokemon
        cursor += N_TYPES + N_TYPES + N_STATUSES + N_STATS + N_BOOSTS

    # --- flags (3) ---
    vec[cursor]     = float(is_active)
    vec[cursor + 1] = float(pokemon.fainted) if not is_unknown else 0.0
    vec[cursor + 2] = float(is_unknown)

    return vec


# ---------------------------------------------------------------------------
# Per-move embedding
# ---------------------------------------------------------------------------

def _embed_move(move, *, is_available: bool) -> np.ndarray:
    """
    Return a float32 vector of shape (MOVE_DIM,) = (24,) for one move.
    A zero vector indicates an empty move slot.
    """
    vec = np.zeros(MOVE_DIM, dtype=np.float32)
    if move is None:
        return vec

    cursor = 0

    # --- base_power normalized (1) ---
    vec[cursor] = move.base_power / MAX_BASE_POWER
    cursor += 1

    # --- type one-hot (N_TYPES) ---
    t = TYPE_INDEX.get(move.type, -1)
    if t >= 0:
        vec[cursor + t] = 1.0
    cursor += N_TYPES

    # --- category one-hot (N_CATEGORIES) ---
    c = CATEGORY_INDEX.get(move.category, -1)
    if c >= 0:
        vec[cursor + c] = 1.0
    cursor += N_CATEGORIES

    # --- pp_fraction (1) ---
    vec[cursor] = move.current_pp / move.max_pp if move.max_pp > 0 else 0.0
    cursor += 1

    # --- is_available (1) ---
    vec[cursor] = float(is_available)

    return vec


# ---------------------------------------------------------------------------
# Field embedding helpers
# ---------------------------------------------------------------------------

def _as_dict(mapping) -> dict:
    """Normalize battle.weather / battle.fields to a plain dict."""
    if isinstance(mapping, dict):
        return mapping
    return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_battle(battle) -> np.ndarray:
    """
    Convert a poke-env Battle into a flat float32 observation vector.

    Parameters
    ----------
    battle : poke_env.environment.battle.Battle
        The current battle state as provided by poke-env.

    Returns
    -------
    np.ndarray
        Shape (825,), dtype float32, all values in [-1, 1].
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    cursor = 0

    # ------------------------------------------------------------------ #
    # Block 1 — Pokemon (708 values)                                       #
    # ------------------------------------------------------------------ #

    active_species = (
        battle.active_pokemon.species if battle.active_pokemon else None
    )

    # Ally team: active Pokemon first, then bench in arbitrary order
    ally_team = sorted(
        battle.team.values(),
        key=lambda p: (0 if p.species == active_species else 1),
    )

    for i in range(N_POKEMON_PER_SIDE):
        if i < len(ally_team):
            pkmn = ally_team[i]
            vec = _embed_pokemon(
                pkmn,
                is_active=(pkmn.species == active_species),
                is_unknown=False,
            )
        else:
            vec = np.zeros(POKEMON_DIM, dtype=np.float32)
        obs[cursor : cursor + POKEMON_DIM] = vec
        cursor += POKEMON_DIM

    # Opponent team: active first, then seen bench, then unknown slots
    opp_active_species = (
        battle.opponent_active_pokemon.species
        if battle.opponent_active_pokemon
        else None
    )
    opp_team = sorted(
        battle.opponent_team.values(),
        key=lambda p: (0 if p.species == opp_active_species else 1),
    )

    for i in range(N_POKEMON_PER_SIDE):
        if i < len(opp_team):
            pkmn = opp_team[i]
            vec = _embed_pokemon(
                pkmn,
                is_active=(pkmn.species == opp_active_species),
                is_unknown=False,
            )
        else:
            # Unrevealed opponent slot
            vec = np.zeros(POKEMON_DIM, dtype=np.float32)
            vec[0]  = 1.0   # hp_fraction = 1.0 (assume full HP)
            vec[-1] = 1.0   # is_unknown = True
        obs[cursor : cursor + POKEMON_DIM] = vec
        cursor += POKEMON_DIM

    assert cursor == POKEMON_BLOCK

    # ------------------------------------------------------------------ #
    # Block 2 — Moves (96 values)                                          #
    # ------------------------------------------------------------------ #

    available_ids = {m.id for m in battle.available_moves}
    active_moves = (
        list(battle.active_pokemon.moves.values())
        if battle.active_pokemon
        else []
    )

    for i in range(N_MOVES):
        move = active_moves[i] if i < len(active_moves) else None
        is_avail = (move is not None) and (move.id in available_ids)
        obs[cursor : cursor + MOVE_DIM] = _embed_move(move, is_available=is_avail)
        cursor += MOVE_DIM

    assert cursor == POKEMON_BLOCK + MOVE_BLOCK

    # ------------------------------------------------------------------ #
    # Block 3 — Field (21 values)                                          #
    # ------------------------------------------------------------------ #

    weather_dict = _as_dict(battle.weather)
    fields_dict  = _as_dict(battle.fields)

    # Weather (5)
    for w in WEATHERS:
        obs[cursor] = float(w in weather_dict)
        cursor += 1

    # Terrain (4)
    for t in TERRAINS:
        obs[cursor] = float(t in fields_dict)
        cursor += 1

    # Own hazards (4)
    # STACKABLE_CONDITIONS (Spikes, Toxic Spikes) store real stack count.
    # Non-stackable (Stealth Rock, Sticky Web) store the turn they were set
    # — treat those as binary (present/absent).
    for hazard, max_stacks in zip(HAZARDS, HAZARD_MAX_STACKS):
        if hazard in STACKABLE_CONDITIONS:
            stacks = battle.side_conditions.get(hazard, 0)
            obs[cursor] = min(stacks, max_stacks) / max_stacks
        else:
            obs[cursor] = float(hazard in battle.side_conditions)
        cursor += 1

    # Opponent hazards (4)
    for hazard, max_stacks in zip(HAZARDS, HAZARD_MAX_STACKS):
        if hazard in STACKABLE_CONDITIONS:
            stacks = battle.opponent_side_conditions.get(hazard, 0)
            obs[cursor] = min(stacks, max_stacks) / max_stacks
        else:
            obs[cursor] = float(hazard in battle.opponent_side_conditions)
        cursor += 1

    # Own screens (2) — store turn number, treat as binary
    for screen in SCREENS:
        obs[cursor] = float(screen in battle.side_conditions)
        cursor += 1

    # Opponent screens (2)
    for screen in SCREENS:
        obs[cursor] = float(screen in battle.opponent_side_conditions)
        cursor += 1

    assert cursor == OBS_SIZE, f"Embedding cursor {cursor} != OBS_SIZE {OBS_SIZE}"
    return obs
