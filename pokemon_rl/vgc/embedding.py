"""
embed_battle(battle: DoubleBattle) → np.ndarray[float32, (925,)]

Convierte el estado de una batalla doubles en un vector numérico plano.

Layout del vector (925 valores)
--------------------------------
[0   : 708]  12 Pokémon × 59 features
               ally_active[0], ally_active[1], ally_bench[0-3],
               opp_active[0],  opp_active[1],  opp_bench[0-3]
[708 : 804]   2 slots × 4 movimientos × 24 features  (slot activo 0)
[804 : 900]   ... (slot activo 1)   ← total 192 valores de movimientos
[900 : 921]  21 features de campo (clima, terreno, trampas, pantallas)
[921 : 925]   4 features VGC (can_tera × 2, force_switch × 2)

Nota: el índice 804 corresponde a MOVE_BLOCK_VGC offset, ver constantes.
"""

import numpy as np
from poke_env.battle import STACKABLE_CONDITIONS

from pokemon_rl.embedding import _embed_move, _embed_pokemon, _as_dict
from .constants import (
    FIELD_DIM,
    HAZARD_MAX_STACKS,
    HAZARDS,
    MOVE_DIM,
    N_ACTIVE_PER_SIDE,
    N_MOVES_PER_SLOT,
    N_POKEMON_PER_SIDE,
    N_TERRAINS,
    N_WEATHERS,
    POKEMON_BLOCK_VGC,
    POKEMON_DIM,
    SCREENS,
    TERRAINS,
    VGC_OBS_SIZE,
    WEATHERS,
)


def embed_battle(battle) -> np.ndarray:
    """
    Convierte un DoubleBattle en un vector float32 de 925 valores.

    Parameters
    ----------
    battle : poke_env.battle.DoubleBattle
        Estado de la batalla doubles tal como lo provee poke-env.

    Returns
    -------
    np.ndarray  shape (925,), dtype float32, valores en [-1, 1].
    """
    obs = np.zeros(VGC_OBS_SIZE, dtype=np.float32)
    cursor = 0

    # ------------------------------------------------------------------ #
    # Bloque 1 — Pokémon (708 valores)                                     #
    # ------------------------------------------------------------------ #

    # Activos aliados (índices 0 y 1 de battle.active_pokemon)
    ally_active = battle.active_pokemon   # List[Optional[Pokemon]], len=2

    for pos in range(N_ACTIVE_PER_SIDE):
        pkmn = ally_active[pos] if pos < len(ally_active) else None
        if pkmn is not None and not pkmn.fainted:
            vec = _embed_pokemon(pkmn, is_active=True, is_unknown=False)
        else:
            vec = np.zeros(POKEMON_DIM, dtype=np.float32)
        obs[cursor : cursor + POKEMON_DIM] = vec
        cursor += POKEMON_DIM

    # Banco aliado (los 6 del equipo menos los 2 activos)
    active_species = {
        p.species for p in ally_active if p is not None
    }
    bench_ally = [
        p for p in battle.team.values()
        if p.species not in active_species
    ]
    bench_slots = N_POKEMON_PER_SIDE - N_ACTIVE_PER_SIDE  # 4
    for i in range(bench_slots):
        if i < len(bench_ally):
            vec = _embed_pokemon(bench_ally[i], is_active=False)
        else:
            vec = np.zeros(POKEMON_DIM, dtype=np.float32)
        obs[cursor : cursor + POKEMON_DIM] = vec
        cursor += POKEMON_DIM

    # Activos del rival (índices 0 y 1)
    opp_active = battle.opponent_active_pokemon  # List[Optional[Pokemon]], len=2

    for pos in range(N_ACTIVE_PER_SIDE):
        pkmn = opp_active[pos] if pos < len(opp_active) else None
        if pkmn is not None and not pkmn.fainted:
            vec = _embed_pokemon(pkmn, is_active=True, is_unknown=False)
        else:
            vec = np.zeros(POKEMON_DIM, dtype=np.float32)
        obs[cursor : cursor + POKEMON_DIM] = vec
        cursor += POKEMON_DIM

    # Banco rival (revelados - desconocidos como unknown)
    opp_active_species = {
        p.species for p in opp_active if p is not None
    }
    bench_opp = [
        p for p in battle.opponent_team.values()
        if p.species not in opp_active_species
    ]
    for i in range(bench_slots):
        if i < len(bench_opp):
            vec = _embed_pokemon(bench_opp[i], is_active=False, is_unknown=False)
        else:
            # Slot desconocido del rival
            vec = np.zeros(POKEMON_DIM, dtype=np.float32)
            vec[0]  = 1.0   # hp_fraction = 1.0 asumida
            vec[-1] = 1.0   # is_unknown = True
        obs[cursor : cursor + POKEMON_DIM] = vec
        cursor += POKEMON_DIM

    assert cursor == POKEMON_BLOCK_VGC

    # ------------------------------------------------------------------ #
    # Bloque 2 — Movimientos (192 valores)                                 #
    # pos 0: [708:804],  pos 1: [804:900]                                  #
    # ------------------------------------------------------------------ #

    for pos in range(N_ACTIVE_PER_SIDE):
        active_mon = ally_active[pos] if pos < len(ally_active) else None
        available_ids = {
            m.id for m in (battle.available_moves[pos] if pos < len(battle.available_moves) else [])
        }
        all_moves = list(active_mon.moves.values()) if active_mon else []

        for i in range(N_MOVES_PER_SLOT):
            move = all_moves[i] if i < len(all_moves) else None
            is_avail = (move is not None) and (move.id in available_ids)
            obs[cursor : cursor + MOVE_DIM] = _embed_move(move, is_available=is_avail)
            cursor += MOVE_DIM

    # ------------------------------------------------------------------ #
    # Bloque 3 — Campo (21 valores)                                        #
    # ------------------------------------------------------------------ #

    weather_dict = _as_dict(battle.weather)
    fields_dict  = _as_dict(battle.fields)

    for w in WEATHERS:
        obs[cursor] = float(w in weather_dict)
        cursor += 1

    for t in TERRAINS:
        obs[cursor] = float(t in fields_dict)
        cursor += 1

    # Trampas propias
    for hazard, max_stacks in zip(HAZARDS, HAZARD_MAX_STACKS):
        if hazard in STACKABLE_CONDITIONS:
            stacks = battle.side_conditions.get(hazard, 0)
            obs[cursor] = min(stacks, max_stacks) / max_stacks
        else:
            obs[cursor] = float(hazard in battle.side_conditions)
        cursor += 1

    # Trampas rivales
    for hazard, max_stacks in zip(HAZARDS, HAZARD_MAX_STACKS):
        if hazard in STACKABLE_CONDITIONS:
            stacks = battle.opponent_side_conditions.get(hazard, 0)
            obs[cursor] = min(stacks, max_stacks) / max_stacks
        else:
            obs[cursor] = float(hazard in battle.opponent_side_conditions)
        cursor += 1

    # Pantallas
    for screen in SCREENS:
        obs[cursor] = float(screen in battle.side_conditions)
        cursor += 1
    for screen in SCREENS:
        obs[cursor] = float(screen in battle.opponent_side_conditions)
        cursor += 1

    # ------------------------------------------------------------------ #
    # Bloque 4 — VGC extras (4 valores)                                    #
    # ------------------------------------------------------------------ #

    # can_tera por posición
    can_tera = battle.can_tera if hasattr(battle, "can_tera") else [False, False]
    obs[cursor]     = float(can_tera[0]) if len(can_tera) > 0 else 0.0
    obs[cursor + 1] = float(can_tera[1]) if len(can_tera) > 1 else 0.0
    cursor += 2

    # force_switch por posición
    force_sw = battle.force_switch if hasattr(battle, "force_switch") else [False, False]
    obs[cursor]     = float(force_sw[0]) if len(force_sw) > 0 else 0.0
    obs[cursor + 1] = float(force_sw[1]) if len(force_sw) > 1 else 0.0
    cursor += 2

    assert cursor == VGC_OBS_SIZE, f"VGC embedding cursor {cursor} != {VGC_OBS_SIZE}"
    return obs
