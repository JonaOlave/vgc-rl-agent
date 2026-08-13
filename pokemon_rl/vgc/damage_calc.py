"""
Estimador de daño para VGC doubles.

Wrapper fino sobre poke_env.calc.damage_calc_gen9 (puerto Python del
calculador de Smogon: https://github.com/smogon/damage-calc) — no
reimplementamos tabla de tipos ni fórmula de daño a mano, sólo adaptamos
la interfaz a objetos Pokemon/Move de poke-env y devolvemos algo directamente
legible (porcentaje de HP, KO garantizado/posible) en vez de un rango de HP
crudo.

Requiere una `battle` en curso (o una battle "fake" construida a mano con
equipos/stats/campo poblados) — no calcula nada a partir de sólo strings de
equipo, porque el daño depende de boosts, clima, terreno, pantallas, etc.,
que sólo existen en el estado de una batalla real.

Uso típico:

    from pokemon_rl.vgc.damage_calc import estimate_damage

    est = estimate_damage(battle, attacker=my_garchomp, defender=opp_slowking, move=earthquake)
    est["pct_max"]        # 43.2  (daño máximo esperado, % del HP máx. del defensor)
    est["guaranteed_ko"]  # False
"""

from typing import TypedDict

from poke_env.battle import DoubleBattle, Move, Pokemon
from poke_env.calc import calculate_damage


class DamageEstimate(TypedDict):
    hp_min: int
    hp_max: int
    pct_min: float
    pct_max: float
    possible_ko: bool
    guaranteed_ko: bool


def estimate_damage(
    battle: DoubleBattle,
    attacker: Pokemon,
    defender: Pokemon,
    move: Move,
) -> DamageEstimate:
    """
    Rango de daño esperado de `move` de `attacker` contra `defender`, dado
    el estado actual de `battle` (clima, terreno, pantallas, boosts, ítems,
    habilidades — todo lo que ya trackea poke-env).

    `attacker`/`defender` deben ser objetos Pokemon ya presentes en
    `battle` (de `battle.team`, `battle.opponent_team`, `battle.active_pokemon`
    o `battle.opponent_active_pokemon`) — se usan para determinar de qué
    lado es cada uno (`battle.player_role` vs `battle.opponent_role`).
    """
    attacker_role = (
        battle.player_role if attacker in battle.team.values() else battle.opponent_role
    )
    defender_role = (
        battle.player_role if defender in battle.team.values() else battle.opponent_role
    )
    assert attacker_role is not None and defender_role is not None

    hp_min, hp_max = calculate_damage(
        attacker.identifier(attacker_role),
        defender.identifier(defender_role),
        move,
        battle,
    )

    defender_max_hp = defender.max_hp or 1

    return DamageEstimate(
        hp_min=hp_min,
        hp_max=hp_max,
        pct_min=round(100 * hp_min / defender_max_hp, 1),
        pct_max=round(100 * hp_max / defender_max_hp, 1),
        possible_ko=hp_max >= defender.current_hp,
        guaranteed_ko=hp_min >= defender.current_hp,
    )
