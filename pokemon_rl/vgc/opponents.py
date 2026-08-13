"""
Oponentes heurísticos propios.

`SimpleHeuristicsPlayer` (de poke-env) nunca elige movimientos de Status:
su fórmula de puntaje (`baselines.py:322-340`) es un producto que arranca
multiplicando por `base_power`, que vale 0 para toda categoría Status —
así que Trick Room, Tailwind, pantallas y clima nunca ganan el max() frente
a cualquier movimiento que haga daño. Confirmado leyendo el código; ver
CLAUDE.md, "Known opponent-bot fact #2".

`SupportAwareHeuristicsPlayer` agrega una rama previa al max() de daño para
esos movimientos de soporte, con condiciones parecidas a las que usaría un
jugador real (Trick Room si el equipo propio es más lento en promedio,
Tailwind si no hay ventaja de velocidad, pantallas/clima temprano en la
partida y sólo si no están ya activos) — el resto de la lógica (switches,
selección por daño, dynamax/tera) queda exactamente igual que en la clase
base.

Alcance deliberadamente acotado: cubre los movimientos que definen la
identidad de los arquetipos del pool actual (Trick Room, Tailwind,
pantallas, clima — ver `vgc/team.py`). No intenta dar lógica realista a
otros movimientos de Status (Thunder Wave, Will-O-Wisp, Encore, Helping
Hand, Rage Powder, etc.) — esos siguen sin dispararse nunca, igual que
antes de este parche.

Nota de implementación: en dobles, la lógica de `choose_move` del padre
llama a `choose_singles_move` con un `PseudoBattle` (ver
`poke_env/player/baselines.py`) que sólo copia un subconjunto de atributos
de la battle real — no incluye `fields`/`weather`/`turn` (los necesarios
para saber si Trick Room/clima ya están activos, o en qué turno estamos).
Por eso `choose_move` los guarda en `self` mientras todavía tiene la
`DoubleBattle` real, y `choose_singles_move` los lee de ahí en vez de
`battle.fields`/`battle.weather`/`battle.turn` directamente.
"""

from typing import Any, Dict, Tuple

from poke_env.battle import AbstractBattle, Field, SideCondition, Weather
from poke_env.player import Player, SimpleHeuristicsPlayer
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder


class SupportAwareHeuristicsPlayer(SimpleHeuristicsPlayer):
    _SCREEN_MOVES: Dict[str, SideCondition] = {
        "lightscreen": SideCondition.LIGHT_SCREEN,
        "reflect": SideCondition.REFLECT,
        "auroraveil": SideCondition.AURORA_VEIL,
    }
    _WEATHER_MOVES: Dict[str, Weather] = {
        "sunnyday": Weather.SUNNYDAY,
        "raindance": Weather.RAINDANCE,
        "sandstorm": Weather.SANDSTORM,
        "hail": Weather.HAIL,
        "snowscape": Weather.SNOW,
    }
    # Pantallas/clima sólo se consideran en los primeros turnos — como un
    # jugador real, no tiene sentido levantarlas cuando la batalla ya está
    # resuelta.
    EARLY_GAME_TURN_LIMIT = 3

    @staticmethod
    def _avg_speed(mons) -> float:
        alive = [m for m in mons if not m.fainted]
        if not alive:
            return 0.0
        return sum(m.base_stats["spe"] for m in alive) / len(alive)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        self._support_fields = getattr(battle, "fields", {}) or {}
        self._support_weather = getattr(battle, "weather", {}) or {}
        self._support_turn = getattr(battle, "turn", 0)
        return super().choose_move(battle)

    def choose_singles_move(
        self, battle: AbstractBattle
    ) -> Tuple[SingleBattleOrder, float]:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        fields: Dict[Any, int] = getattr(self, "_support_fields", {})
        weather: Dict[Any, int] = getattr(self, "_support_weather", {})
        turn: int = getattr(self, "_support_turn", 0)

        if active is not None and opponent is not None and battle.available_moves:
            our_speed = self._avg_speed(battle.team.values())
            opp_speed = self._avg_speed(battle.opponent_team.values())

            for move in battle.available_moves:
                if move.id == "trickroom":
                    if Field.TRICK_ROOM not in fields and our_speed < opp_speed:
                        return Player.create_order(move), 0

                elif move.id == "tailwind":
                    if (
                        SideCondition.TAILWIND not in battle.side_conditions
                        and our_speed <= opp_speed
                    ):
                        return Player.create_order(move), 0

                elif (
                    move.id in self._SCREEN_MOVES
                    and turn <= self.EARLY_GAME_TURN_LIMIT
                ):
                    condition = self._SCREEN_MOVES[move.id]
                    if condition not in battle.side_conditions:
                        return Player.create_order(move), 0

                elif (
                    move.id in self._WEATHER_MOVES
                    and turn <= self.EARLY_GAME_TURN_LIMIT
                ):
                    move_weather = self._WEATHER_MOVES[move.id]
                    if move_weather not in weather:
                        return Player.create_order(move), 0

        return SimpleHeuristicsPlayer.choose_singles_move(battle)
