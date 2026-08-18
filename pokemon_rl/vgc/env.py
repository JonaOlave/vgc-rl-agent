"""
VGCEnv — subclase de DoublesEnv para VGC / Random Doubles.

Dos modos de uso
----------------
1. Sin equipo (Random Doubles Battle) — ideal para empezar a entrenar:

    env = make_vgc_env(battle_format="gen9randomdoublesbattle")

2. Con equipo VGC real:

    from pokemon_rl.vgc.team import SAMPLE_TEAM_REG_H
    env = make_vgc_env(
        battle_format="gen9vgc2025regg",
        team=SAMPLE_TEAM_REG_H,
    )

Espacio de observación
-----------------------
    Dict({
        "observation": Box(-1, 1, shape=(925,), float32),
        "action_mask": Box(0, 1, shape=(214,), int64),  # 107 × 2
    })

Espacio de acción
-----------------
    MultiDiscrete([107, 107])
    — una acción por cada Pokémon activo.
"""

import random as _random
from typing import Any, List, Optional, Union

import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from poke_env import LocalhostServerConfiguration
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move import SPECIAL_MOVES, MoveCategory, Target
from poke_env.environment import DoublesEnv, SingleAgentWrapper
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder, DoubleBattleOrder
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from .constants import N_ACTIONS_DOUBLES_GEN9, VGC_OBS_SIZE
from .embedding import embed_battle as _embed
from .opponents import SupportAwareHeuristicsPlayer
from .teambuilder import RandomTeamPool


class VGCEnv(DoublesEnv):
    """
    Entorno de batalla VGC doubles listo para RL.

    Hereda la lógica de protocolo y sincronización de DoublesEnv.
    Implementa embed_battle() y calc_reward().
    """

    def __init__(
        self,
        opponent_team_pool: Optional[List[str]] = None,
        own_team_pool: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: spaces.Box(
                low=-1.0, high=1.0, shape=(VGC_OBS_SIZE,), dtype=np.float32
            )
            for agent in self.possible_agents
        }
        # PokeEnv construye agent1 (nosotros) y agent2 (rival) con el mismo
        # `team` — por diseño, ambos lados de la batalla comparten un único
        # Teambuilder. Reemplazamos el Teambuilder de cada lado por separado
        # después de la inicialización según lo que se haya pedido:
        #   - opponent_team_pool → sólo el rival varía (self sigue fijo en `team`)
        #   - own_team_pool      → nuestro propio lado también sortea equipo
        #     por batalla (self-play: aprender a pilotar cualquier equipo del
        #     pool, no memorizar uno solo). Si se pasan ambos con el mismo
        #     pool, cada lado sortea su equipo de forma independiente — no
        #     necesariamente terminan igual.
        if opponent_team_pool:
            self.agent2._team = RandomTeamPool(opponent_team_pool)
        if own_team_pool:
            self.agent1._team = RandomTeamPool(own_team_pool)

    def embed_battle(self, battle: Any) -> np.ndarray:
        """Convierte el estado doubles en un vector float32 de 925 valores."""
        return _embed(battle)

    def calc_reward(self, battle: Any) -> float:
        """
        Recompensa diferencial para doubles VGC.

        En VGC importa más el número de Pokémon que sobreviven que el HP,
        por eso fainted_value es el peso dominante.

        Pesos:
          fainted_value  → 0.5  por Pokémon derribado (doble que singles)
          hp_value       → 0.1  por HP proporcional
          victory_value  → 1.0  por ganar el combate
        """
        return self.reward_computing_helper(
            battle,
            fainted_value=0.5,
            hp_value=0.1,
            victory_value=1.0,
        )

    @staticmethod
    def action_to_order(
        action: "npt.NDArray[np.int64]",
        battle: DoubleBattle,
        fake: bool = False,
        strict: bool = False,
    ) -> BattleOrder:
        # Standard PPO ignores the action mask, so during forced-switch turns the
        # agent regularly outputs invalid actions (pass / move when a switch is
        # required).  The PS server then rejects with [Invalid choice], poke-env
        # fires _trying_again, and the same invalid state loops forever.
        # Detect the force-switch case here and pick a valid combination directly,
        # bypassing the (useless) agent action entirely.
        if (
            not battle.teampreview
            and not battle._wait
            and any(battle.force_switch)
        ):
            valid = DoubleBattleOrder.join_orders(*battle.valid_orders)
            if valid:
                return valid[int(_random.random() * len(valid))]
            return DefaultBattleOrder()

        action = VGCEnv._normalize_targets(action, battle)
        return DoublesEnv.action_to_order(action, battle, fake=fake, strict=strict)

    # Targets that mean "no specific Pokemon to pick" — Trick Room, Protect,
    # Tailwind, screens, Substitute, self-boosts, etc. Mirrors the dict in
    # poke_env.battle.double_battle.DoubleBattle.get_possible_showdown_targets,
    # minus the ownership pre-check (see _normalize_targets below for why).
    _NO_TARGET_MOVE_TARGETS = frozenset(
        {
            Target.ALL,
            Target.ALL_ADJACENT,
            Target.ALL_ADJACENT_FOES,
            Target.ALLIES,
            Target.ALLY_SIDE,
            Target.ALLY_TEAM,
            Target.FOE_SIDE,
            Target.SELF,
            Target.RANDOM_NORMAL,
            Target.SCRIPTED,
        }
    )

    @staticmethod
    def _normalize_targets(
        action: "npt.NDArray[np.int64]", battle: DoubleBattle
    ) -> "npt.NDArray[np.int64]":
        # Same root cause as the force-switch case above: standard PPO ignores
        # the action mask, so it regularly samples a target offset that isn't
        # valid for the chosen move. This matters a lot for moves that don't
        # target a specific Pokemon (Trick Room, Protect, Tailwind, screens,
        # Substitute, etc.) — poke-env only accepts target=0 (no target) for
        # those, so any other sampled target value silently invalidates the
        # whole order and the move never gets executed even when the agent
        # "chose" it correctly on the move-index axis.
        #
        # We'd normally ask battle.get_possible_showdown_targets() for the
        # valid target list, but it requires the move to be in
        # battle.available_moves[pos] at the exact instant it's called, which
        # isn't reliably true from here (it raises "not owned by any active
        # ally Pokemon" even for the Pokemon's own known moves). So instead we
        # replicate the no-target-move classification directly from
        # move.deduced_target, which is a static property of the move and
        # doesn't have that timing dependency.
        normalized = action.copy()
        for pos in (0, 1):
            a = int(action[pos])
            if a < 7:
                continue
            active_mon = battle.active_pokemon[pos]
            if active_mon is None:
                continue
            move_idx = (a - 7) % 20 // 5
            gimmick = (a - 7) // 20
            target = (a - 7) % 5 - 2
            if target == 0:
                continue  # already untargeted, nothing to fix

            known_moves = list(active_mon.moves.values())[:4]
            if move_idx >= len(known_moves):
                continue
            move = known_moves[move_idx]

            is_dynamaxing = gimmick == 3 or active_mon.is_dynamaxed
            no_target = move.id not in SPECIAL_MOVES and (
                (is_dynamaxing and move.category == MoveCategory.STATUS)
                or (not is_dynamaxing and move.deduced_target in VGCEnv._NO_TARGET_MOVE_TARGETS)
            )
            if no_target:
                normalized[pos] = 7 + 5 * move_idx + 2 + 20 * gimmick  # target=0
        return normalized


def make_vgc_env(
    battle_format: str = "gen9randomdoublesbattle",
    server_host: str = "localhost",
    server_port: int = 8000,
    team: Optional[str] = None,
    opponent: str = "random",
    strict: bool = False,
    opponent_team_pool: Optional[List[str]] = None,
    own_team_pool: Optional[List[str]] = None,
) -> SingleAgentWrapper:
    """
    Construye el entorno VGC Gymnasium listo para entrenar.

    Parameters
    ----------
    battle_format : str
        - ``"gen9randomdoublesbattle"``  → no requiere equipo (recomendado para empezar)
        - ``"gen9vgc2025regg"``          → requiere pasar `team`
    server_host / server_port : str / int
        Dirección del servidor PS local.
    team : str, optional
        Equipo en formato Showdown. Requerido para formatos VGC.
        Usa ``pokemon_rl.vgc.team.SAMPLE_TEAM_REG_H`` como punto de partida.
        Ignorado para nuestro lado si se pasa ``own_team_pool``.
    opponent : str
        ``"random"`` → RandomPlayer  |  ``"heuristic"`` → SimpleHeuristicsPlayer  |
        ``"support_heuristic"`` → SupportAwareHeuristicsPlayer (igual que heuristic,
        pero sí usa Trick Room/Tailwind/pantallas/clima — ver ``vgc/opponents.py``)
    strict : bool
        Si True lanza error en acciones inválidas; si False usa fallback aleatorio.
    opponent_team_pool : list of str, optional
        Si se pasa, el oponente elige un equipo al azar del pool en cada
        batalla (ver ``RandomTeamPool``) en vez de usar siempre `team`.
    own_team_pool : list of str, optional
        Si se pasa, **nuestro propio agente** también elige un equipo al azar
        del pool en cada batalla, en vez de usar siempre `team` fijo —
        self-play: el modelo aprende a pilotar cualquier equipo del pool, no
        solo uno memorizado. Independiente de `opponent_team_pool` — cada
        lado sortea el suyo por separado, no quedan sincronizados.

    Returns
    -------
    SingleAgentWrapper
        gymnasium.Env estándar con:
          observation_space = Dict({"observation": Box(925,), "action_mask": Box(214,)})
          action_space      = MultiDiscrete([107, 107])
    """
    server_config = ServerConfiguration(
        websocket_url=f"ws://{server_host}:{server_port}/showdown/websocket",
        authentication_url="https://play.pokemonshowdown.com/action.php?",
    )

    raw_env = VGCEnv(
        battle_format=battle_format,
        server_configuration=server_config,
        team=team,
        start_listening=True,
        strict=strict,
        choose_on_teampreview=False,
        opponent_team_pool=opponent_team_pool,
        own_team_pool=own_team_pool,
    )

    # NOTA: el `team` que se le pasa acá al objeto `opponent` no tiene efecto
    # en qué equipo usa el rival — SingleAgentWrapper solo usa este objeto
    # para decidir movimientos (choose_move/teampreview); el equipo real de
    # ambos lados lo asigna VGCEnv/PokeEnv internamente (agent1/agent2, ver
    # arriba). Se deja `team=team` por consistencia, no por necesidad.
    if opponent == "heuristic":
        opp = SimpleHeuristicsPlayer(
            battle_format=battle_format,
            server_configuration=server_config,
            team=team,
            account_configuration=AccountConfiguration.generate("VGCOpponent", rand=True),
        )
    elif opponent == "support_heuristic":
        opp = SupportAwareHeuristicsPlayer(
            battle_format=battle_format,
            server_configuration=server_config,
            team=team,
            account_configuration=AccountConfiguration.generate("VGCOpponent", rand=True),
        )
    else:
        opp = RandomPlayer(
            battle_format=battle_format,
            server_configuration=server_config,
            team=team,
            account_configuration=AccountConfiguration.generate("VGCOpponent", rand=True),
        )

    return SingleAgentWrapper(raw_env, opp)
