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
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from poke_env import LocalhostServerConfiguration
from poke_env.battle.double_battle import DoubleBattle
from poke_env.environment import DoublesEnv, SingleAgentWrapper
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder, DoubleBattleOrder
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from .constants import N_ACTIONS_DOUBLES_GEN9, VGC_OBS_SIZE
from .embedding import embed_battle as _embed


class VGCEnv(DoublesEnv):
    """
    Entorno de batalla VGC doubles listo para RL.

    Hereda la lógica de protocolo y sincronización de DoublesEnv.
    Implementa embed_battle() y calc_reward().
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: spaces.Box(
                low=-1.0, high=1.0, shape=(VGC_OBS_SIZE,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

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

        return DoublesEnv.action_to_order(action, battle, fake=fake, strict=strict)


def make_vgc_env(
    battle_format: str = "gen9randomdoublesbattle",
    server_host: str = "localhost",
    server_port: int = 8000,
    team: Optional[str] = None,
    opponent: str = "random",
    strict: bool = False,
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
    opponent : str
        ``"random"`` → RandomPlayer  |  ``"heuristic"`` → SimpleHeuristicsPlayer
    strict : bool
        Si True lanza error en acciones inválidas; si False usa fallback aleatorio.

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
    )

    if opponent == "heuristic":
        opp = SimpleHeuristicsPlayer(
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
