"""
PokemonRL — subclase de SinglesEnv de poke-env 0.15.

Arquitectura real de poke-env 0.15
------------------------------------
  SinglesEnv (PettingZoo multi-agent)
    └── battle_queue / order_queue  ← async bridge interno
         ↕
  SingleAgentWrapper  ← convierte a gymnasium.Env de un solo agente
         ↕
  Stable-Baselines3 / cualquier bucle de entrenamiento

No es necesario implementar el bridge async/sync manualmente:
poke-env 0.15 ya lo maneja con _EnvPlayer + _AsyncQueue.

Lo único que debemos implementar son:
  • embed_battle(battle) → np.ndarray    (nuestro embedding de 825 valores)
  • calc_reward(battle)  → float         (recompensa diferencial)

El resto (action_to_order, get_action_mask, reset, step) es heredado.

Uso básico
----------
    from pokemon_rl.env import make_env

    env = make_env()
    obs, info = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

    env.close()
"""

from typing import Any, Optional, Union

import numpy as np
from gymnasium import spaces

from poke_env import LocalhostServerConfiguration
from poke_env.battle import AbstractBattle  # type: ignore[attr-defined]
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import RandomPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from .constants import OBS_SIZE
from .embedding import embed_battle as _embed


class PokemonRL(SinglesEnv):
    """
    Entorno de batalla Pokemon Showdown listo para RL.

    Hereda toda la lógica de protocolo y sincronización de SinglesEnv.
    Solo sobreescribimos la función de embedding y la de recompensa.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Registrar el espacio de observación crudo.
        # PokeEnv.__setattr__ lo envuelve automáticamente en
        # Dict({"observation": <raw>, "action_mask": Box(N_ACTIONS)})
        self.observation_spaces = {
            agent: spaces.Box(
                low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

    # ------------------------------------------------------------------
    # Métodos abstractos requeridos por SinglesEnv / PokeEnv
    # ------------------------------------------------------------------

    def embed_battle(self, battle: Any) -> np.ndarray:
        """Convierte el estado de la batalla en un vector float32 de 825 valores."""
        return _embed(battle)

    def calc_reward(self, battle: Any) -> float:
        """
        Recompensa diferencial usando el helper integrado de poke-env.

        El helper rastrea el estado anterior internamente (_reward_buffer),
        por lo que no necesitamos guardar prev_battle manualmente.

        Pesos:
          fainted_value  → 0.3  por Pokémon derribado
          hp_value       → 0.1  por punto de HP proporcional
          victory_value  → 1.0  por ganar/perder el combate
        """
        return self.reward_computing_helper(
            battle,
            fainted_value=0.3,
            hp_value=0.1,
            victory_value=1.0,
        )


def make_env(
    battle_format: str = "gen9randombattle",
    server_host: str = "localhost",
    server_port: int = 8000,
    strict: bool = False,
) -> SingleAgentWrapper:
    """
    Construye el entorno Gymnasium listo para entrenar.

    Parameters
    ----------
    battle_format : str
        Formato de batalla PS. "gen9randombattle" no requiere teambuilder.
    server_host : str
        Host del servidor PS local.
    server_port : int
        Puerto del servidor PS local.
    strict : bool
        Si True lanza error ante acciones inválidas; si False elige un movimiento
        aleatorio válido como fallback. Recomendado False durante entrenamiento.

    Returns
    -------
    SingleAgentWrapper
        Un gymnasium.Env estándar. La observación devuelta es un dict:
          {"observation": np.ndarray(825,), "action_mask": np.ndarray(26,)}
    """
    server_config = ServerConfiguration(
        websocket_url=f"ws://{server_host}:{server_port}/showdown/websocket",
        authentication_url="https://play.pokemonshowdown.com/action.php?",
    )

    raw_env = PokemonRL(
        battle_format=battle_format,
        server_configuration=server_config,
        start_listening=True,
        strict=strict,
        choose_on_teampreview=False,
    )

    opponent = RandomPlayer(
        battle_format=battle_format,
        server_configuration=server_config,
        account_configuration=AccountConfiguration.generate("RLOpponent", rand=True),
    )

    return SingleAgentWrapper(raw_env, opponent)
