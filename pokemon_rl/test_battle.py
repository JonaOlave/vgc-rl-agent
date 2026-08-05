"""
Prueba de conexión al servidor PS y una batalla completa.

Uso:
    source pokemon_rl/.venv/bin/activate
    python -m pokemon_rl.test_battle

Requiere el servidor PS corriendo. Si no lo tienes instalado:
    cd ~
    git clone https://github.com/smogon/pokemon-showdown.git
    cd pokemon-showdown
    node pokemon-showdown start --no-security
"""

import asyncio
import logging

from poke_env import LocalhostServerConfiguration
from poke_env.player import RandomPlayer


async def run_test_battle(n_battles: int = 3):
    """Corre N batallas entre dos RandomPlayer y muestra resultados."""

    logging.basicConfig(level=logging.WARNING)
    print(f"Conectando al servidor en {LocalhostServerConfiguration.websocket_url}...")

    p1 = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        log_level=logging.WARNING,
    )
    p2 = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        log_level=logging.WARNING,
    )

    print(f"Corriendo {n_battles} batallas de prueba...")
    await p1.battle_against(p2, n_battles=n_battles)

    print("\n=== Resultados ===")
    print(f"Jugador 1 ({p1.username}): {p1.n_won_battles} victorias / {p1.n_finished_battles} batallas")
    print(f"Jugador 2 ({p2.username}): {p2.n_won_battles} victorias / {p2.n_finished_battles} batallas")

    # Muestra el embedding de la última batalla
    if p1.battles:
        from pokemon_rl.embedding import embed_battle
        last = list(p1.battles.values())[-1]
        obs = embed_battle(last)
        print(f"\n=== Embedding de la última batalla ===")
        print(f"Shape: {obs.shape}")
        print(f"Rango: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"Valores no-cero: {(obs != 0).sum()} / {obs.size}")
        print(f"Primeros 10 valores: {obs[:10]}")
        print("\n✓ El embedding funciona correctamente.")


if __name__ == "__main__":
    asyncio.run(run_test_battle())
