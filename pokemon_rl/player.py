"""
En poke-env 0.15, el bridge async/sync está integrado en PokeEnv mediante
_EnvPlayer + _AsyncQueue. Ya no es necesario un Player personalizado.

Este módulo existe como referencia para crear jugadores adicionales
(por ejemplo, para self-play o evaluación).

Referencia de acción para SinglesEnv gen9 (26 acciones):
  0-5   → cambiar al Pokémon en el slot 0-5 del equipo
  6-9   → usar movimiento slot 0-3 (sin gimmick)
  10-13 → usar movimiento slot 0-3 + mega evolución
  14-17 → usar movimiento slot 0-3 + z-move
  18-21 → usar movimiento slot 0-3 + dynamax
  22-25 → usar movimiento slot 0-3 + terastalización
"""

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer

__all__ = ["RandomPlayer", "SimpleHeuristicsPlayer"]
