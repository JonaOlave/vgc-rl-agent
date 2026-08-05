"""
Constantes específicas para el entorno VGC Doubles.
Reutiliza los enums y dimensiones de pokemon_rl.constants y agrega
las particularidades de doubles (2 activos por lado, bloque VGC).
"""

# Reutilizar todo de singles
from pokemon_rl.constants import (
    BOOST_KEYS,
    CATEGORY_INDEX,
    FIELD_DIM,
    HAZARD_MAX_STACKS,
    HAZARDS,
    MAX_BASE_POWER,
    MAX_BASE_STAT,
    MOVE_DIM,
    N_BOOSTS,
    N_CATEGORIES,
    N_POKEMON_PER_SIDE,
    N_STATS,
    N_STATUSES,
    N_TERRAINS,
    N_TYPES,
    N_WEATHERS,
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
# Doubles-specific layout
# ---------------------------------------------------------------------------
N_ACTIVE_PER_SIDE = 2   # 2 Pokémon activos por lado en doubles
N_MOVES_PER_SLOT = 4    # 4 movimientos por Pokémon activo

# Bloque de Pokémon: misma estructura que singles (12 × 59 = 708)
# Orden:  ally_active[0], ally_active[1], ally_bench[0-3],
#         opp_active[0],  opp_active[1],  opp_bench[0-3]
POKEMON_BLOCK_VGC = N_POKEMON_PER_SIDE * 2 * POKEMON_DIM  # 708

# Bloque de movimientos: 2 activos × 4 movimientos × 24 = 192
MOVE_BLOCK_VGC = N_ACTIVE_PER_SIDE * N_MOVES_PER_SLOT * MOVE_DIM  # 192

# Bloque VGC-específico (4 valores):
#   [0] can_tera  posición 0
#   [1] can_tera  posición 1
#   [2] force_switch posición 0
#   [3] force_switch posición 1
VGC_EXTRA_DIM = 4

# Tamaño total del vector de observación
# 708 + 192 + 21 + 4 = 925
VGC_OBS_SIZE = POKEMON_BLOCK_VGC + MOVE_BLOCK_VGC + FIELD_DIM + VGC_EXTRA_DIM

# Action space: MultiDiscrete([107, 107]) para gen9
# Cada elemento:
#   0          → pass
#   1-6        → switch (slot 0-5 del equipo)
#   7-106      → movimiento + objetivo + gimmick
#     (action - 7) % 20 // 5   → índice de movimiento (0-3)
#     (action - 7) % 5 - 2     → objetivo (-2, -1, 0, 1, 2)
#     (action - 7) // 20        → gimmick (0=nada, 1=mega, 2=z, 3=dmax, 4=tera)
N_ACTIONS_DOUBLES_GEN9 = 107
