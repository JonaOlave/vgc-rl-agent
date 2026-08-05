"""
All enums, index maps, and dimension constants used for battle embedding.
Import order matters: this module has zero internal dependencies.
"""

from poke_env.battle import (
    Field,
    MoveCategory,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
TYPES = [
    PokemonType.NORMAL,   PokemonType.FIRE,    PokemonType.WATER,
    PokemonType.ELECTRIC, PokemonType.GRASS,   PokemonType.ICE,
    PokemonType.FIGHTING, PokemonType.POISON,  PokemonType.GROUND,
    PokemonType.FLYING,   PokemonType.PSYCHIC, PokemonType.BUG,
    PokemonType.ROCK,     PokemonType.GHOST,   PokemonType.DRAGON,
    PokemonType.DARK,     PokemonType.STEEL,   PokemonType.FAIRY,
]
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}
N_TYPES = len(TYPES)  # 18

# ---------------------------------------------------------------------------
# Status conditions
# ---------------------------------------------------------------------------
STATUSES = [
    Status.BRN, Status.FRZ, Status.PAR,
    Status.PSN, Status.TOX, Status.SLP,
]
STATUS_INDEX = {s: i for i, s in enumerate(STATUSES)}
N_STATUSES = len(STATUSES)  # 6

# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
WEATHERS = [
    Weather.SUNNYDAY,
    Weather.RAINDANCE,
    Weather.SANDSTORM,
    Weather.HAIL,
    Weather.SNOW,
]
WEATHER_INDEX = {w: i for i, w in enumerate(WEATHERS)}
N_WEATHERS = len(WEATHERS)  # 5

# ---------------------------------------------------------------------------
# Terrain (subset of Field)
# ---------------------------------------------------------------------------
TERRAINS = [
    Field.ELECTRIC_TERRAIN,
    Field.GRASSY_TERRAIN,
    Field.MISTY_TERRAIN,
    Field.PSYCHIC_TERRAIN,
]
TERRAIN_INDEX = {t: i for i, t in enumerate(TERRAINS)}
N_TERRAINS = len(TERRAINS)  # 4

# ---------------------------------------------------------------------------
# Hazards and their max stack counts
# ---------------------------------------------------------------------------
HAZARDS = [
    SideCondition.STEALTH_ROCK,   # max 1
    SideCondition.SPIKES,         # max 3
    SideCondition.TOXIC_SPIKES,   # max 2
    SideCondition.STICKY_WEB,     # max 1
]
HAZARD_MAX_STACKS = [1, 3, 2, 1]
N_HAZARDS = len(HAZARDS)  # 4

# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------
SCREENS = [SideCondition.LIGHT_SCREEN, SideCondition.REFLECT]
N_SCREENS = len(SCREENS)  # 2

# ---------------------------------------------------------------------------
# Move categories
# ---------------------------------------------------------------------------
MOVE_CATEGORIES = [
    MoveCategory.PHYSICAL,
    MoveCategory.SPECIAL,
    MoveCategory.STATUS,
]
CATEGORY_INDEX = {c: i for i, c in enumerate(MOVE_CATEGORIES)}
N_CATEGORIES = len(MOVE_CATEGORIES)  # 3

# ---------------------------------------------------------------------------
# Stat keys (used for base_stats and boosts)
# ---------------------------------------------------------------------------
STAT_KEYS  = ["hp",  "atk", "def", "spa", "spd", "spe"]
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
N_STATS  = len(STAT_KEYS)   # 6
N_BOOSTS = len(BOOST_KEYS)  # 7

# ---------------------------------------------------------------------------
# Normalization ceilings
# ---------------------------------------------------------------------------
MAX_BASE_STAT  = 255   # Blissey HP
MAX_BASE_POWER = 250   # highest realistic base power

# ---------------------------------------------------------------------------
# Observation space layout
# ---------------------------------------------------------------------------
#
# Per-Pokemon vector (POKEMON_DIM = 59):
#   [0]       hp_fraction
#   [1-18]    type1 one-hot (18)
#   [19-36]   type2 one-hot (18)
#   [37-42]   status one-hot (6)
#   [43-48]   base_stats normalized (6)
#   [49-55]   boosts normalized -1..+1 (7)
#   [56]      is_active
#   [57]      is_fainted
#   [58]      is_unknown
#
POKEMON_DIM = 1 + N_TYPES + N_TYPES + N_STATUSES + N_STATS + N_BOOSTS + 3  # 59

#
# Per-Move vector (MOVE_DIM = 24):
#   [0]       base_power normalized
#   [1-18]    type one-hot (18)
#   [19-21]   category one-hot (3)
#   [22]      pp_fraction
#   [23]      is_available
#
MOVE_DIM = 1 + N_TYPES + N_CATEGORIES + 1 + 1  # 24

N_POKEMON_PER_SIDE = 6
N_SIDES            = 2   # ally + opponent
N_MOVES            = 4   # max moves per Pokemon

#
# Field vector (FIELD_DIM = 21):
#   [0-4]    weather one-hot (5)
#   [5-8]    terrain one-hot (4)
#   [9-12]   own hazards normalized (4)
#   [13-16]  opp hazards normalized (4)
#   [17-18]  own screens (2)
#   [19-20]  opp screens (2)
#
FIELD_DIM = N_WEATHERS + N_TERRAINS + N_HAZARDS * 2 + N_SCREENS * 2  # 21

POKEMON_BLOCK = N_POKEMON_PER_SIDE * N_SIDES * POKEMON_DIM  # 708
MOVE_BLOCK    = N_MOVES * MOVE_DIM                           # 96

OBS_SIZE = POKEMON_BLOCK + MOVE_BLOCK + FIELD_DIM  # 825

# Gen 9 action space (from SinglesEnv.get_action_space_size):
#   0-5   switch to Pokemon slot 0-5
#   6-9   move 0-3  (no gimmick)
#   10-13 move 0-3  + mega evolve
#   14-17 move 0-3  + z-move
#   18-21 move 0-3  + dynamax
#   22-25 move 0-3  + terastallize
N_ACTIONS_GEN9 = 26
