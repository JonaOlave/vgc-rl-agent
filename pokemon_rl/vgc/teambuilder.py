"""
Teambuilder que elige un equipo al azar de un pool fijo en cada batalla.

poke-env llama a yield_team() cada vez que necesita un equipo nuevo
(una vez por batalla), así que usar esto como `team=` del oponente hace
que cada batalla de entrenamiento enfrente un equipo distinto en vez de
ser siempre el mismo mirror match.
"""

import random as _random
from typing import List

from poke_env.teambuilder import Teambuilder


class RandomTeamPool(Teambuilder):
    def __init__(self, teams: List[str]):
        self.teams = teams

    def yield_team(self) -> str:
        team = _random.choice(self.teams)
        return self.join_team(self.parse_showdown_team(team))
