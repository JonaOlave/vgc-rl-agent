from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass(frozen=True)
class BattleResult:
    battle_num: int
    result: str  # "WIN" | "LOSS" | "DRAW"
    reward: float
    our_fainted: int
    opp_fainted: int


@dataclass
class Evaluation:
    id: str
    timestamp: datetime
    model_path: str
    opponent: str
    n_battles: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    mean_reward: float
    std_reward: float
    mean_our_fainted: float
    mean_opp_fainted: float
    battles: List[BattleResult] = field(default_factory=list)
