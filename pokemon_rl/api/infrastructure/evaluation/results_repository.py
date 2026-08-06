import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pokemon_rl.api.domain.evaluation.entities import BattleResult, Evaluation

_DATA_PATH = Path("data/evaluation_results.json")

_SEED_DATA: List[dict] = [
    {
        "id": "hist-1",
        "timestamp": "2026-08-03T10:00:00",
        "model_path": "models/vgc/final",
        "opponent": "random",
        "n_battles": 50,
        "wins": 39, "losses": 11, "draws": 0,
        "win_rate": 0.78,
        "mean_reward": 1.54, "std_reward": 1.2,
        "mean_our_fainted": 1.82, "mean_opp_fainted": 3.54,
        "battles": [],
    },
    {
        "id": "hist-2",
        "timestamp": "2026-08-04T15:00:00",
        "model_path": "models/vgc/final",
        "opponent": "random",
        "n_battles": 50,
        "wins": 47, "losses": 3, "draws": 0,
        "win_rate": 0.94,
        "mean_reward": 2.1, "std_reward": 0.9,
        "mean_our_fainted": 1.2, "mean_opp_fainted": 4.1,
        "battles": [],
    },
    {
        "id": "hist-3",
        "timestamp": "2026-08-05T11:00:00",
        "model_path": "models/vgc/final",
        "opponent": "heuristic",
        "n_battles": 30,
        "wins": 10, "losses": 20, "draws": 0,
        "win_rate": 0.333,
        "mean_reward": -0.78, "std_reward": 1.9,
        "mean_our_fainted": 3.53, "mean_opp_fainted": 2.77,
        "battles": [],
    },
    {
        "id": "hist-4",
        "timestamp": "2026-08-05T16:00:00",
        "model_path": "models/vgc/final",
        "opponent": "heuristic",
        "n_battles": 30,
        "wins": 12, "losses": 18, "draws": 0,
        "win_rate": 0.40,
        "mean_reward": -0.36, "std_reward": 1.79,
        "mean_our_fainted": 3.40, "mean_opp_fainted": 3.10,
        "battles": [],
    },
    {
        "id": "hist-5",
        "timestamp": "2026-08-06T14:26:51",
        "model_path": "models/vgc/final",
        "opponent": "heuristic",
        "n_battles": 30,
        "wins": 20, "losses": 10, "draws": 0,
        "win_rate": 0.667,
        "mean_reward": 0.61, "std_reward": 1.86,
        "mean_our_fainted": 2.93, "mean_opp_fainted": 3.43,
        "battles": [
            {"battle_num": 1,  "result": "WIN",  "reward":  2.10, "our_fainted": 2, "opp_fainted": 4},
            {"battle_num": 2,  "result": "WIN",  "reward":  2.65, "our_fainted": 1, "opp_fainted": 4},
            {"battle_num": 3,  "result": "WIN",  "reward":  1.55, "our_fainted": 3, "opp_fainted": 4},
            {"battle_num": 4,  "result": "WIN",  "reward":  1.56, "our_fainted": 3, "opp_fainted": 4},
            {"battle_num": 5,  "result": "LOSS", "reward": -2.15, "our_fainted": 4, "opp_fainted": 2},
            {"battle_num": 6,  "result": "LOSS", "reward": -2.11, "our_fainted": 4, "opp_fainted": 2},
            {"battle_num": 7,  "result": "LOSS", "reward": -2.74, "our_fainted": 4, "opp_fainted": 1},
            {"battle_num": 8,  "result": "WIN",  "reward":  1.54, "our_fainted": 3, "opp_fainted": 4},
            {"battle_num": 9,  "result": "WIN",  "reward":  2.69, "our_fainted": 1, "opp_fainted": 4},
            {"battle_num": 10, "result": "WIN",  "reward":  1.53, "our_fainted": 3, "opp_fainted": 4},
            {"battle_num": 11, "result": "WIN",  "reward":  1.60, "our_fainted": 3, "opp_fainted": 4},
            {"battle_num": 12, "result": "LOSS", "reward": -1.56, "our_fainted": 4, "opp_fainted": 3},
            {"battle_num": 13, "result": "WIN",  "reward":  1.60, "our_fainted": 3, "opp_fainted": 4},
            {"battle_num": 14, "result": "WIN",  "reward":  1.56, "our_fainted": 3, "opp_fainted": 4},
            {"battle_num": 15, "result": "WIN",  "reward":  2.66, "our_fainted": 1, "opp_fainted": 4},
            {"battle_num": 16, "result": "LOSS", "reward": -1.51, "our_fainted": 4, "opp_fainted": 3},
            {"battle_num": 17, "result": "WIN",  "reward":  2.08, "our_fainted": 2, "opp_fainted": 4},
            {"battle_num": 18, "result": "WIN",  "reward":  2.10, "our_fainted": 2, "opp_fainted": 4},
            {"battle_num": 19, "result": "LOSS", "reward": -2.70, "our_fainted": 4, "opp_fainted": 1},
            {"battle_num": 20, "result": "WIN",  "reward":  1.52, "our_fainted": 3, "opp_fainted": 4},
            {"battle_num": 21, "result": "LOSS", "reward": -2.12, "our_fainted": 4, "opp_fainted": 2},
            {"battle_num": 22, "result": "WIN",  "reward":  2.15, "our_fainted": 2, "opp_fainted": 4},
            {"battle_num": 23, "result": "LOSS", "reward": -1.60, "our_fainted": 4, "opp_fainted": 3},
            {"battle_num": 24, "result": "WIN",  "reward":  1.57, "our_fainted": 3, "opp_fainted": 4},
            {"battle_num": 25, "result": "LOSS", "reward": -1.50, "our_fainted": 4, "opp_fainted": 3},
            {"battle_num": 26, "result": "WIN",  "reward":  1.60, "our_fainted": 3, "opp_fainted": 4},
            {"battle_num": 27, "result": "LOSS", "reward": -1.54, "our_fainted": 4, "opp_fainted": 3},
            {"battle_num": 28, "result": "WIN",  "reward":  2.10, "our_fainted": 2, "opp_fainted": 4},
            {"battle_num": 29, "result": "WIN",  "reward":  1.50, "our_fainted": 3, "opp_fainted": 4},
            {"battle_num": 30, "result": "WIN",  "reward":  2.17, "our_fainted": 2, "opp_fainted": 4},
        ],
    },
]


def _to_entity(d: dict) -> Evaluation:
    return Evaluation(
        id=d["id"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
        model_path=d["model_path"],
        opponent=d["opponent"],
        n_battles=d["n_battles"],
        wins=d["wins"],
        losses=d["losses"],
        draws=d["draws"],
        win_rate=d["win_rate"],
        mean_reward=d["mean_reward"],
        std_reward=d["std_reward"],
        mean_our_fainted=d["mean_our_fainted"],
        mean_opp_fainted=d["mean_opp_fainted"],
        battles=[BattleResult(**b) for b in d.get("battles", [])],
    )


def _to_dict(e: Evaluation) -> dict:
    return {
        "id": e.id,
        "timestamp": e.timestamp.isoformat(),
        "model_path": e.model_path,
        "opponent": e.opponent,
        "n_battles": e.n_battles,
        "wins": e.wins,
        "losses": e.losses,
        "draws": e.draws,
        "win_rate": e.win_rate,
        "mean_reward": e.mean_reward,
        "std_reward": e.std_reward,
        "mean_our_fainted": e.mean_our_fainted,
        "mean_opp_fainted": e.mean_opp_fainted,
        "battles": [
            {
                "battle_num": b.battle_num,
                "result": b.result,
                "reward": b.reward,
                "our_fainted": b.our_fainted,
                "opp_fainted": b.opp_fainted,
            }
            for b in e.battles
        ],
    }


class ResultsRepository:
    def __init__(self, data_path: Path = _DATA_PATH):
        self.data_path = data_path
        if not self.data_path.exists():
            self._write({"evaluations": _SEED_DATA})

    def _read(self) -> dict:
        with open(self.data_path) as f:
            return json.load(f)

    def _write(self, data: dict) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def list_all(self) -> List[Evaluation]:
        return [_to_entity(d) for d in self._read()["evaluations"]]

    def find_by_id(self, eval_id: str) -> Optional[Evaluation]:
        for d in self._read()["evaluations"]:
            if d["id"] == eval_id:
                return _to_entity(d)
        return None

    def save(self, evaluation: Evaluation) -> None:
        data = self._read()
        data["evaluations"].append(_to_dict(evaluation))
        self._write(data)
