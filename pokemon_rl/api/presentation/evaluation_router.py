from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pokemon_rl.api.application.evaluation.commands import RunEvaluationCommand, run_evaluation
from pokemon_rl.api.application.evaluation.queries import get_evaluations, get_evaluation
from pokemon_rl.api.infrastructure.evaluation.sqlite_results_repository import SqliteResultsRepository
from pokemon_rl.api.infrastructure.evaluation.runner import EvaluationRunner

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])
_results_repo = SqliteResultsRepository()
_runner = EvaluationRunner()


class RunEvaluationRequest(BaseModel):
    model_path: str = "models/vgc/final"
    opponent: str = "random"
    n_battles: int = 30
    use_champions_team: bool = True


def _eval_to_dict(e) -> dict:
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
        "round_number": e.round_number,
        "our_team": e.our_team,
        "opponent_archetype": e.opponent_archetype,
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


@router.get("/results")
def list_results():
    return [_eval_to_dict(e) for e in get_evaluations(_results_repo)]


@router.get("/results/{eval_id}")
def get_result(eval_id: str):
    ev = get_evaluation(_results_repo, eval_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return _eval_to_dict(ev)


@router.post("/run")
def run_eval(body: RunEvaluationRequest):
    cmd = RunEvaluationCommand(
        model_path=body.model_path,
        opponent=body.opponent,
        n_battles=body.n_battles,
        use_champions_team=body.use_champions_team,
    )
    ev = run_evaluation(cmd, _results_repo, _runner)
    return _eval_to_dict(ev)
