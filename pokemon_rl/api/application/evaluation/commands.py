from dataclasses import dataclass

from pokemon_rl.api.domain.evaluation.entities import Evaluation
from pokemon_rl.api.application.evaluation.queries import IEvaluationRepository


@dataclass
class RunEvaluationCommand:
    model_path: str
    opponent: str
    n_battles: int
    use_champions_team: bool = True


def run_evaluation(
    cmd: RunEvaluationCommand,
    results_repo: IEvaluationRepository,
    runner,
) -> Evaluation:
    evaluation = runner.run(cmd)
    results_repo.save(evaluation)
    return evaluation
