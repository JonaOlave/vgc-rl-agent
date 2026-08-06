from typing import List, Optional, Protocol

from pokemon_rl.api.domain.evaluation.entities import Evaluation


class IEvaluationRepository(Protocol):
    def list_all(self) -> List[Evaluation]: ...
    def find_by_id(self, eval_id: str) -> Optional[Evaluation]: ...
    def save(self, evaluation: Evaluation) -> None: ...


def get_evaluations(repo: IEvaluationRepository) -> List[Evaluation]:
    return sorted(repo.list_all(), key=lambda e: e.timestamp)


def get_evaluation(repo: IEvaluationRepository, eval_id: str) -> Optional[Evaluation]:
    return repo.find_by_id(eval_id)
