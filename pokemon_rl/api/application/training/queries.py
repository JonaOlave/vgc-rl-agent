from typing import List, Protocol

from pokemon_rl.api.domain.training.entities import Checkpoint


class ICheckpointRepository(Protocol):
    def list_all(self) -> List[Checkpoint]: ...


def get_checkpoints(repo: ICheckpointRepository) -> List[Checkpoint]:
    return sorted(repo.list_all(), key=lambda c: c.steps)
