from fastapi import APIRouter
from typing import List

from pokemon_rl.api.application.training.queries import get_checkpoints
from pokemon_rl.api.infrastructure.training.checkpoint_repository import CheckpointRepository

router = APIRouter(prefix="/api/training", tags=["training"])
_repo = CheckpointRepository()


@router.get("/checkpoints")
def list_checkpoints():
    checkpoints = get_checkpoints(_repo)
    return [
        {
            "name": c.name,
            "steps": c.steps,
            "timestamp": c.timestamp.isoformat(),
            "size_mb": c.size_mb,
            "path": c.path,
        }
        for c in checkpoints
    ]
