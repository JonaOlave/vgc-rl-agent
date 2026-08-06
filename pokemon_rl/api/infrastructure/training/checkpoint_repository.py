import os
import re
from datetime import datetime
from pathlib import Path
from typing import List

from pokemon_rl.api.domain.training.entities import Checkpoint

_MODELS_DIR = Path("models/vgc")
_STEPS_RE = re.compile(r"vgc_ppo_(\d+)_steps")


class CheckpointRepository:
    def __init__(self, models_dir: Path = _MODELS_DIR):
        self.models_dir = models_dir

    def list_all(self) -> List[Checkpoint]:
        if not self.models_dir.exists():
            return []

        checkpoints: List[Checkpoint] = []
        for zip_path in self.models_dir.glob("*.zip"):
            name = zip_path.stem
            match = _STEPS_RE.match(name)
            if match:
                steps = int(match.group(1))
            elif name == "final":
                steps = -1  # sentinel — shown last
            else:
                continue

            stat = zip_path.stat()
            checkpoints.append(Checkpoint(
                name=name,
                steps=steps,
                timestamp=datetime.fromtimestamp(stat.st_mtime),
                size_mb=round(stat.st_size / 1_048_576, 2),
                path=str(zip_path),
            ))

        return checkpoints
