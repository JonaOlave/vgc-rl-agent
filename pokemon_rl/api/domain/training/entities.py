from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Checkpoint:
    name: str
    steps: int
    timestamp: datetime
    size_mb: float
    path: str
