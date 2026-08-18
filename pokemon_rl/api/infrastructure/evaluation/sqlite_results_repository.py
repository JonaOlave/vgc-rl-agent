from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pokemon_rl.api.domain.evaluation.entities import BattleResult, Evaluation
from pokemon_rl.vgc.results_db import connect

_DB_PATH = Path("data/results.sqlite3")

_SELECT_EVALUATION = """
    SELECT e.id, e.training_run_id, t.round_number, e.our_team, e.opponent_archetype,
           e.opponent_behavior, e.n_battles, e.wins, e.losses, e.draws, e.win_rate,
           e.mean_reward, e.std_reward, e.mean_our_fainted, e.mean_opp_fainted,
           e.evaluated_at, t.final_checkpoint_path
    FROM evaluations e
    LEFT JOIN training_runs t ON t.id = e.training_run_id
"""


def _row_to_entity(row, battles: List[BattleResult]) -> Evaluation:
    (eval_id, _run_id, round_number, our_team, opponent_archetype, opponent_behavior,
     n_battles, wins, losses, draws, win_rate, mean_reward, std_reward,
     mean_our_fainted, mean_opp_fainted, evaluated_at, checkpoint_path) = row
    return Evaluation(
        id=str(eval_id),
        timestamp=datetime.fromisoformat(evaluated_at) if evaluated_at else datetime.min,
        model_path=checkpoint_path or "",
        opponent=opponent_behavior,
        n_battles=n_battles,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=win_rate,
        mean_reward=mean_reward,
        std_reward=std_reward,
        mean_our_fainted=mean_our_fainted,
        mean_opp_fainted=mean_opp_fainted,
        battles=battles,
        round_number=round_number,
        our_team=our_team,
        opponent_archetype=opponent_archetype,
    )


class SqliteResultsRepository:
    """Respaldo en SQLite (data/results.sqlite3) para IEvaluationRepository —
    reemplazo directo del ResultsRepository basado en JSON, misma interfaz
    (list_all/find_by_id/save). Ver pokemon_rl/vgc/results_db.py para el
    esquema y el backfill del historial real (rondas 3-15)."""

    def __init__(self, db_path: Path = _DB_PATH):
        self.db_path = db_path

    def _connect(self):
        return connect(str(self.db_path))

    def _battles_for(self, conn, eval_id: int) -> List[BattleResult]:
        rows = conn.execute(
            "SELECT battle_num, result, reward, our_fainted, opp_fainted "
            "FROM battle_results WHERE evaluation_id = ? ORDER BY battle_num",
            (eval_id,),
        ).fetchall()
        return [BattleResult(*r) for r in rows]

    def list_all(self) -> List[Evaluation]:
        conn = self._connect()
        rows = conn.execute(_SELECT_EVALUATION).fetchall()
        result = [_row_to_entity(row, self._battles_for(conn, row[0])) for row in rows]
        conn.close()
        return result

    def find_by_id(self, eval_id: str) -> Optional[Evaluation]:
        conn = self._connect()
        row = conn.execute(_SELECT_EVALUATION + " WHERE e.id = ?", (eval_id,)).fetchone()
        if row is None:
            conn.close()
            return None
        entity = _row_to_entity(row, self._battles_for(conn, row[0]))
        conn.close()
        return entity

    def save(self, evaluation: Evaluation) -> None:
        conn = self._connect()
        cur = conn.execute(
            """INSERT INTO evaluations
               (training_run_id, our_team, opponent_archetype, opponent_behavior,
                n_battles, wins, losses, draws, win_rate, mean_reward, std_reward,
                mean_our_fainted, mean_opp_fainted, evaluated_at)
               VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (evaluation.our_team, evaluation.opponent_archetype, evaluation.opponent,
             evaluation.n_battles, evaluation.wins, evaluation.losses, evaluation.draws,
             evaluation.win_rate, evaluation.mean_reward, evaluation.std_reward,
             evaluation.mean_our_fainted, evaluation.mean_opp_fainted,
             evaluation.timestamp.isoformat()),
        )
        eval_id = cur.lastrowid
        # SQLite asigna el id real acá (autoincrement), no el uuid que traía
        # el objeto al entrar — lo actualizamos en el propio objeto para que
        # el caller (ej. la respuesta de POST /api/evaluation/run) devuelva
        # el id que después realmente sirve para find_by_id().
        evaluation.id = str(eval_id)
        for b in evaluation.battles:
            conn.execute(
                """INSERT INTO battle_results
                   (evaluation_id, battle_num, result, reward, our_fainted, opp_fainted)
                   VALUES (?,?,?,?,?,?)""",
                (eval_id, b.battle_num, b.result, b.reward, b.our_fainted, b.opp_fainted),
            )
        conn.commit()
        conn.close()
