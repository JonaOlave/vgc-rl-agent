"""
Base de resultados (SQLite) — historial de rondas de entrenamiento y sus
evaluaciones, con el detalle batalla por batalla.

Reemplaza el registro manual que veníamos haciendo en memoria/conversación:
antes de esto, saber "de qué ronda salió este checkpoint" o "qué cambió
entre la ronda 13 y la 14" exigía releer notas. Acá queda como datos
consultables.

Vive en `data/results.sqlite3` (mismo directorio que el
`evaluation_results.json` de semilla que ya usa la API — ver
`pokemon_rl/api/infrastructure/`).

Esquema
-------
training_runs   — una fila por ronda de entrenamiento (`train()` invocado
                   una vez). `warm_start_run_id` es autorreferencia: de qué
                   ronda anterior arrancó (NULL para la primera, basada en
                   el checkpoint de BC).
evaluations     — una fila por llamada a `evaluate()` (normalmente una por
                   matchup dentro de una ronda — 6 para "un equipo vs pool
                   completo", 8 para las rondas de self-play con Track A/B).
battle_results  — una fila por batalla individual dentro de una evaluación.
                   No todas las evaluaciones tienen esto: las rondas 3-5
                   son anteriores a que el script de evaluación imprimiera
                   detalle por batalla, sólo se conservan sus agregados.
"""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY,
    round_number INTEGER UNIQUE NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    total_timesteps INTEGER,
    actual_timesteps INTEGER,
    duration_seconds INTEGER,
    warm_start_run_id INTEGER REFERENCES training_runs(id),
    own_team TEXT,
    own_team_pool TEXT,
    opponent_behavior TEXT NOT NULL,
    opponent_team_pool TEXT,
    final_checkpoint_path TEXT,
    final_checkpoint_md5 TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY,
    training_run_id INTEGER REFERENCES training_runs(id),
    our_team TEXT NOT NULL,
    opponent_archetype TEXT NOT NULL,
    opponent_behavior TEXT NOT NULL,
    n_battles INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    draws INTEGER NOT NULL,
    win_rate REAL NOT NULL,
    mean_reward REAL,
    std_reward REAL,
    mean_our_fainted REAL,
    mean_opp_fainted REAL,
    evaluated_at TEXT
);

CREATE TABLE IF NOT EXISTS battle_results (
    id INTEGER PRIMARY KEY,
    evaluation_id INTEGER NOT NULL REFERENCES evaluations(id),
    battle_num INTEGER NOT NULL,
    result TEXT NOT NULL,
    reward REAL,
    our_fainted INTEGER,
    opp_fainted INTEGER
);

CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(training_run_id);
CREATE INDEX IF NOT EXISTS idx_battles_eval ON battle_results(evaluation_id);
"""


def connect(db_path: str = "data/results.sqlite3") -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# Tracking en vivo — usado por train.py (una ronda) y por scripts de
# evaluación (eval_roundN_matchups.py en adelante) para que las rondas 16+
# queden registradas solas, sin necesitar otro backfill manual.
# ---------------------------------------------------------------------------

import hashlib
import json as _json


def _team_archetypes():
    # Import diferido: evita que cualquier cosa que sólo necesite el
    # esquema (ej. la API) tenga que importar pokemon_rl.vgc.team.
    from . import team as _t
    return {
        _t.SAMPLE_TEAM_CHAMPIONS_REGMB: "champions",
        _t.SAMPLE_TEAM_TRICKROOM: "trickroom",
        _t.SAMPLE_TEAM_TAILWIND: "tailwind",
        _t.SAMPLE_TEAM_RAIN: "rain",
        _t.SAMPLE_TEAM_SAND: "sand",
        _t.SAMPLE_TEAM_SUN: "sun",
    }


def team_to_archetype(team_str):
    """Equipo (string Showdown) -> nombre corto de arquetipo, o None si no
    matchea ninguno de los equipos conocidos en vgc/team.py (ej. un equipo
    custom nuevo que todavía no tiene nombre asignado acá)."""
    if team_str is None:
        return None
    return _team_archetypes().get(team_str)


def _file_md5(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except FileNotFoundError:
        return None


def register_training_start(
    conn, *, total_timesteps, own_team=None, own_team_pool=None,
    opponent_behavior, opponent_team_pool=None, warm_start_checkpoint_path=None,
    notes=None,
) -> int:
    """Inserta la fila de training_runs al arrancar una ronda. Determina el
    round_number como max(existente)+1 (numeración secuencial, igual a como
    se vino haciendo a mano). Intenta resolver warm_start_run_id comparando
    el md5 del checkpoint de arranque contra final_checkpoint_md5 de rondas
    ya registradas — funciona sin importar qué alias de ruta se haya usado
    (models/vgc/final vs. una ruta de vgc_history/), porque compara
    contenido, no el string de la ruta."""
    next_round = (conn.execute("SELECT COALESCE(MAX(round_number), 0) FROM training_runs").fetchone()[0]) + 1

    warm_start_run_id = None
    if warm_start_checkpoint_path:
        checkpoint_file = warm_start_checkpoint_path
        if not checkpoint_file.endswith(".zip"):
            checkpoint_file += ".zip"
        md5 = _file_md5(checkpoint_file)
        if md5:
            row = conn.execute(
                "SELECT id FROM training_runs WHERE final_checkpoint_md5 = ?", (md5,)
            ).fetchone()
            if row:
                warm_start_run_id = row[0]

    cur = conn.execute(
        """INSERT INTO training_runs
           (round_number, started_at, total_timesteps, warm_start_run_id,
            own_team, own_team_pool, opponent_behavior, opponent_team_pool, notes)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (next_round, datetime_now_iso(), total_timesteps, warm_start_run_id,
         own_team, _json.dumps(own_team_pool) if own_team_pool else None,
         opponent_behavior, _json.dumps(opponent_team_pool) if opponent_team_pool else None,
         notes),
    )
    conn.commit()
    print(f"[results_db] Ronda {next_round} registrada (training_run_id={cur.lastrowid})"
          + (f", warm-start desde ronda {conn.execute('SELECT round_number FROM training_runs WHERE id=?', (warm_start_run_id,)).fetchone()[0]}" if warm_start_run_id else ""))
    return cur.lastrowid


def register_training_complete(conn, run_id, *, actual_timesteps, final_checkpoint_path):
    """Completa la fila al terminar la ronda: timesteps reales, duración,
    checkpoint final + su md5."""
    started_at = conn.execute(
        "SELECT started_at FROM training_runs WHERE id = ?", (run_id,)
    ).fetchone()[0]
    completed_at = datetime_now_iso()
    duration = None
    if started_at:
        from datetime import datetime
        duration = int((datetime.fromisoformat(completed_at) - datetime.fromisoformat(started_at)).total_seconds())

    checkpoint_file = final_checkpoint_path
    if not checkpoint_file.endswith(".zip"):
        checkpoint_file += ".zip"
    md5 = _file_md5(checkpoint_file)

    conn.execute(
        """UPDATE training_runs
           SET completed_at=?, actual_timesteps=?, duration_seconds=?,
               final_checkpoint_path=?, final_checkpoint_md5=?
           WHERE id=?""",
        (completed_at, actual_timesteps, duration, final_checkpoint_path, md5, run_id),
    )
    conn.commit()
    round_number = conn.execute("SELECT round_number FROM training_runs WHERE id=?", (run_id,)).fetchone()[0]
    print(f"[results_db] Ronda {round_number} completada (training_run_id={run_id})")


def datetime_now_iso():
    from datetime import datetime
    return datetime.now().isoformat()


def record_evaluation(
    conn, *, training_run_id=None, our_team, opponent_archetype, opponent_behavior,
    result: dict, battles=None,
) -> int:
    """Persiste el resultado de una evaluación (lo que devuelve evaluate())
    más el detalle batalla-por-batalla si `battles` viene poblado (ver el
    nuevo campo "battles" que evaluate() devuelve desde ahora)."""
    cur = conn.execute(
        """INSERT INTO evaluations
           (training_run_id, our_team, opponent_archetype, opponent_behavior,
            n_battles, wins, losses, draws, win_rate, mean_reward, std_reward,
            mean_our_fainted, mean_opp_fainted, evaluated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (training_run_id, our_team, opponent_archetype, opponent_behavior,
         result["wins"] + result["losses"] + result.get("draws", 0),
         result["wins"], result["losses"], result.get("draws", 0), result["win_rate"],
         result.get("mean_reward"), result.get("std_reward"),
         result.get("mean_our_fainted"), result.get("mean_opp_fainted"),
         datetime_now_iso()),
    )
    eval_id = cur.lastrowid
    for b in battles or []:
        conn.execute(
            """INSERT INTO battle_results
               (evaluation_id, battle_num, result, reward, our_fainted, opp_fainted)
               VALUES (?,?,?,?,?,?)""",
            (eval_id, b["battle_num"], b["result"], b["reward"], b["our_fainted"], b["opp_fainted"]),
        )
    conn.commit()
    return eval_id
