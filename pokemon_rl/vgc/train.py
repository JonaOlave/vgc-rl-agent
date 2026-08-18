"""
Entrenamiento PPO para VGC Doubles.

Uso:
    # Random Doubles (sin equipo, para empezar)
    python -m pokemon_rl.vgc.train

    # Champions VGC 2026 Reg M-B con warm-start BC
    python -m pokemon_rl.vgc.train --format gen9championsvgc2026regmb \\
        --champions-team --opponent heuristic --bc-model models/bc_pretrained

    # VGC 2025 Reg G con equipo real
    python -m pokemon_rl.vgc.train --format gen9vgc2025regg --vgc-team --opponent heuristic
"""

import argparse
import os
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn

from .env import make_vgc_env
from .results_db import connect as _connect_results_db
from .results_db import (
    register_training_complete,
    register_training_start,
    team_to_archetype,
)
from .team import (
    OPPONENT_TEAM_POOL_CHAMPIONS_REGMB,
    SAMPLE_TEAM_CHAMPIONS_REGMB,
    SAMPLE_TEAM_REG_H,
)


def train(
    battle_format: str = "gen9randomdoublesbattle",
    team: str | None = None,
    opponent: str = "random",
    total_timesteps: int = 500_000,
    save_path: str = "models/vgc/",
    log_path: str = "logs/vgc/",
    bc_model_path: Optional[str] = None,
    vary_opponent_team: bool = False,
    vary_own_team: bool = False,
    opponent_team_pool: Optional[list[str]] = None,
    own_team_pool: Optional[list[str]] = None,
):
    """
    opponent_team_pool / own_team_pool : list of str, optional
        Pool explícito a usar en vez del pool completo de 6 equipos que
        activan vary_opponent_team/vary_own_team — para acotar la
        superficie combinatoria (ej. entrenar contra sólo 2 rivales en vez
        de 6) sin tener que exponer un flag de CLI nuevo por cada
        combinación. Si se pasa, tiene prioridad sobre el flag booleano
        correspondiente.
    """
    resolved_opponent_pool = (
        opponent_team_pool
        if opponent_team_pool is not None
        else (OPPONENT_TEAM_POOL_CHAMPIONS_REGMB if vary_opponent_team else None)
    )
    resolved_own_pool = (
        own_team_pool
        if own_team_pool is not None
        else (OPPONENT_TEAM_POOL_CHAMPIONS_REGMB if vary_own_team else None)
    )

    print(f"Formato: {battle_format}")
    print(f"Oponente: {opponent}")
    if resolved_opponent_pool:
        print(f"Pool de equipos rivales: {len(resolved_opponent_pool)} equipos (variación activada)")
    if resolved_own_pool:
        print(f"Pool de equipos propios: {len(resolved_own_pool)} equipos (self-play activado)")
    print("Creando entorno VGC doubles...")

    db_conn = _connect_results_db()
    training_run_id = register_training_start(
        db_conn,
        total_timesteps=total_timesteps,
        own_team=team_to_archetype(team),
        own_team_pool=(
            [team_to_archetype(t) or t for t in resolved_own_pool] if resolved_own_pool else None
        ),
        opponent_behavior=opponent,
        opponent_team_pool=(
            [team_to_archetype(t) or t for t in resolved_opponent_pool] if resolved_opponent_pool else None
        ),
        warm_start_checkpoint_path=bc_model_path,
    )
    db_conn.close()

    env = make_vgc_env(
        battle_format=battle_format,
        server_host="localhost",
        server_port=8000,
        team=team,
        opponent=opponent,
        strict=False,
        opponent_team_pool=resolved_opponent_pool,
        own_team_pool=resolved_own_pool,
    )

    os.makedirs(save_path, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=save_path,
        name_prefix="vgc_ppo",
        verbose=1,
    )

    try:
        import tensorboard  # noqa: F401
        tb_log = log_path
    except ImportError:
        tb_log = None

    if bc_model_path:
        # Load BC pre-trained policy weights; replace the env reference for RL
        print(f"Cargando modelo BC desde {bc_model_path} ...")
        model = PPO.load(bc_model_path, env=env, tensorboard_log=tb_log)
        # Restore RL-appropriate hyperparameters (may differ from BC training).
        # learning_rate and clip_range must be schedule callables, not plain floats.
        model.learning_rate = get_schedule_fn(3e-4)
        model.clip_range = get_schedule_fn(0.2)
        model.n_steps = 512
        model.batch_size = 64
        model.n_epochs = 10
        model.gamma = 0.99
        model.gae_lambda = 0.95
        print("Pesos BC cargados — iniciando fine-tuning RL")
    else:
        # MultiInputPolicy for Dict obs: {"observation": Box(925,), "action_mask": Box(214,)}
        # Action space: MultiDiscrete([107, 107])
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=tb_log,
        )

    print(f"Iniciando entrenamiento VGC — {total_timesteps:,} timesteps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save(f"{save_path}/final")
    print(f"Modelo guardado → {save_path}/final.zip")
    env.close()

    db_conn = _connect_results_db()
    register_training_complete(
        db_conn, training_run_id,
        actual_timesteps=model.num_timesteps,
        final_checkpoint_path=f"{save_path}/final",
    )
    db_conn.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--format", default="gen9randomdoublesbattle",
                   help="Formato PS  (gen9randomdoublesbattle | gen9vgc2025regg | gen9championsvgc2026regmb)")
    p.add_argument("--opponent", default="random",
                   choices=["random", "heuristic", "support_heuristic"],
                   help="Tipo de oponente durante el entrenamiento")
    p.add_argument("--vgc-team", action="store_true",
                   help="Usar equipo VGC 2025 Reg G (gen9vgc2025regg)")
    p.add_argument("--champions-team", action="store_true",
                   help="Usar equipo Champions VGC 2026 Reg M-B (gen9championsvgc2026regmb)")
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--bc-model", default=None,
                   help="Ruta al modelo BC pre-entrenado .zip (sin extensión)")
    p.add_argument("--vary-opponent-team", action="store_true",
                   help="El oponente elige un equipo al azar por batalla "
                        "(pool en team.OPPONENT_TEAM_POOL_CHAMPIONS_REGMB) en vez "
                        "de ser siempre un mirror match del equipo propio")
    p.add_argument("--vary-own-team", action="store_true",
                   help="Nuestro propio agente también elige un equipo al azar por "
                        "batalla (mismo pool que --vary-opponent-team) en vez de usar "
                        "siempre --champions-team fijo — self-play, para que el modelo "
                        "aprenda a pilotar cualquier equipo del pool, no solo uno")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.champions_team:
        team = SAMPLE_TEAM_CHAMPIONS_REGMB
    elif args.vgc_team:
        team = SAMPLE_TEAM_REG_H
    else:
        team = None
    train(
        battle_format=args.format,
        team=team,
        opponent=args.opponent,
        total_timesteps=args.timesteps,
        bc_model_path=args.bc_model,
        vary_opponent_team=args.vary_opponent_team,
        vary_own_team=args.vary_own_team,
    )
