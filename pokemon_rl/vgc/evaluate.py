"""
Evaluación del modelo PPO entrenado para VGC Champions.

Uso:
    # Evaluar modelo final contra random (50 batallas)
    python -m pokemon_rl.vgc.evaluate --champions-team

    # Evaluar un checkpoint específico contra heuristic
    python -m pokemon_rl.vgc.evaluate --champions-team \\
        --model models/vgc/vgc_ppo_500000_steps --opponent heuristic --n-battles 30

    # Comparar dos modelos (ejecutar dos veces)
    python -m pokemon_rl.vgc.evaluate --model models/vgc/final --n-battles 50
"""

import argparse
from typing import Optional

import numpy as np
from stable_baselines3 import PPO

from .env import make_vgc_env
from .team import SAMPLE_TEAM_CHAMPIONS_REGMB, SAMPLE_TEAM_REG_H


def evaluate(
    model_path: str = "models/vgc/final",
    battle_format: str = "gen9championsvgc2026regmb",
    team: Optional[str] = None,
    opponent: str = "random",
    n_battles: int = 50,
    deterministic: bool = True,
    opponent_team_pool: Optional[list[str]] = None,
) -> dict:
    """
    Carga el modelo entrenado y lo enfrenta contra un oponente N veces.

    opponent_team_pool : list of str, optional
        Si se pasa, el rival elige equipo al azar de este pool en cada batalla
        (ver make_vgc_env/RandomTeamPool) en vez de usar `team`. Pasar una
        lista de un solo elemento fija el equipo rival a un matchup específico
        (ej. `[SAMPLE_TEAM_TRICKROOM]`) mientras `team` sigue fijando el
        nuestro — así se arma la tabla de matchups (mirror + arquetipos).

    Retorna un dict con: win_rate, wins, losses, draws, mean_reward,
    mean_our_fainted, mean_opp_fainted.
    """
    print(f"Modelo   : {model_path}")
    print(f"Formato  : {battle_format}")
    print(f"Oponente : {opponent}  |  Batallas: {n_battles}")
    print(f"Modo     : {'determinístico' if deterministic else 'estocástico'}\n")

    env = make_vgc_env(
        battle_format=battle_format,
        team=team,
        opponent=opponent,
        strict=False,
        opponent_team_pool=opponent_team_pool,
    )

    model = PPO.load(model_path, env=env)

    wins = losses = draws = 0
    total_reward = 0.0
    our_fainted_log: list[int] = []
    opp_fainted_log: list[int] = []
    rewards_log: list[float] = []

    print(f"{'Batalla':>7}  {'Resultado':6}  {'Reward':>7}  {'Caídos (nos/op)':>16}")
    print("-" * 47)

    for ep in range(1, n_battles + 1):
        obs, _ = env.reset()
        terminated = truncated = False
        ep_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

        # Resultado de la batalla
        battle = env.env.battle1
        if battle.won:
            wins += 1
            tag = "WIN"
        elif battle.lost:
            losses += 1
            tag = "LOSS"
        else:
            draws += 1
            tag = "DRAW"

        # Pokémon caídos en ambos equipos
        our_fainted = sum(1 for p in battle.team.values() if p.fainted)
        opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)

        our_fainted_log.append(our_fainted)
        opp_fainted_log.append(opp_fainted)
        total_reward += ep_reward
        rewards_log.append(ep_reward)

        print(
            f"{ep:>7d}  {tag:6s}  {ep_reward:>+7.2f}  "
            f"     {our_fainted}/6  vs  {opp_fainted}/6"
        )

    env.close()

    n = n_battles
    win_rate = wins / n

    print("\n" + "=" * 55)
    print(f"  Resultados — {n} batallas vs {opponent}")
    print("=" * 55)
    print(f"  Victorias  : {wins:3d} / {n}   ({win_rate * 100:.1f} %)")
    print(f"  Derrotas   : {losses:3d} / {n}   ({losses / n * 100:.1f} %)")
    print(f"  Empates    : {draws:3d} / {n}   ({draws / n * 100:.1f} %)")
    print(f"  Reward medio       : {np.mean(rewards_log):+.4f}")
    print(f"  Reward std         : {np.std(rewards_log):.4f}")
    print(f"  Caídos nuestros    : {np.mean(our_fainted_log):.2f} ± {np.std(our_fainted_log):.2f}")
    print(f"  Caídos oponente    : {np.mean(opp_fainted_log):.2f} ± {np.std(opp_fainted_log):.2f}")
    print("=" * 55)

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "mean_reward": float(np.mean(rewards_log)),
        "std_reward": float(np.std(rewards_log)),
        "mean_our_fainted": float(np.mean(our_fainted_log)),
        "mean_opp_fainted": float(np.mean(opp_fainted_log)),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Evaluación del modelo VGC PPO")
    p.add_argument("--model", default="models/vgc/final",
                   help="Ruta al modelo .zip (sin extensión)")
    p.add_argument("--format", default="gen9championsvgc2026regmb",
                   dest="battle_format",
                   help="Formato PS")
    p.add_argument("--champions-team", action="store_true",
                   help="Usar equipo Champions VGC 2026 Reg M-B")
    p.add_argument("--vgc-team", action="store_true",
                   help="Usar equipo VGC 2025 Reg G")
    p.add_argument("--opponent", default="random",
                   choices=["random", "heuristic", "support_heuristic"],
                   help="Tipo de oponente")
    p.add_argument("--n-battles", type=int, default=50,
                   help="Número de batallas de evaluación")
    p.add_argument("--stochastic", action="store_true",
                   help="Usar política estocástica (por defecto: determinística)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.champions_team:
        team = SAMPLE_TEAM_CHAMPIONS_REGMB
    elif args.vgc_team:
        team = SAMPLE_TEAM_REG_H
    else:
        team = None

    evaluate(
        model_path=args.model,
        battle_format=args.battle_format,
        team=team,
        opponent=args.opponent,
        n_battles=args.n_battles,
        deterministic=not args.stochastic,
    )
