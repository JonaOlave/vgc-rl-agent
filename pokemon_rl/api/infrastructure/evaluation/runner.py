import uuid
from datetime import datetime
from typing import List

import numpy as np
from stable_baselines3 import PPO

from pokemon_rl.api.application.evaluation.commands import RunEvaluationCommand
from pokemon_rl.api.domain.evaluation.entities import BattleResult, Evaluation
from pokemon_rl.vgc.env import make_vgc_env
from pokemon_rl.vgc.team import SAMPLE_TEAM_CHAMPIONS_REGMB


class EvaluationRunner:
    def run(self, cmd: RunEvaluationCommand) -> Evaluation:
        team = SAMPLE_TEAM_CHAMPIONS_REGMB if cmd.use_champions_team else None
        env = make_vgc_env(
            battle_format="gen9championsvgc2026regmb",
            team=team,
            opponent=cmd.opponent,
            strict=False,
        )

        model = PPO.load(cmd.model_path, env=env)

        wins = losses = draws = 0
        battles: List[BattleResult] = []
        rewards_log: List[float] = []
        our_fainted_log: List[int] = []
        opp_fainted_log: List[int] = []

        for ep in range(1, cmd.n_battles + 1):
            obs, _ = env.reset()
            terminated = truncated = False
            ep_reward = 0.0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += float(reward)

            battle = env.env.battle1
            if battle.won:
                wins += 1
                result = "WIN"
            elif battle.lost:
                losses += 1
                result = "LOSS"
            else:
                draws += 1
                result = "DRAW"

            our_fainted = sum(1 for p in battle.team.values() if p.fainted)
            opp_fainted = sum(1 for p in battle.opponent_team.values() if p.fainted)

            rewards_log.append(ep_reward)
            our_fainted_log.append(our_fainted)
            opp_fainted_log.append(opp_fainted)
            battles.append(BattleResult(
                battle_num=ep,
                result=result,
                reward=round(ep_reward, 4),
                our_fainted=our_fainted,
                opp_fainted=opp_fainted,
            ))

        env.close()

        n = cmd.n_battles
        return Evaluation(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            model_path=cmd.model_path,
            opponent=cmd.opponent,
            n_battles=n,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=round(wins / n, 4),
            mean_reward=round(float(np.mean(rewards_log)), 4),
            std_reward=round(float(np.std(rewards_log)), 4),
            mean_our_fainted=round(float(np.mean(our_fainted_log)), 2),
            mean_opp_fainted=round(float(np.mean(opp_fainted_log)), 2),
            battles=battles,
        )
