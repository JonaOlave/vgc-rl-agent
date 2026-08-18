"""
Salida rankeada de acciones — wrapper de inferencia sobre un checkpoint PPO
ya entrenado.

`model.predict()` normalmente da una sola acción por turno (muestreada o
determinística). Para un caso de uso de asesor conviene mostrar varias
opciones legales con la confianza que la política les asigna, no sólo la
elegida. No requiere reentrenar — funciona con cualquier checkpoint PPO ya
entrenado sobre este mismo espacio de observación/acción.

Importante — qué significa (y qué NO significa) la probabilidad acá:
la probabilidad de la política es π(a|s): qué tan seguido el modelo
entrenado elige esa acción en ese estado, no una estimación de qué tan
buena es en resultado esperado. PPO sólo calcula V(s) (el valor del estado
completo), no un valor por acción individual — eso sería más parecido a
Q-learning/DQN, que esta arquitectura no usa. Tratar el ranking como
"confianza del modelo", no como "la jugada objetivamente mejor".

Además: la política nunca aprendió a respetar la máscara de acciones (ver
CLAUDE.md — es el motivo original de los fixes de force-switch/target en
vgc/env.py), así que su distribución cruda le asigna probabilidad no-nula a
acciones ilegales. Por eso acá se filtra explícitamente contra
`DoublesEnv.get_action_mask_individual` antes de rankear — no alcanza con
tomar el top-k crudo de la política.
"""

from typing import Dict, List, TypedDict

import numpy as np
from poke_env.battle import DoubleBattle
from poke_env.environment import DoublesEnv
from stable_baselines3 import PPO


class RankedAction(TypedDict):
    action: int
    probability: float
    order: str


def rank_actions(
    model: PPO,
    obs: Dict[str, np.ndarray],
    battle: DoubleBattle,
    top_k: int = 5,
) -> Dict[int, List[RankedAction]]:
    """
    Para cada posición activa (0 y 1), retorna hasta `top_k` acciones
    LEGALES ordenadas por probabilidad de la política, con su orden
    decodificada a texto legible (vía la misma lógica que usa
    `DoublesEnv` para convertir un int en una orden real de Showdown).
    """
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    distribution = model.policy.get_distribution(obs_tensor)
    per_position_probs = [
        d.probs.detach().cpu().numpy().squeeze(0) for d in distribution.distribution
    ]

    result: Dict[int, List[RankedAction]] = {}
    for pos in (0, 1):
        legal_mask = DoublesEnv.get_action_mask_individual(battle, pos)
        probs = per_position_probs[pos]
        candidates = sorted(
            (
                (action_id, float(probs[action_id]))
                for action_id, legal in enumerate(legal_mask)
                if legal
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        ranked: List[RankedAction] = []
        for action_id, prob in candidates[:top_k]:
            order = DoublesEnv._action_to_order_individual(
                np.int64(action_id), battle, fake=True, pos=pos
            )
            ranked.append(
                RankedAction(action=action_id, probability=prob, order=str(order))
            )
        result[pos] = ranked
    return result


def estimate_state_value(model: PPO, obs: Dict[str, np.ndarray]) -> float:
    """
    V(s) — valor estimado del estado actual completo (no de una acción
    puntual). Sirve como contexto general ("qué tan bien vamos"), no como
    parte del ranking por acción.
    """
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    value = model.policy.predict_values(obs_tensor)
    return float(value.item())
