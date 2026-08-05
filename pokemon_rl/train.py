"""
Bucle de entrenamiento PPO con Stable-Baselines3.

La observación de poke-env 0.15 es un dict:
  {"observation": np.ndarray(825,), "action_mask": np.ndarray(26,)}

SB3 maneja dict observations automáticamente con CombinedExtractor.
Para usar action masking instala sb3-contrib y cambia PPO → MaskablePPO.

Ejecutar:
    source pokemon_rl/.venv/bin/activate
    python -m pokemon_rl.train
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from .env import make_env


def train(
    total_timesteps: int = 500_000,
    save_path: str = "models/",
    log_path: str = "logs/",
):
    print("Creando entorno...")
    env = make_env(
        battle_format="gen9randombattle",
        server_host="localhost",
        server_port=8000,
        strict=False,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=save_path,
        name_prefix="pokemon_ppo",
        verbose=1,
    )

    # Tensorboard es opcional — solo se activa si está instalado
    try:
        import tensorboard  # noqa: F401
        tb_log = log_path
    except ImportError:
        tb_log = None

    # MultiInputPolicy maneja el dict observation space
    # ({"observation": Box(825), "action_mask": Box(26)})
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

    print(f"Iniciando entrenamiento — {total_timesteps:,} timesteps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save(f"{save_path}/final")
    print(f"Modelo guardado → {save_path}/final.zip")
    env.close()


if __name__ == "__main__":
    train()
