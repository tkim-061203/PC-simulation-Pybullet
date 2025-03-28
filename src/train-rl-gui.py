from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from quadruped_env import QuadrupedBulletEnv
import numpy as np
import torch

def train():
    # Create and wrap the environment
    env = QuadrupedBulletEnv(render=True)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-08
    )

    # Create the model with tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,  # Lower learning rate for stability
        n_steps=2048,  # Increased steps per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,  # Slightly increased exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./quadruped_tensorboard/",
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256, 128],  # Deeper network
                vf=[256, 256, 128]
            ),
            activation_fn=torch.nn.ReLU,
            ortho_init=True
        )
    )

    # Setup checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./logs/",
        name_prefix="quadruped_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    # Train the model
    try:
        model.learn(
            total_timesteps=1000000,
            callback=checkpoint_callback,
            log_interval=10
        )
    except Exception as e:
        print(f"Training error: {e}")
        env.close()
        return

    # Save the final model
    model.save("quadruped_final_model")
    env.save("vec_normalize.pkl")
    env.close()

if __name__ == "__main__":
    train()