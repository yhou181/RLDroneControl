from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
import torch
from stable_baselines3 import A2C, PPO, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium
import PyFlyt.gym_envs # noqa

offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU)
env = gymnasium.make("PyFlyt/QuadX-Hover-v1", render_mode="human")
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='hover_models/')

model = DDPG(td3ddpgMlpPolicy,
                    env,
                    policy_kwargs=offpolicy_kwargs,
                    verbose=1,
                    seed=0)
model.learn(total_timesteps=600000,
                log_interval=100, callback=checkpoint_callback)