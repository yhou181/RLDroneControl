from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
import torch
from stable_baselines3 import  A2C
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium
import PyFlyt.gym_envs # noqa

offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU)
env = gymnasium.make("PyFlyt/QuadX-Hover-v1")

eval_callback = EvalCallback(env,
                             verbose=1,
                             log_path= 'tensorboard_logs/',
                             eval_freq=1000,
                             deterministic=True,
                             render=False
                             )

model = A2C(a2cppoMlpPolicy,
                    env,
                    policy_kwargs=offpolicy_kwargs,
                    verbose=1,
                    seed=0,tensorboard_log="tensorboard_logs/")
model.learn(total_timesteps=200000,
                log_interval=100, callback=eval_callback)