from stable_baselines3 import  DDPG
import gymnasium
import numpy as np
from gymnasium.envs.registration import register

register(
    id='QuadX-Hover-Test-v1',
    entry_point='quadx_hover_test_env:QuadXHoverTestEnv',
)

model_path = "hover_models/hover_model.zip"
model = DDPG.load(model_path)

env = gymnasium.make("QuadX-Hover-Test-v1", render_mode="human", start_pos = np.array([[0.5,0.5,1.0]]))
num_episodes = 100
all_rewards = []

obs,_ = env.reset()
done = False
total_rewards = 0
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info,_ = env.step(action)
    total_rewards += rewards

