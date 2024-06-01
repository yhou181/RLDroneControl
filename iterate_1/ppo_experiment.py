import gymnasium
import PyFlyt.gym_envs # noqa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mean = self.fc2(x)
        log_std = self.fc3(x).clamp(-20, 2)  # Clamping to avoid numerical instability
        std = torch.exp(log_std)
        return mean, std


class PPO:
    def __init__(self, env, policy, optimizer, gamma=0.99, clip_ratio=0.2, ppo_steps=2048, ppo_epochs=100,
                 mini_batch_size=64):
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.ppo_steps = ppo_steps
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.policy(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.numpy()[0], log_prob.sum(dim=1).item()

    def compute_returns_and_advantages(self, rewards, dones, values):
        returns = []
        advantages = []
        gae = 0
        next_value = 0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[step])
            next_value = values[step]
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)

        # Normalizing advantages
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def train(self):


        # Policy update
        for _ in range(self.ppo_epochs):
            state, other_info = self.env.reset()

            done = False
            score = 0
            data = []

            for step in range(self.ppo_steps):
                action, log_prob = self.get_action(state)
                next_state, reward, done, truncation, info = self.env.step(action)
                data.append((state, action, log_prob, reward, done))
                state = next_state
                score += reward

                if done or truncation:
                    state,other_info = self.env.reset()
                    print("Episode Score: ", score)
                    score = 0

            # Prepare batch data
            states, actions, log_probs, rewards, dones = zip(*data)
            values = [self.policy(torch.FloatTensor(state))[0].item() for state in states]  # Estimating value function
            returns, advantages = self.compute_returns_and_advantages(rewards, dones, values)

            idx = np.random.randint(0, len(data), self.mini_batch_size)
            sampled_states = torch.FloatTensor([states[i] for i in idx])
            sampled_actions = torch.FloatTensor([actions[i] for i in idx])
            sampled_log_probs = torch.FloatTensor([log_probs[i] for i in idx])
            sampled_returns = torch.FloatTensor([returns[i] for i in idx])
            sampled_advantages = torch.FloatTensor([advantages[i] for i in idx])

            new_means, new_stds = self.policy(sampled_states)
            new_dists = Normal(new_means, new_stds)
            new_log_probs = new_dists.log_prob(sampled_actions).sum(axis=1)

            ratio = (new_log_probs - sampled_log_probs).exp()
            surr1 = ratio * sampled_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * sampled_advantages
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    env = gymnasium.make("PyFlyt/QuadX-Hover-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    print(state_dim)
    action_dim = env.action_space.shape[0]
    print(action_dim)
    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    ppo = PPO(env, policy, optimizer)
    ppo.train()



