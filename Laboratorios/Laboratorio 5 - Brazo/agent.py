import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from model import ActorCritic

import numpy as np


class PPOAgent:
    def __init__(self, model: ActorCritic, optimizer: optim.Optimizer, gamma: float, gae_lambda: float,
                 clip_epsilon: float, ppo_epochs: int, value_coeff: float, entropy_coeff: float, max_grad_norm: float):

        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm

        self.data_buffer = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            mean, std, value = self.model(state)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)

        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def compute_gae(self, next_value, rewards, masks, values):
        values = torch.cat([values, next_value.unsqueeze(0)])
        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * \
                values[i+1] * masks[i] - values[i]
            gae = delta + self.gamma * self.gae_lambda * masks[i] * gae
            returns.insert(0, gae + values[i])
        return torch.tensor(returns), torch.tensor(values[:-1])

    def update(self, states, actions, old_log_probs, old_values, rewards, masks, last_value):
        returns, values = self.compute_gae(
            last_value, rewards, masks, old_values)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            mean, std, new_values = self.model(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(axis=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                1.0 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = (new_values.squeeze() - returns).pow(2).mean()

            entropy = dist.entropy().mean()

            loss = actor_loss + self.value_coeff * \
                critic_loss - self.entropy_coeff * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
