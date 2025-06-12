import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(ActorCritic, self).__init__()

        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim * 2)
        )

        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        actor_output = self.actor_net(state)
        mean = actor_output[:, :self.log_std.shape[1]]
        std = torch.exp(self.log_std.expand_as(mean))
        std = torch.clamp(std, min=1e-6, max=1.0)

        value = self.critic_net(state)

        return mean, std, value
