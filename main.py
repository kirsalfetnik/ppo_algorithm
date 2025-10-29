import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import safety_gymnasium

ENV_NAME = "SafetyCarGoal1-v0"

TOTAL_UPDATES = 400
STEPS_PER_ROLLOUT = 2048
MINI_BATCH_SIZE = 256
PPO_EPOCHS = 10

GAMMA = 0.99
LAMBDA_GAE = 0.95
CLIP_EPS = 0.2

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super.__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        self.mu_head = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        std = torch.exp(self.log_std)
        return mu, std
    
    def get_dist(self, x):
        mu, std = self.forward(x)
        return Normal(mu, std)
    
    def act(self, x):
        dist = self.get_dist(x)
        action = dist.sample()
        logp = dist.log_prob(action).sum(axis=-1)
        return action, logp
    
    def log_prob(self, x, actions):
        dist = self.get_dist(x)
        return dist.log_prob(actions).sum(axis=-1)
    
    def entropy(self, x):
        dist = self.get_dist(x)
        return dist.entropy().sum(axis=-1)
    
class Critic(nn.Module):
    def __init__(self, obs_dim):
        super.__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64)
            nn.Tanh()
            nn.Linear(64, 1)
        )

    def forward(self, x):
        v = self.net(x).squeeze(-1)
        return v
    
def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=LAMBDA_GAE):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_nonterminal = 1.0 - dones[t+1]
            next_val = values[t+1]


