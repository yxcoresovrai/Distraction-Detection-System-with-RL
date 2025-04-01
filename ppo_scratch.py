# agents/ppo_scratch.py - Upgraded PPO from Scratch (Offline Sovereign Edition)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from agents.env_agent import FocusEnvironment

GAMMA = 0.99
LR = 1e-3
EPS_CLIP = 0.2
K_EPOCHS = 4
TOTAL_EPISODES = 3000
BATCH_SIZE = 32

NUM_ACTIONS = 5
STATE_DIM = 4

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.pi = nn.Linear(64, NUM_ACTIONS)
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.pi(x), self.v(x)

class PPOBuffer:
    def __init__(self):
        self.states, self.actions, self.rewards, self.next_states, self.dones, self.logprobs = [], [], [], [], [], []

    def store(self, s, a, r, s_, done, lp):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_states.append(s_)
        self.dones.append(done)
        self.logprobs.append(lp)

    def clear(self):
        self.__init__()

class PPOAgentScratch:
    def __init__(self):
        self.policy = ActorCritic()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.buffer = PPOBuffer()
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.policy(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def train(self):
        states = torch.FloatTensor(self.buffer.states)
        actions = torch.LongTensor(self.buffer.actions).unsqueeze(1)
        rewards = torch.FloatTensor(self.buffer.rewards)
        next_states = torch.FloatTensor(self.buffer.next_states)
        logprobs = torch.stack(self.buffer.logprobs)

        _, values = self.policy(states)
        _, next_values = self.policy(next_states)
        returns = rewards + GAMMA * next_values.squeeze()
        advantages = returns - values.squeeze()

        for _ in range(K_EPOCHS):
            logits, state_values = self.policy(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_logprobs = dist.log_prob(actions.squeeze())

            ratio = (new_logprobs - logprobs.detach()).exp()
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages.detach()

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.loss_fn(state_values.squeeze(), returns.detach())
            entropy_bonus = dist.entropy().mean()  # optional
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.buffer.clear()

    def encode_state(self, state_dict):
        decode = {
                'last_action': {0: 'none', 1: 'popup', 2: 'break', 3: 'pomodoro', 4: 'rewrite_goal'}
        }
        if isinstance(state_dict['last_action'], int):
            state_dict['last_action'] = decode['last_action'].get(state_dict['last_action'], 'none')

        enc = {
            'time_of_day': {'morning': 0, 'afternoon': 1, 'evening': 2},
            'distraction_level': {'low': 0, 'medium': 1, 'high': 2},
            'last_feedback': {'none': 0, 'tired': 1, 'distracted': 2, 'unclear': 3, 'burnout': 4, 'overstimulated': 5, 'terminal_fear': 6},
            'last_action': {'none': 0, 'popup': 1, 'break': 2, 'pomodoro': 3, 'rewrite_goal': 4},
        }
        return [
            enc['time_of_day'].get(state_dict['time_of_day'], 0),
            enc['distraction_level'].get(state_dict['distraction_level'], 1),
            enc['last_feedback'].get(state_dict['last_feedback'], 0),   # fallback here
            enc['last_action'].get(state_dict['last_action'], 0),
        ]

def train_scratch_agent():
    env = FocusEnvironment()
    agent = PPOAgentScratch()

    for ep in range(TOTAL_EPISODES):
        state = env.reset()
        total_reward = 0

        for _ in range(10):
            encoded = agent.encode_state(state)
            action, logprob, _ = agent.select_action(encoded)
            # Map int index to action string for environment compatibility
            action_str = env.states['last_action'][action]
            next_state, reward, _, _ = env.step(action_str)

            print(f"Episode {ep}, Step {_}, Action: {action_str}, Reward: {reward}")

            # 🔧 Reward shaping
            if reward == -1:
                reward = -0.1  # softer penalty

            encoded_next = agent.encode_state(next_state)
            agent.buffer.store(encoded, action, reward, encoded_next, False, logprob)
            state = next_state
            total_reward += reward

        agent.train()

        if ep % 50 == 0:
            print(f"Episode {ep}, Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    train_scratch_agent()
