# MountainCarContinuous-V0

import numpy as np
import gym
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import adam
from torch.distributions import Normal

env = gym.make('MountainCarContinuous-v0')
env = env.unwrapped
env.seed(1)

torch.manual_seed(1)
plt.ion()


#Hyperparameters
learning_rate = 0.02
gamma = 0.995
episodes = 1000

eps = np.finfo(np.float32).eps.item()

#action_space = env.action_space.n
action_space = 1
state_space = env.observation_space.shape[0]


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(state_space, 20)
        self.fc3 = nn.Linear(20, 20)
        self.mu_head = nn.Linear(20, 1)
        self.sigma_head = nn.Linear(20, 1)

        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        mu = F.sigmoid(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return mu, sigma

policy = Policy()
optimizer = adam.Adam(policy.parameters(), lr=learning_rate)

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    mu, sigma = policy(state)
    dist = Normal(mu, sigma)
    action = dist.sample()
    policy.saved_log_probs.append(dist.log_prob(np.clip(action, -1, 1)))
    action = action.item()
    return action

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []

    for r in policy.rewards[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Formalize reward
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean())/(rewards.std() + eps)

    # get loss
    for reward, log_prob in zip(rewards, policy.saved_log_probs):
        policy_loss.append(-log_prob * reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()



    del policy.rewards[:]
    del policy.saved_log_probs[:]

def plot(steps):
    #ax = plt.subplot(111)
    #ax.cla()
    #plt.set_title('Training')
    #plt.set_xlabel('Episode')
    #ax.set_ylabel('Run Time per Episode')
    plt.plot(steps, 'blue')
    plt.xlabel('Episode')
    plt.ylabel('Run Time per Episode')
    plt.gcf().set_size_inches(10.5, 6.5)
    plt.grid()
    plt.savefig('MountainCarContinuousPG.png')
    plt.pause(0.0000001)



def main():

    running_reward = 0
    steps = []
    for episode in count(0):
        state = env.reset()
        print(episode)
        for t in range(10000):
            action = select_action(state)
            action = [action]
            state, reward ,done, info = env.step(action)
            #env.render()
            policy.rewards.append(reward)

            if done:
                print("Episode {}, live time = {}".format(episode, t))
                steps.append(t)
                plot(steps)
                break

        running_reward = running_reward * policy.gamma - t*0.01
        finish_episode()

if __name__ == '__main__':
    main()
