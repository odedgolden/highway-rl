import copy
from collections import Counter

import gym
import highway_env
import sys
from torch.distributions import Categorical

from final_project.display_utils import wrap_env

sys.path.insert(0, "/content/highway-env/scripts/")
from tqdm.notebook import trange

# from utils import record_videos, show_videos
import numpy as np
from gym import logger as gymlogger
from gym.wrappers import Monitor
from gym.utils import seeding
from gym import error, spaces, utils

gymlogger.set_level(40)  # error only
import io
import base64
import os
import random
import matplotlib.pyplot as plt
import math
import glob
from pyvirtualdisplay import Display
from IPython.display import HTML
from IPython import display as ipythondisplay
import pygame
import json
import ast

# %load_ext tensorboard
# %matplotlib inline

# =============== DO NOT DELETE ===============
file = open("../highway-config/config_ex1.txt", "r")
contents = file.read()
config1 = ast.literal_eval(contents)
file.close()
# ============================================

env = gym.make("highway-fast-v0")
config1["duration"] = 500
env.configure(config1)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

obs = env.reset()
a = np.max(obs)
for j in range(1):
    obs, _, _, _ = env.step(0)

    _, axes = plt.subplots(ncols=4, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(obs[i, ...].T, cmap=plt.get_cmap("gray"))

env = wrap_env(env)
observation = env.reset()
done = False
iter = 0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform(m.weight, -3 * 1.0e-4, 3 * 1.0e-4)
        m.bias.data.fill_(0.0001)


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.shared_layers = nn.Sequential(  # todo: name
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear(64 * 12 * 12, 5),
        )
        self.shared_layers.apply(init_weights)

        # self.optimizer = optim.Adam(actor_net.parameters(), lr=0.0001)

    def forward(self, x):
        input_tensor = torch.tensor(x, dtype=torch.float)
        hidden = self.shared_layers(input_tensor)
        # output = self.output(hidden)
        actions_probs = Categorical(F.softmax(hidden, dim=-1))

        return actions_probs, hidden


# conv1 = nn.Conv3d(4, 1, (5,5))
# tensor_t = torch.tensor(x.transpose((1,2,0)))
# tensor_t.reshape((1, *tensor_t.shape))


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        # self.shared_layers = nn.Sequential(  # todo: name
        #     nn.Conv2d(4, 128, kernel_size=4, stride=2),
        #     nn.Conv2d(128, 256, kernel_size=2, stride=1),
        #     nn.Conv2d(256, 128, kernel_size=2, stride=1),
        #     nn.Flatten(start_dim=0),
        #     nn.Linear(61 * 61 * 128, 128),
        #     nn.Linear(128, 1)
        # )
        self.shared_layers = nn.Sequential(  # todo: name
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(start_dim=0),
            nn.Linear(64 * 12 * 12, 1),
        )

        # self.optimizer = optim.Adam(actor_net.parameters(), lr=0.0001)

    def forward(self, x):
        input_tensor = torch.FloatTensor(x)
        hidden = self.shared_layers(input_tensor)

        return hidden


## play single env


def main():

    actor_net = ActorNet()
    actor_net_state = copy.deepcopy(actor_net)
    critic_net = CriticNet()
    critic_net_state = copy.deepcopy(critic_net)
    #
    curr_state = env.reset()
    # state_t = curr_state.transpose((1,2,0))

    optimizer_actor = optim.Adam(
        actor_net.parameters(), lr=0.0001
    )  # todo: which optimize
    optimizer_critic = optim.Adam(
        critic_net.parameters(), lr=0.001
    )  # todo: which optimize

    optimizer_state_actor = optim.Adam(
        actor_net_state.parameters(), lr=0.0001
    )  # todo: which optimize
    optimizer_state_critic = optim.Adam(
        critic_net_state.parameters(), lr=0.001
    )  # todo: which optimize
    # optimizer.zero_grad()
    # actor_net.eval() # from here
    for i in range(5_000):
        single_episode(actor_net, critic_net, env, episode_num=i)
        if len(REPLAY_RETURNS) >= 64:
            train(
                optimizer_actor,
                optimizer_critic,
                actor_net,
                critic_net,
                optimizer_state_actor,
                optimizer_state_critic,
                critic_net_state,
                actor_net_state,
            )


def train(
    optimizer_actor,
    optimizer_critic,
    actor_net,
    critic_net,
    optimizer_state_actor,
    optimizer_state_critic,
    critic_net_state,
    actor_net_state,
):
    global REPLAY_VALUES, REPLAY_REWARDS, REPLAY_NEXT_VALUES, REPLAY_RETURNS, REPLAY_LOG_PROBS
    current_buffer_size = len(REPLAY_RETURNS)
    if current_buffer_size < 50:
        print("buffer size is too small:", current_buffer_size)
        return

    chosen_indexes = random.sample(range(current_buffer_size), 32)

    log_probs = REPLAY_LOG_PROBS[chosen_indexes]
    returns = REPLAY_RETURNS[chosen_indexes]
    values = REPLAY_VALUES[chosen_indexes]
    rewards = REPLAY_REWARDS[chosen_indexes]
    next_values = REPLAY_NEXT_VALUES[chosen_indexes]

    critic_loss = F.mse_loss(
        values, (rewards + next_values)
    )  # todo: validate why 0.999

    actor_loss = -(
        log_probs * (returns - values)
    ).mean()  # -(log_probs * advantage.detach()).mean()
    # actor_loss = log_prob * advantage
    #
    #
    optimizer_state_actor.zero_grad()
    optimizer_state_critic.zero_grad()
    critic_loss.backward(retain_graph=True)
    print(f"action loss: {actor_loss.item()}, critic_loss:{critic_loss.item()}")
    actor_loss.backward()  # todo

    # critic_net_before_step = [x.data.clone() for x in critic_net.parameters()]
    # optimizer_state_actor.step()
    optimizer_actor.step()

    optimizer_state_critic.step()

    soft_tau = 0.001
    with torch.no_grad():
        for critic_state_net_params, critic_net_params in zip(
            critic_net_state.parameters(), critic_net.parameters()
        ):
            critic_net_params.data.copy_(
                +critic_net_params.data * (1.0 - soft_tau)
                + critic_state_net_params.data * soft_tau
            )
            # a = critic_net_params.data.copy_(critic_state_net_params.data)

    # for actor_state_net_params, actor_net_params in zip(actor_net_state.parameters(), actor_net.parameters()):
    #     actor_net_params.data.copy_(
    #         + actor_state_net_params.data * (1.0 - soft_tau) + actor_net_params.data * soft_tau
    #     )

    REPLAY_VALUES = torch.FloatTensor()
    REPLAY_REWARDS = torch.FloatTensor()
    REPLAY_NEXT_VALUES = torch.FloatTensor()
    REPLAY_RETURNS = torch.FloatTensor()
    REPLAY_LOG_PROBS = torch.FloatTensor()


def compute_returns(next_value, rewards, masks, values, discount_factor=0.99):
    # "update V using target r+...)
    returns = []
    R = next_value
    for step in reversed(range(len(rewards))):
        R = (
            rewards[step] + (discount_factor * R) * masks[step]
        )  # todo: in some version it without "- values[step]"
        # next_value_ = values[step]
        returns.insert(0, R)
    return returns


REPLAY_VALUES = torch.FloatTensor()
REPLAY_REWARDS = torch.FloatTensor()
REPLAY_NEXT_VALUES = torch.FloatTensor()
REPLAY_RETURNS = torch.FloatTensor()
REPLAY_LOG_PROBS = torch.FloatTensor()
# assert len(torch_values)==  len(torch_rewards) == len(torch_next_values) == len(torch_asa)

EPSILON_GREEDY_VALUE = 0.6

# def _choose_action_use_epsinlon_greedy(actions_probs):
#     if random.uniform(0, 1) <= EPSILON_GREEDY_VALUE:
#         return actions_probs.sample()
#     rand_index = random.randint(0,3)
#     return actions_probs[rand_index]


def single_episode(actor_net, critic_net, env, episode_num):
    global REPLAY_VALUES, REPLAY_REWARDS, REPLAY_NEXT_VALUES, REPLAY_RETURNS, REPLAY_LOG_PROBS
    # print('start training epoch ' , epoch)
    curr_state = env.reset()
    entropy = 0  # why this
    log_probs = []
    values = []
    v_pi_hat = []
    rewards = []
    masks = []  # i don't know this
    done = False
    steps = 0
    rewards_sum = 0
    actions_counter = Counter()
    while not done:
        steps += 1
        # 1. take action
        (actions_probs, actions_return), value = actor_net(curr_state), critic_net(
            curr_state
        )
        # print(f"probs {actions_probs.probs.detach().numpy()} returns {actions_return}")
        # action = _choose_action_use_epsinlon_greedy(actions_probs)
        action = actions_probs.sample()
        actions_counter[action.item()] += 1
        next_state, reward, done, extra_info = env.step(action.item())
        if episode_num > 100 and episode_num % 20 == 0:
            screen = env.render(mode="rgb_array")
            plt.imshow(screen)

        log_prob = actions_probs.log_prob(action).unsqueeze(
            0
        )  # math.log(actions_probs.probs[action]) # ==
        entropy += actions_probs.entropy().mean()  # why this

        log_probs.append(log_prob)
        values.append(value)
        rewards_sum += reward
        rewards.append(torch.tensor([reward], dtype=torch.float))
        masks.append(torch.tensor([1 - done], dtype=torch.float))
        curr_state = next_state  # need?
        if done:
            print(
                f"epoch: {episode_num}, Steps: {steps} , rewards-sum:{rewards_sum} extra-info:{extra_info} , "
                f"{actions_probs.probs.detach().numpy()}, actions-counter:{actions_counter}"
            )
            break

    next_state = torch.FloatTensor(next_state)
    next_value = critic_net(next_state)

    ### update V
    # critic_loss = F.mse_loss(torch.cat(values), (torch.cat(rewards) + 0.9 * torch.cat(values[1:] + [next_value])))
    # critic_loss = (torch.cat(rewards) + 0.9 * torch.cat(values)).mean()
    ###

    # Evaluate
    returns = compute_returns(next_value, rewards, masks, values)

    torch_values = torch.cat(values)
    torch_rewards = torch.cat(rewards)
    torch_next_values = torch.cat(values[1:] + [next_value])
    torch_returns = torch.cat(returns)
    torch_log_probs = torch.cat(log_probs)

    assert (
        len(torch_values)
        == len(torch_rewards)
        == len(torch_next_values)
        == len(torch_returns)
        == len(torch_log_probs)
    )

    REPLAY_VALUES = torch.cat((REPLAY_VALUES, torch_values), 0)
    REPLAY_REWARDS = torch.cat((REPLAY_REWARDS, torch_rewards), 0)
    REPLAY_NEXT_VALUES = torch.cat((REPLAY_NEXT_VALUES, torch_next_values), 0)
    REPLAY_RETURNS = torch.cat((REPLAY_RETURNS, torch_returns), 0)
    REPLAY_LOG_PROBS = torch.cat((REPLAY_LOG_PROBS, torch_log_probs), 0)

    #
    # log_probs = torch.cat(log_probs) # np.array(log_probs)
    # returns = torch.cat(returns).detach() #torch.cat(returns).detach()
    # values = torch.cat(values) #torch.cat(values)
    #
    # advantage = returns - values
    # "evaluate A..."
    # returns = torch.cat(rewards) + advantage


# 2. update V (critic)

if __name__ == "__main__":
    main()

# while (iter < 10) or not done:
#     if done:
#         break
#     iter += 1
#     action = env.action_space.sample()
#     observation, reward, done, _ = env.step(action)
#     screen = env.render(mode='rgb_array')
#     plt.imshow(screen)
#     print(f'iteration: {iter}, action: {action}, reward: {reward}, done: {done}')
