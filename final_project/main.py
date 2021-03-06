import copy
import os
import random
from datetime import datetime
from collections import Counter

import gym
import pandas as pd
import torch.nn.functional as F
import highway_env
import uuid
import sys

import torch
from torch.distributions import Categorical

from final_project.replay_buffer import ReplayBuffer


sys.path.insert(0, "/content/highway-env/scripts/")
from tqdm.notebook import trange

# from utils import record_videos, show_videos
import numpy as np
from gym import logger as gymlogger
from gym.wrappers import Monitor
from gym.utils import seeding
from gym import error, spaces, utils

gymlogger.set_level(40)  # error only
import matplotlib.pyplot as plt
import actor_critic_models
import ast

# %load_ext tensorboard
# %matplotlib inline

# =============== DO NOT DELETE ===============
file = open("../highway-config/config_ex1.txt", "r")
contents = file.read()
config1 = ast.literal_eval(contents)
config1["duration"] = 500
file.close()
# ============================================
GAMMA = 0.99

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


EPOCHS = 500
DEFAULT_BUFFER_MAX_SIZE = 500
SHOW_VIDEO = False
STOCHASTIC_RULE = 0.85

EXPERIMENT_FILENAME = "highway_experiments.csv"


class Agent:
    def __init__(
        self,
        buffer_max_size=None,
        gamma=0.99,
        tau=0.001,
        experiment_description="",
        replay_buffer_sampling_percent=0.7,
    ):
        self.actor_net = actor_critic_models.ActorNet()
        self.critic_net = actor_critic_models.CriticNet()
        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.replay_buffer_memory = ReplayBuffer(
            buffer_max_size=buffer_max_size,
            sampling_percent=replay_buffer_sampling_percent,
        )
        self.gamma = gamma
        self.tau = tau
        self.unq_id = str(uuid.uuid4())  # for tracking experiments
        self.experiment_description = experiment_description
        self.replay_buffer_sampling_percent = replay_buffer_sampling_percent
        self.start_time = datetime.now()

        self.all_rewards_sum = []
        self.all_learning_durations = []
        self.number_of_steps = []
        self.car_crashed = []
        self.all_actor_loss = []
        self.all_critic_loss = []
        self._init_env()

    def _init_env(self):
        self.env = gym.make("highway-fast-v0")
        self.env.configure(config1)
        obs = self.env.reset()

    def single_episode(self, episode_num):
        global SHOW_VIDEO
        self.actor_net.eval()
        self.critic_net.eval()
        curr_state = self.env.reset()
        done = False

        actions_counter = Counter()

        all_probs = []
        all_rewards = []
        sum_rewards = 0
        steps = 0
        all_next_state_values = []
        all_done = []

        while not done:
            actions_probs = self.actor_net.forward([curr_state])
            all_probs.append(actions_probs)
            action = self._choose_action(actions_probs)[0]
            action_step = self._apply_stochastic_env(action)
            next_state, reward, done, extra_info = self.env.step(action.item())
            if extra_info["crashed"]:
                before = reward
                # reward -=15
                print(f"crashed!! reward was {before} and now {reward}")
                self.car_crashed.append(True)
            else:
                self.car_crashed.append(False)
            sum_rewards += reward
            all_rewards.append(reward)
            next_state_value = self._target_critic_net_on_step(next_state)
            log_prob = actions_probs.log_prob(action).unsqueeze(0)
            all_next_state_values.append(next_state_value)
            all_done.append(done)

            if (episode_num > 100 and episode_num % 20 == 0) and SHOW_VIDEO:
                screen = self.env.render(mode="rgb_array")
                plt.imshow(screen)

            self.replay_buffer_memory.add(
                curr_states=curr_state,
                next_states=next_state,
                rewards=reward,
                dones=done,
                actions=action,
                log_probs=log_prob,
            )
            actions_counter[action.item()] += 1

            steps += 1
            curr_state = next_state

        self.all_rewards_sum.append(sum_rewards)

        # probs_mean = (
        #     np.array([x.probs.clone().detach().numpy() for x in all_probs])
        #     .transpose()
        #     .mean(axis=1)
        # )
        # logist_mean = (
        #     np.array([x.logits.clone().detach().numpy() for x in all_probs])
        #     .transpose()
        #     .mean(axis=1)
        # )
        self.number_of_steps.append(steps)
        print(
            f"done episode number {episode_num} with sum-rewards {sum_rewards} after {steps} steps actions-counter:{actions_counter}"
        )

    def _estimate_step_value(self, done, next_state, reward):
        next_state_value = self._target_critic_net_on_step(next_state)
        reward_tensor = torch.FloatTensor(reward)[:, None]
        done_tensor = torch.FloatTensor(done)[:, None]
        target_value = reward_tensor + self.gamma * next_state_value * (1 - done_tensor)
        return target_value

    def learn(self, episode_num):
        self.critic_net.train()
        self.actor_net.train()
        self.critic_net.optimizer.zero_grad()
        self.actor_net.optimizer.zero_grad()

        current_buffer = self.replay_buffer_memory.sample_values()
        if not current_buffer:
            return

        buffer_curr_states = current_buffer["curr_states"]
        buffer_next_states = current_buffer["next_states"]
        buffer_rewards = current_buffer["rewards"]
        buffer_dones = current_buffer["dones"]
        buffer_actor_probs = self.target_actor_net.forward(buffer_curr_states)
        buffer_log_probs = buffer_actor_probs.log_prob(buffer_actor_probs.sample())

        buffer_target_value = self._estimate_step_value(
            buffer_dones, torch.FloatTensor(buffer_next_states), buffer_rewards
        )
        buffer_critic_value = self._target_critic_net_on_step(
            torch.FloatTensor(buffer_curr_states)
        )

        critic_loss = F.mse_loss(
            buffer_critic_value, buffer_target_value
        )  # todo: note - i beleive it's mse_loss(buffer_critic_value, buffer_target_value)
        critic_loss.backward(retain_graph=True)

        actor_loss = -(
            buffer_log_probs * (buffer_target_value - buffer_critic_value)
        ).mean()
        actor_loss.backward()
        self.actor_net.optimizer.step()
        self.critic_net.optimizer.step()

        self.soft_update_networks_weights()
        self.all_actor_loss.append(actor_loss)
        self.all_critic_loss.append(critic_loss)
        print(
            f"done train episode #{episode_num}, actor-loss {actor_loss} and critic_loss {critic_loss}"
        )

    def play(self):
        for i in range(EPOCHS):
            self.single_episode(episode_num=i)
            start_time = datetime.now()
            self.learn(episode_num=i)
            learning_duration = (datetime.now() - start_time).total_seconds()
            self.all_learning_durations.append(learning_duration)

            if i % 5:
                self._record_experiment()

    def _choose_action(self, actions_probs):
        action = actions_probs.sample()
        return action

    def _target_critic_net_on_step(self, state):
        self.target_actor_net.eval()
        self.target_critic_net.eval()

        states_batches = torch.Tensor([state]) if len(state.shape) == 3 else state
        actions_probs = self.target_actor_net.forward(states_batches)
        action = self._choose_action(actions_probs)
        action_batch = action[:, None]
        critic_value = self.target_critic_net.forward(states_batches, action_batch)

        return critic_value

    def soft_update_networks_weights(self):
        with torch.no_grad():
            self._soft_update_network(
                main_net=self.critic_net, target_net=self.target_critic_net
            )
            self._soft_update_network(
                main_net=self.actor_net, target_net=self.target_actor_net
            )

    def _soft_update_network(self, main_net, target_net):
        for (param_name, param_values), (
            target_params_name,
            target_params_values,
        ) in zip(main_net.named_parameters(), target_net.named_parameters()):
            assert param_name == target_params_name
            updated_param = (
                self.tau * param_values.clone().detach()
                + (1 - self.tau) * target_params_values.clone().detach()
            )
            param_values.copy_(updated_param)

    def _record_experiment(self):
        if os.path.exists(EXPERIMENT_FILENAME):
            df_experiments = pd.read_csv(EXPERIMENT_FILENAME, index_col=0)
        else:
            df_experiments = pd.DataFrame()

        if len(df_experiments):
            df_experiments = df_experiments[df_experiments["unq_id"] != self.unq_id]

        df_experiments = df_experiments.append(
            [
                {
                    "unq_id": self.unq_id,
                    "env_name": str(self.env.env),
                    "experiment_description": self.experiment_description,
                    "gamma": self.gamma,
                    "tau": self.tau,
                    "training_episodes": len(self.all_rewards_sum),
                    "rewards_sums_array": self.all_rewards_sum,
                    "replay_buffer_sampling_percent": self.replay_buffer_sampling_percent,
                    "total_rewards_sum": sum(self.all_rewards_sum),
                    "learning_durations_seconds": self.all_learning_durations,
                    "total_learning_duration": sum(self.all_learning_durations),
                    "number_of_steps": self.number_of_steps,
                    "total_number_of_steps": sum(self.number_of_steps),
                    "all_actor_loss": self.all_actor_loss,
                    "all_critic_loss": self.all_critic_loss,
                    "car_crashed": self.car_crashed,
                    "start_time": self.start_time,
                    "last_update": datetime.now(),
                }
            ]
        )

        df_experiments.to_csv(EXPERIMENT_FILENAME)

    def _apply_stochastic_env(self, action):
        action_step = action.item()
        if random.random() <= STOCHASTIC_RULE:
            return action_step
        else:
            all_other_actions = [x for x in range(5) if x != action_step]
            return random.choice(all_other_actions)


if __name__ == "__main__":
    highway_agent = Agent(
        buffer_max_size=DEFAULT_BUFFER_MAX_SIZE,
        gamma=0.99,
        tau=0.001,
        replay_buffer_sampling_percent=0.7,
    )
    highway_agent.play()