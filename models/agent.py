# General
import copy
import os
import random
from datetime import datetime
from collections import Counter
import pandas as pd
import ast
import numpy as np
import uuid
import sys

# Local
from models.replay_buffer import ReplayBuffer, Transition
from models.actor_critic_models import ActorNet, CriticNet

# Gym Env
import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
from gym.utils import seeding
from gym import error, spaces, utils
import highway_env

# Neural Networks
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

sys.path.insert(0, "/content/highway-env/scripts/")
from tqdm.notebook import trange

gymlogger.set_level(40)  # error only
import matplotlib.pyplot as plt

# =============== DO NOT DELETE ===============
file = open("./highway-config/config_ex1.txt", "r")
contents = file.read()
config1 = ast.literal_eval(contents)
config1["duration"] = 500
file.close()
# ============================================

np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


EXPERIMENT_FILENAME = "highway_experiments.csv"


class Agent:
    def __init__(
        self,
        buffer_max_size=500,
        gamma=0.99,
        tau=0.001,
        epochs=500,
        env_stochasticity=0.15,
        experiment_description="",
        replay_buffer_sampling_percent=0.7,
        min_buffer_size_for_learn=300
    ):
        # Experiment Params
        self.gamma = gamma
        self.tau = tau
        self.epochs = epochs
        self.env_stochasticity = env_stochasticity
        self.unq_id = str(uuid.uuid4())  # for tracking experiments
        self.experiment_description = experiment_description
        self.replay_buffer_sampling_percent = replay_buffer_sampling_percent
        self.buffer_max_size = buffer_max_size
        self.min_buffer_size_for_learn = min_buffer_size_for_learn
        self.start_time = datetime.now()

        # Init Logging
        self.all_rewards_sum = []
        self.all_learning_durations = []
        self.number_of_steps = []
        self.car_crashed = []
        self.all_actor_loss = []
        self.all_critic_loss = []

 
        # Init Agent
        self.actor_net = ActorNet()
        self.critic_net = CriticNet()
        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.replay_buffer_memory = ReplayBuffer(
            buffer_max_size=buffer_max_size,
            sampling_percent=replay_buffer_sampling_percent,
            min_buffer_size_for_learn=min_buffer_size_for_learn,
        )
        self._init_env()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def _init_env(self):
        self.env = gym.make("highway-fast-v0")
        self.env.configure(config1)
        obs = self.env.reset()

    def single_episode(self, episode_num):
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
            actions_probs = self.actor_net.forward(np.expand_dims(curr_state, axis=0))
            all_probs.append(actions_probs)
            action = self._choose_action(actions_probs)[0]
            action_step = self._apply_stochastic_env(action)
            next_state, reward, done, extra_info = self.env.step(action_step)
            if extra_info["crashed"]:
                before = reward
                # reward -=15
                # print(f"crashed!! reward was {before} and now {reward}")
                self.car_crashed.append(True)
            else:
                self.car_crashed.append(False)
            sum_rewards += reward
            all_rewards.append(reward)
            next_state_value = self._target_critic_net_on_step(next_state)
            log_prob = actions_probs.log_prob(action).unsqueeze(0)
            all_next_state_values.append(next_state_value)
            all_done.append(done)

            # if episode_num % 10 == 0:
            #     screen = self.env.render(mode="rgb_array")
            #     plt.imshow(screen)

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
        self.number_of_steps.append(steps)
        return sum_rewards, steps, actions_counter

        # print(
        #     f"done episode number {episode_num} with sum-rewards {sum_rewards} after {steps} steps actions-counter:{actions_counter}"
        # )

    def _estimate_step_value(self, done, next_state, reward):
        next_state_value = self._target_critic_net_on_step(next_state)
        reward_tensor = torch.FloatTensor(reward).to(device=self.device)[:, None]
        done_tensor = torch.FloatTensor(done).to(device=self.device)[:, None]
        target_value = reward_tensor + self.gamma * next_state_value * (1 - done_tensor)
        return target_value

    def learn(self, episode_num):
        self.critic_net.train()
        self.actor_net.train()
        self.critic_net.optimizer.zero_grad()
        self.actor_net.optimizer.zero_grad()

        sampled_values = self.replay_buffer_memory.sample_values()
        if not sampled_values:
            return

        current_buffer = Transition(*zip(*sampled_values))

        buffer_curr_states = np.array(current_buffer.curr_states)
        buffer_next_states = np.array(current_buffer.next_states)
        buffer_rewards = current_buffer.rewards
        buffer_dones = current_buffer.dones
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
        # print(
        #     f"done train episode #{episode_num}, actor-loss {actor_loss} and critic_loss {critic_loss}"
        # )

    def play(self):
        self.best_average_score = self.env.reward_range[0]
        self.best_score = self.env.reward_range[0]
        self.score_history = []
        
        for i in range(self.epochs):
            score, steps, actions_counter = self.single_episode(episode_num=i)
            
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])
            self.best_score = max(score, self.best_score)
            self.best_average_score = max(self.best_average_score, avg_score)

            print(f'\nEpisode: {i}, Best Average Score {self.best_average_score}, Average Score: {avg_score}, Best Score: {self.best_score}, Steps: {steps} \n')
            print(f'Action Count: {actions_counter}, Time: {datetime.now()}\n\n')
            
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

        states_batches = torch.Tensor(np.expand_dims(state, axis=0)) if len(state.shape) == 3 else state
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
                    "min_buffer_size_for_learn": self.min_buffer_size_for_learn,
                    "buffer_max_size": self.buffer_max_size,
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
        if random.random() > self.env_stochasticity:
            return action_step
        else:
            all_other_actions = [x for x in range(5) if x != action_step]
            return random.choice(all_other_actions)
