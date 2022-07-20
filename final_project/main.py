import copy
from datetime import datetime
from collections import Counter

import gym
import torch.nn.functional as F
import highway_env
import sys

import torch
from torch.distributions import Categorical

from final_project.replay_buffer import ReplayBuffer

TAU_DEFAULT = 0.001

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
file.close()
# ============================================
GAMMA = 0.99
env = gym.make("highway-fast-v0")
config1["duration"] = 500
env.configure(config1)
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

obs = env.reset()


ALL_REWARDS_SUM = []
LAST_NET_STATUS = []
EPOCHS = 500
DEFAULT_BUFFER_MAX_SIZE = 500
FORCE_SHOW = False


class Agent:
    def __init__(self, buffer_max_size=None):
        self.actor_net = actor_critic_models.ActorNet()
        self.critic_net = actor_critic_models.CriticNet()
        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        buffer_max_size = buffer_max_size or DEFAULT_BUFFER_MAX_SIZE
        self.replay_buffer_memory = ReplayBuffer(buffer_max_size=buffer_max_size)

    def single_episode(self, episode_num, force_show=False):
        global FORCE_SHOW
        self.actor_net.eval()
        self.critic_net.eval()
        curr_state = env.reset()
        done = False

        actions_counter = Counter()

        all_probs = []
        all_rewards = []
        sum_rewards = 0
        steps = 0
        all_next_state_values = []
        all_done = []

        while not done:
            FORCE_SHOW = force_show  # hack, allow us choosing if to render during debug
            actions_probs = self.actor_net.forward([curr_state])
            all_probs.append(actions_probs)
            action = self._choose_action(actions_probs)[0]
            next_state, reward, done, extra_info = env.step(action.item())
            if extra_info["crashed"]:
                before = reward
                # reward -=15
                print(f"crashed!! reward was {before} and now {reward}")
            sum_rewards += reward
            all_rewards.append(reward)
            next_state_value = self._target_critic_net_on_step(next_state)
            log_prob = actions_probs.log_prob(action).unsqueeze(0)
            all_next_state_values.append(next_state_value)
            all_done.append(done)

            if (
                (episode_num > 100 and episode_num % 20 == 0)
                or FORCE_SHOW
                or force_show
            ):
                screen = env.render(mode="rgb_array")
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

        ALL_REWARDS_SUM.append(sum_rewards)

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
        print(
            f"done episode number {episode_num} with sum-rewards {sum_rewards} after {steps} steps actions-counter:{actions_counter}"
        )

    def _estimate_step_value(self, done, next_state, reward):
        next_state_value = self._target_critic_net_on_step(next_state)
        reward_tensor = torch.FloatTensor(reward)[:, None]
        done_tensor = torch.FloatTensor(done)[:, None]
        target_value = reward_tensor + GAMMA * next_state_value * (1 - done_tensor)
        return target_value

    def learn(self, episode_num):
        start_time = datetime.now()
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
        duration = (datetime.now() - start_time).total_seconds()
        print(
            f"done train episode #{episode_num}, actor-loss {actor_loss} and critic_loss {critic_loss}, took {duration} seconds"
        )

    def play(self):
        for i in range(EPOCHS):
            self.single_episode(episode_num=i)
            self.learn(episode_num=i)

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
            _soft_update_network(
                main_net=self.critic_net, target_net=self.target_critic_net
            )
            _soft_update_network(
                main_net=self.actor_net, target_net=self.target_actor_net
            )


def _soft_update_network(main_net, target_net):
    for (param_name, param_values), (target_params_name, target_params_values) in zip(
        main_net.named_parameters(), target_net.named_parameters()
    ):
        assert param_name == target_params_name
        updated_param = (
            TAU_DEFAULT * param_values.clone().detach()
            + (1 - TAU_DEFAULT) * target_params_values.clone().detach()
        )
        param_values.copy_(updated_param)


if __name__ == "__main__":
    highway_agent = Agent()
    highway_agent.play()
