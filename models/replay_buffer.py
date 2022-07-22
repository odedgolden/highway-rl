import random
from collections import deque, namedtuple

import numpy as np
import torch
Transition = namedtuple('Transition',
                        ("curr_states", "actions", "next_states", "rewards", "dones", "log_probs"))

class ReplayBuffer:
    buffers = {"curr_states", "actions", "next_states", "rewards", "dones", "log_probs"}

    def __init__(self, buffer_max_size=500, sampling_percent=None):
        assert sampling_percent
        self.memory = deque([], maxlen=buffer_max_size)
        self.log_probs = []
        self.buffer_max_size = buffer_max_size
        self.sampling_percent = sampling_percent

    def add(self, **kargs):
        self.memory.append(Transition(**kargs))

    def sample_values(self):
        curr_buffer_size = len(self.memory)
        if curr_buffer_size < 30:
            print(f"not enough data for train {curr_buffer_size} values")
            return []
        batch_size = (
            round(self.sampling_percent * curr_buffer_size)
            if curr_buffer_size > 150
            else curr_buffer_size // 2
        )
        # print(f"training on replay buffer of size {batch_size}\{curr_buffer_size}")

        sampled_torches = random.sample(self.memory, batch_size)
        return sampled_torches