import numpy as np
import torch


class ReplayBuffer:
    buffers = {"curr_states", "actions", "next_states", "rewards", "dones", "log_probs"}

    def __init__(self, buffer_max_size=500):
        self.actions = []
        self.curr_states = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.buffer_max_size = buffer_max_size

    def add(self, **kwargs):
        for key, value in kwargs.items():
            assert key in self.buffers
            current_buffer = getattr(self, key)
            current_buffer.append(value)
            if len(current_buffer) > self.buffer_max_size:
                setattr(self, key, current_buffer[-self.buffer_max_size :])

    def sample_values(self):
        assert len({len(getattr(self, key)) for key in self.buffers}) == 1
        curr_buffer_size = len(self.actions)
        if curr_buffer_size < 30:
            print(f"not enough data for train {curr_buffer_size} values")
            return []
        batch_size = (
            round(0.7 * curr_buffer_size)
            if curr_buffer_size > 150
            else curr_buffer_size // 2
        )
        # print(f"training on replay buffer of size {batch_size}\{curr_buffer_size}")
        batch_indexes = np.random.choice(
            curr_buffer_size, batch_size, replace=False
        )  # todo

        sampled_torches = {
            "curr_states": [
                x for i, x in enumerate(self.curr_states) if i in batch_indexes
            ],
            "actions": [
                x.detach() for i, x in enumerate(self.actions) if i in batch_indexes
            ],
            "next_states": [
                x for i, x in enumerate(self.next_states) if i in batch_indexes
            ],
            "rewards": [x for i, x in enumerate(self.rewards) if i in batch_indexes],
            "dones": [x for i, x in enumerate(self.dones) if i in batch_indexes],
            "log_probs": torch.stack(self.log_probs)[batch_indexes],
        }

        for key in self.buffers:
            setattr(self, key, [])

        return sampled_torches
