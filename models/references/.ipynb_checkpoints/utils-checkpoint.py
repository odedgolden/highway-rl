import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.memory_size = max_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, *input_shape))
        self.new_state_memory = np.zeros((self.memory_size, *input_shape))
        self.action_memory = np.zeros((self.memory_size, n_actions))
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)
        print(self.state_memory.shape)
        print(*input_shape)
    
    def store_transition(self, state, action, reward, new_state, done):
        
        index = self.memory_counter % self.memory_size
        
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        self.memory_counter += 1
        
    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)
        
        batch = np.random.choice(max_memory, batch_size)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones 