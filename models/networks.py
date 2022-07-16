import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
# from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


import numpy as np


class BaseNetwork(nn.Module):
    def __init__(self, name, chkpt_dir, lr):
        super(BaseNetwork, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_path  = os.path.join(self.checkpoint_dir, name + '_sac') 
        self.lr = lr
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.flat = nn.Flatten(start_dim=1)
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_path)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_path))
        
    def figure_out_device(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)
    
    def forward(self, state):
        state_value = self.conv1(state)
        state_value = F.relu(state_value)
        state_value = self.conv2(state_value)
        state_value = F.relu(state_value)
        state_value = self.conv3(state_value)
        state_value = F.relu(state_value)
        state_value = self.flat(state_value)

        return state_value


        
class CriticNetwork(BaseNetwork):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__(name=name, chkpt_dir=chkpt_dir, lr=lr)
        
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims


        self.fc1 = nn.Linear(64 * 12 * 12 + 1, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
        
        self.figure_out_device()
        
    def forward(self, state, action):
        state_value = super().forward(state)
        
        # print(f'state_value.shape: {state_value.size()}')
        # print(f'action.shape: {action.size()}')
        
        action_value = T.cat([state_value, action], dim=1)
        
        # print(f'action_value.shape: {action_value.size()}')
            
        action_value = self.fc1(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        
        q = self.q(action_value)
        
        return q

class ValueNetwork(BaseNetwork):
    def __init__(self, lr, input_dims, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__(name=name, chkpt_dir=chkpt_dir, lr=lr)
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(64 * 12 * 12 , self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)
        
        self.figure_out_device()
        
    def forward(self, state):
        state_value = super().forward(state)
        state_value = self.fc1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        
        v = self.v(state_value)
        
        return v
    

class ActorNetwork(BaseNetwork):
    def __init__(self, lr, input_dims, max_action, fc1_dims=256, fc2_dims=256, n_actions=5, name='actor', chkpt_dir='tmp/sac', droput_p=0.3):
        super(ActorNetwork, self).__init__(name=name, chkpt_dir=chkpt_dir, lr=lr)
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.max_action = max_action
        self.reparam_noise = 1e-6
        
        self.fc1 = nn.Linear(64 * 12 * 12 , self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.probs = nn.Linear(self.fc2_dims, self.n_actions)        
        self.drop = nn.Dropout(p=droput_p)

        self.figure_out_device()


    def forward(self, state):
        prob = super().forward(state)
        prob = self.fc1(prob)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        prob = self.probs(prob)
        prob = F.relu(prob)
        ptob = self.drop(prob)
        prob = F.softmax(prob)        
        return prob
    
    def sample_categorical(self, state):

        probabilities = self.forward(state)
        # print(f'probabilities.size(): {probabilities.size()}')
        # print(f'probabilities: {probabilities}')        
        # Since we are dealing with discrete actions:
        distribution = Categorical(probabilities)
        action = distribution.sample()
        # print(f'action.size(): {action.size()}')
        log_probs = distribution.log_prob(action)

        # Not sure if this is relevant:
        
        # log_probs -= T.log(1-action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(-1, keepdim=True)
        
        ######
        

        
        return action.unsqueeze(1), log_probs
