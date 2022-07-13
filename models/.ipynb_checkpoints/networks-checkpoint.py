import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class BaseNetwork(nn.Module):
    def __init__(self, name, chkpt_dir, lr):
        super(BaseNetwork, self).__init__()
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_path  = os.path.join(self.checkpoint_dir, name + '_sac') 
        self.lr = lr
        
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_path)
        
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_path))
        
    def figure_out_device(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)        
        


        
class CriticNetwork(BaseNetwork):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__(name=name, chkpt_dir=chkpt_dir, lr=lr)
        
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.fc1 = nn.Linear(self.input_dims[0] + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
        
        self.figure_out_device()
        
    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
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

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)
        
        self.figure_out_device()
        
    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state)
        state_value = F.relu(state_value)
        
        v = self.v(action_value)
        
        return v
    

class ActorNetwork(BaseNetwork):
    def __init__(self, lr, input_dims, max_action, fc1_dims=256, fc2_dims=256, n_actions=2, name='actor', chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__(name=name, chkpt_dir=chkpt_dir, lr=lr)
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.max_action = max_action
        self.reparam_noise = 1e-6
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.figure_out_device()


    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)
        
        return mu, sigma
    
    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probaabilities = Normal(mu, sigma)
        
        if reparameterize:
            actions = probaabilities.rsample()
        else:
            actions = probaabilities.sample()
            
        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        log_probs = probaabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        
        return action, log_probs
