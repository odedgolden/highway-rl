import os
import torch as T
import torch.nn.functional as F
import numpy as np
from .utils import ReplayBuffer
from .networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, 
                 alpha=0.0003,
                 beta=0.0003, 
                 input_dims=[8], 
                 env=None, 
                 gamma=0.99, 
                 n_actions=2, 
                 max_size=100000, 
                 tau=0.005, 
                 layer1_size=256, 
                 layer2_size=256, 
                 batch_size=256, 
                 reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        # print(input_dims)
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        
        self.actor = ActorNetwork(lr=alpha,
                                  input_dims=input_dims, 
                                  n_actions=n_actions, 
                                  name='actor', 
                                  max_action=n_actions-1)
        
        self.critic_1 = CriticNetwork(lr=beta,
                                      input_dims=input_dims, 
                                      n_actions=n_actions,
                                      name='critic_1')
        self.critic_2 = CriticNetwork(lr=beta,
                                      input_dims=input_dims,
                                      n_actions=n_actions,
                                      name='critic_2')
        self.value = ValueNetwork(lr=beta, 
                                  input_dims=input_dims, 
                                  name='value')
        self.target_value = ValueNetwork(lr=beta, 
                                         input_dims=input_dims, 
                                         name='target_value')
        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        
    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        actions = actions.cpu().detach().numpy()[0]
        action = np.argmax(actions)
        return action
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def update_network_parameters(self, tau=None, learning=False):
        if tau is None:
            tau = self.tau
            
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()
        
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)
        
        # if learning:
        #     print(f'\ntarget_value_state_dict: {target_value_state_dict}\n')
        #     print(f'\value_state_dict: {value_state_dict}\n')
        #     print(f'\ntau: {tau}\n')
        
        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()
            
        self.target_value.load_state_dict(value_state_dict)
        
    def save_models(self):
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        
    def load_models(self):
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        
    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return
        # print("\n\nLearning...\n\n")
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        
        value = self.value(state).view(-1)
        target_value = self.target_value(new_state).view(-1)
        target_value[done] = 0.0
        
        # print("\nself.actor.sample_normal(state, reparameterize=False)\n")
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        self.value.optimizer.zero_grad()
        # print(f'critic_value.shape: {critic_value.shape}')
        # print(f'log_probs.shape: {log_probs.shape}')
        # print(f'\n T.isnan(critic_value).any(): {T.isnan(critic_value).any()}\n')
        # print(f'\n T.isnan(log_probs).any(): {T.isnan(log_probs).any()}\n')        
        value_target = critic_value - log_probs
        
        # sanity_check = (value == 0).any() or (value_target == 0).any()
        # sanity_check = T.isnan(value).any() or T.isnan(value_target).any()        
        # print(f'\n sanity_check: {sanity_check}\n')
        # print(f'\n T.isnan(value).any(): {T.isnan(value).any()}\n')
        # print(f'\n T.isnan(value_target).any(): {T.isnan(value_target).any()}\n')
        
        value_loss = 0.5*F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()
        # print("\nDone 1\n")
        
        # print("\nself.actor.sample_normal(state, reparameterize=True)\n")        
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)        
        
        # print("\nDone 2\n")
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q_hat = self.scale*reward + self.gamma*target_value
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5*F.mse_loss(q2_old_policy, q_hat)
        
        critic_loss = critic_1_loss + critic_2_loss
        
        # print("\nGoing backward...\n")
        critic_loss.backward()

        # print("\nOptimizer 1 step...\n")
        self.critic_1.optimizer.step()
        # print("\nOptimizer 2 step...\n")        
        self.critic_2.optimizer.step()
        # print("\nupdate_network_parameters...\n")        
        self.update_network_parameters(learning=True)