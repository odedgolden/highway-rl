import numpy as np
import torch

from models.icm_model.icm_net import ICMModel
import torch.optim as optim
from torch import nn
import torch.nn.functional as F

class IcmAgent():
    def __init__(self, output_size):
        self.reward_model = ICMModel(input_size=(4,128, 128), output_size=output_size, use_cuda=False)
        self.optimizer = optim.Adam(self.reward_model.parameters(), lr=0.0001)
        self.reverse_scale = 10 # todo
        self.output_size = output_size
        self.eta = 0.01
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train_model(self, states, next_states, actions):
        # reminder: from each

        real_next_state_feature, pred_next_state_feature, pred_action_logit = self.reward_model.forward(
            states, next_states, actions
        )

        # predicted_actions= np.argmax(pred_action_logit.detach(), axis=1)
        ground_truth_class_indices = torch.LongTensor(actions)
        inverse_loss = F.cross_entropy(pred_action_logit, ground_truth_class_indices)
        forward_loss = F.mse_loss(pred_next_state_feature, real_next_state_feature)
        loss =  self.reverse_scale * inverse_loss + forward_loss
        print(f'icm loss: {loss}')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def predict(self):
        pass

    def compute_intrinsic_reward(self, state, next_state, action):
        tensor_state = torch.FloatTensor(state)
        tensor_next_state = torch.FloatTensor(next_state)
        tensor_action = torch.FloatTensor(action)

        real_next_state_feature, pred_next_state_feature, pred_action = self.reward_model.forward(tensor_state, tensor_next_state, tensor_action)
        intrinsic_reward = self.eta * F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction='none').mean(-1)
        return intrinsic_reward


# if __name__ == '__main__':
#