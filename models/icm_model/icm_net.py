import torch
import torch.nn as nn


class ICMModel(nn.Module):
    def __init__(self, input_size, output_size, use_cuda=True):
        super(ICMModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        action_size = 1
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        feature_output = 12 * 12 * 64
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(feature_output, 512)
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Softmax()
        )

        self.residual = [nn.Sequential(
            nn.Linear(action_size + 512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        ).to(self.device)] * 8

        self.forward_net_1 = nn.Sequential(  # get action+feature, and output new feature
            nn.Linear(action_size + 512, 512),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(action_size + 512, 512),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, states, next_states, actions):
        torch_state = torch.FloatTensor(states).to(device=self.device)
        torch_next_state = torch.FloatTensor(next_states).to(device=self.device)
        torch_action = torch.FloatTensor(actions).to(device=self.device)

        encode_state = self.feature(torch_state)
        encode_next_state = self.feature(torch_next_state)
        # get pred action
        encoded_state_and_next_state = torch.cat((encode_state, encode_next_state), 1)
        pred_action_probs = self.inverse_net(encoded_state_and_next_state)
        # ---------------------

        # get pred next state
        torch_action_transpose = torch_action[:, None]
        state_features_and_action = torch.cat((encode_state, torch_action_transpose), 1)
        pred_next_state_feature_orig = self.forward_net_1(state_features_and_action)

        # # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, torch_action_transpose), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, torch_action_transpose), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, torch_action_transpose), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action_probs