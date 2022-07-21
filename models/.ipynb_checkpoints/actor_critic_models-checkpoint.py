import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim

torch.manual_seed(100)

# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.uniform_(m.weight,  0, 3*1.e-4)
#         # m.bias.data.fill_(0.0001)



class ActorNet(nn.Module):
    def __init__(self, name="actor"):
        super(ActorNet, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Conv2d(4, 32, (5, 5)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 124 * 124, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

        # self.shared_layers.apply(init_weights)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.name = name
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  

    def forward(self, x):
        input_tensor = torch.FloatTensor(x).to(device=self.device)
        hidden = self.shared_layers(input_tensor)
        return Categorical(F.softmax(hidden, dim=-1))


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.shared_layers = nn.Sequential(  # todo: name
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            # nn.ReLU(),
            # nn.Flatten(start_dim=0),
            # nn.Linear(64 * 12 * 12, 64),
        )

        self.value_layer = nn.Sequential(  # todo: name
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            # convgru?
            nn.Conv2d(64, 1, kernel_size=(3, 3), padding=0),
            # nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(36, 128),
            # nn.ReLU(),
            nn.Linear(36, 128),
        )

        self.action_layer = nn.Linear(1, 128)
        self.action_layer2 = nn.Linear(128, 256)
        self.final_layer = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x, action):
        input_tensor = torch.FloatTensor(x).to(device=self.device)
        hidden = self.shared_layers(input_tensor)
        hidden = self.value_layer(hidden)

        input_action = action.type(torch.FloatTensor).to(device=self.device)
        action_layer = self.action_layer(input_action)  # self.action_layer(action)
        
        state_action_value = F.relu(torch.add(hidden, action_layer))
        state_action_value = F.relu(self.action_layer2(state_action_value))
        state_action_value = self.final_layer(state_action_value)

        return state_action_value
