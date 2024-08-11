import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, action_dim, hidden_dim = 512):
        super(DQN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=8,
                      stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features = 64 * 7 * 7, out_features = hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features = 512, out_features = action_dim))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    action_dim = 2
    net = DQN(action_dim)
    state = torch.randn(1, 4, 84, 84)
    output = net(state)
    print(output)
