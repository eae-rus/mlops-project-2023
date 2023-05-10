import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, window_size: int):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * window_size, window_size),
            nn.ReLU(),
            nn.Linear(window_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.layers(x)
