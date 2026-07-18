import torch.nn as nn


class FeedForwardNN(nn.Module):
    """Landmark classifier shared by train.py, the demo apps, and the tests.

    This is the single source of truth for the architecture: the saved
    state_dict in best_model.pth only loads if every consumer builds the
    exact same layers, so nobody redefines this class locally.
    """

    def __init__(self, input_dim=63, output_dim=5):
        super(FeedForwardNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)
