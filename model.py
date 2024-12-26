from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(9, 64),
            nn.Dropout(0.2),
            nn.LogSoftmax(dim=1),
            nn.Linear(64, 256),
            nn.Dropout(0.2),
            nn.LogSoftmax(dim=1),
            nn.Linear(256, 4),
            nn.LogSoftmax(dim=1),
        )
    def forward(self, x):
        x = self.sequential(x)
        return x