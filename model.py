import math
import torch
import torch.nn as nn


class Model1(nn.Module):
    def __init__(self, num_i, num_h, output):
        super(Model1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_i, num_h),
            nn.ReLU(),
            nn.Linear(num_h, num_h),
            nn.ReLU(),
            nn.Linear(num_h, output),
        )

    def forward(self, src):
        return self.model(src)
