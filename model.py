import math
import torch
import torch.nn as nn


class Model1(nn.Module):
    def __init__(self, num_input, para_1, para_2, para_3, num_output):
        super(Model1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_input, para_1),
            nn.ReLU(),
            nn.Linear(para_1, para_2),
            nn.ReLU(),
            nn.Linear(para_2, para_3),
            nn.ReLU(),
            nn.Linear(para_3, num_output),
        )

    def forward(self, src):
        src = self.model(src)
        return src[:, 0]


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, est_rows, act_rows):
        est_rows = torch.abs(est_rows)
        act = torch.maximum(act_rows, est_rows)
        est = torch.minimum(act_rows, est_rows)
        est = torch.where(est == 0, 1.0, est)
        q_error = torch.div(act, est)

        loss = torch.pow(q_error, 2)
        loss = torch.mean(loss)
        return loss.float()


