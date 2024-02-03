import torch
import torch.nn as nn



class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.epsilon = 1.e-8

    def forward(self, y_pred, y_true):
        mse = nn.MSELoss()
        return torch.sqrt(mse(y_pred, y_true)+ self.epsilon).clone()
