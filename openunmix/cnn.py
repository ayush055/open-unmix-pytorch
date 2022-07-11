from email.headerregistry import Group
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Conv3d, Linear, GroupNorm, GELU, MaxPool3d, Dropout3d, Dropout, BatchNorm3d, BatchNorm1d

class CNNModel(nn.Module):
    def __init__(self, hidden_size):
        super(CNNModel, self).__init__()

        self.hidden_size = hidden_size

        self.conv1 = self.conv_set(3, 32)
        self.conv2 = self.conv_set(32, 64)
        self.fc1 = Linear(2**3*64, self.hidden_size)
        self.activation = 'tanh'
        self.norm = BatchNorm1d(self.hidden_size)
        self.drop = Dropout(0.5)

    def conv_set(self, c_in, c_out):
        conv_layer = nn.Sequential(
            Conv3d(c_in, c_out, kernel_size=3, padding=0),
            GELU(),            
            MaxPool3d(kernel_size=2),
            BatchNorm3d(c_out),
        )

        return conv_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.norm(x)
        # x = self.drop(x)
        return x