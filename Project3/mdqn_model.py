#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn, Tensor
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
                i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        """ self.conv1 = nn.Conv2d(4,16,kernel_size=5,stride =2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,kernel_size =5,stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,kernel_size =5,stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc4 = nn.Linear(7 * 7 * 32, 512)
        self.fc5 = nn.Linear(512,num_actions) """

        self.num_actions = num_actions
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Expected (sample) dummy input = zeros(batch_size, in_channels, 84, 84)
        h_out = w_out = self._conv2d_size_out(
            self._conv2d_size_out(self._conv2d_size_out(84, 8, 4), 4, 2),
            3,
            1
        )
        no_filters_last_conv_layer = 64

        self.in_features = int(h_out * w_out * no_filters_last_conv_layer)

        self.fc_stack = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )

    # Get the no. of features in the output of the conv-relu-layers-stack which
    # is required to be known for the Linear layer 'in_features' arg.

    # Following is simplified version. Checkout link below for the detailed one
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    @staticmethod
    def _conv2d_size_out(size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) / stride + 1



    def forward(self, obs):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        """ x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.view(x.size(0),-1)))
        x = F.relu(self.fc5(x)) """

        obs = obs.to(device)
        intermediate_output = self.conv_relu_stack(obs)
        intermediate_output = intermediate_output.view(obs.size()[0], -1)
        return self.fc_stack(intermediate_output)
        ###########################
        #return x
