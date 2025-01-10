import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_size = configs.input_size
        self.conv_channels = configs.conv_channels
        self.kernel_sizes = configs.kernel_sizes
        self.pool_sizes = configs.pool_sizes
        self.dropout_rate = configs.dropout_rate
        self.fc_units = configs.fc_units
        self.relu = nn.ReLU()

         # 构建卷积层
        self.convs = nn.ModuleList()
        in_channels = self.input_size
        for i in range(len(self.conv_channels)):
            out_channels = self.conv_channels[i]
            kernel_size = self.kernel_sizes[i]
            pool_size = self.pool_sizes[i]
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=pool_size, stride=1)
                )
            )
            in_channels = out_channels

            
         # 计算卷积层输出的特征维度
        conv_output_size = self._calculate_conv_output_size()

        # 构建全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(conv_output_size, self.fc_units),
            nn.BatchNorm1d(self.fc_units),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        self.fc2 = nn.Linear(self.fc_units, 50)
        self.fc3 = nn.Linear(50, self.pred_len)

        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.input_size, out_channels=32, kernel_size=2),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=1), # seq_len=18
        # )

        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=1),# seq_len=16
        # )

        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=1),# seq_len=16
        # )

        # self.Linear1 = nn.Linear(64 * 16, 50)
        # self.Linear2 = nn.Linear(50, self.pred_len)
        # self.dropout1 = nn.Dropout(p=0.5)
        # self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = torch.flatten(x, start_dim=1)  # 使用 torch.flatten 展平张量
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = x.reshape(x.shape[0], self.pred_len, -1)

        return x

        # x = self.conv1(x)
        # x = self.conv2(x)
        # # print(x.size())  
        # x = x.view(x.size(0), -1)
        # # x = nn.Flatten(x)
        # # print(x.size())
        
        # x = self.Linear1(x)
        # # x = nn.ReLU(x)
        # x = self.dropout1(x)
        # x = self.Linear2(x)
        # # x = nn.ReLU(x)
        # x = self.dropout2(x)
        # return x

    def _calculate_conv_output_size(self):
        # 计算卷积层的输出特征维度
        seq_len = self.seq_len
        for i in range(len(self.conv_channels)):
            kernel_size = self.kernel_sizes[i]
            pool_size = self.pool_sizes[i]
            seq_len = seq_len - kernel_size + 1
            seq_len = seq_len - pool_size + 1
        return seq_len * self.conv_channels[-1]

