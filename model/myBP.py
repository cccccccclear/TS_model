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
        self.hidden_units = configs.hidden_units  # 隐藏层单元数列表
        self.dropout_rate = configs.dropout_rate

        # 构建全连接层
        self.fc_layers = nn.ModuleList()
        in_features = self.seq_len * self.input_size
        for hidden_units in self.hidden_units:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, hidden_units),
                    nn.BatchNorm1d(hidden_units),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate)
                )
            )
            in_features = hidden_units

            
        # 最后一层全连接层
        self.fc_final = nn.Linear(in_features, self.pred_len)


    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # 将输入展平为一维向量
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        x = self.fc_final(x)
        x = x.reshape(x.shape[0], self.pred_len, -1)
        return x


   

