import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# from utils.timefeatures import time_features
import warnings

class TimeSeriesDataset(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='temp.csv',
                 target='OT', scale_x=True, scale_y=False):
        super().__init__()
        # size [seq_len, label_len, pred_len]
        # info
        if size == None: 
            self.seq_len = 20 * 2
            self.label_len = 0 # 默认序列不重合
            self.pred_len = 1 # 默认预测一个label
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale_x = scale_x
        self.scale_y = scale_y


        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['timestamp', ...(other features), target feature]
        '''
        
        # 确保输入的格式正确
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('TimeStamp')
        df_raw = df_raw[['TimeStamp'] + cols + [self.target]]

        # 划分训练集,验证集和测试集
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = int(len(df_raw)) - num_test - num_train
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 确定输入特征是单变量S or 多变量M or 多变量和时间戳MS
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:] # 剔除了时间列
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 归一化, 对x
        if self.scale_x:
            train_data_x = df_data.iloc[border1s[0]:border2s[0], :-1]
            self.scaler_x.fit(train_data_x.values)
            data_x = self.scaler_x.transform(df_data.iloc[:, :-1].values)
        else:
            data_x = df_data.iloc[:, :-1].values

        # 归一化, 对y
        if self.scale_y:
            train_data_y = df_data.iloc[border1s[0]:border2s[0], -1]
            self.scaler_y.fit(train_data_y.values.reshape(-1,1))
            data_y = self.scaler_y.transform(df_data.iloc[:, -1].values.reshape(-1,1))
        else:
            data_y = df_data.iloc[:, -1].values.reshape(-1,1)

        # df_stamp = df_raw[['TimeStamp']][border1:border2]
        # df_stamp['timestamp'] = pd.to_datetime(df_stamp.timestamp)

        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]

        print(data_x.shape)
        # self.data_x = np.zeros((100,5))
        # self.data_y = np.zeros((100,5))

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len - 1 # 目标序列和输入序列重合一部分，这样子可以学习到序列的连续性，默认不重合
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        if len(seq_y.shape) == 1:
            seq_y = seq_y.reshape(-1,1)

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1 + 1

    def inverse_transform_x(self, data):
        return self.scaler_x.inverse_transform(data)

    def inverse_transform_y(self, data):
        return self.scaler_y.inverse_transform(data)