from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.evaluates import evaluate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x = batch_x.permute(0, 2, 1)  # 调整形状为 (batch_size, channels, seq_len)
                batch_y = batch_y.permute(0, 2, 1)  # 调整形状为 (batch_size, channels, label_len + pred_len)

                outputs = self.model(batch_x) # 输出形状应为 (batch_size, pred_len, n_features)

                # 根据数据任务选择输出的样式
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss



    def train(self, settings):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        # 初始化误差列表
        train_losses = []
        vali_losses = []
        test_losses = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x = batch_x.permute(0, 2, 1)  # 调整形状为 (batch_size, channels, seq_len)
                batch_y = batch_y.permute(0, 2, 1)  # 调整形状为 (batch_size, channels, label_len + pred_len)

                iter_count += 1
                model_optim.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # 保存误差
            train_losses.append(train_loss)
            vali_losses.append(vali_loss)
            test_losses.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.10f} Vali Loss: {3:.10f} Test Loss: {4:.10f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # train_loss = np.average(train_loss)
            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # print(f'Epoch {epoch+1}, Loss: {train_loss}')
        
        # 绘制误差曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(vali_losses, label='Validation Loss', color='green')
        plt.plot(test_losses, label='Test Loss', color='red')
        plt.title('Training, Validation, and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # plt.show()
        plt.savefig('./pic/loss_curve.png')

        # 保存训练好的模型
        torch.save(self.model.state_dict(), 'model.pth')
        return self.model


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load('model.pth'))

        # preds = []
        # trues = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        criterion = self._select_criterion

        self.model.eval()  # 将模型设置为评估模式
        test_loss = []
        test_preds = []
        test_trues = []
            
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x = batch_x.permute(0, 2, 1)  # 调整形状为 (batch_size, channels, seq_len)
                batch_y = batch_y.permute(0, 2, 1)  # 调整形状为 (batch_size, channels, label_len + pred_len)

                outputs = self.model(batch_x) # 输出形状应为 (batch_size, pred_len, n_features)

                # 根据数据任务选择输出的样式
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                # 反归一化处理
                if test_data.scale_y and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform_y(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform_y(batch_y.squeeze(0)).reshape(shape)
                
                pred = outputs
                true = batch_y

                test_preds.append(pred)
                test_trues.append(true)


        test_preds = np.concatenate(test_preds, axis=0)
        test_trues = np.concatenate(test_trues, axis=0)

        mae, mse, rmse, mape, mspe, r_squared, accu = evaluate(test_preds, test_trues)

        print('test shape:', test_preds.shape, test_trues.shape)
        print('mse:{}, mae:{}, r_squared:{}, accu:{}'.format(mse, mae, r_squared, accu))

        

        return 


    
    