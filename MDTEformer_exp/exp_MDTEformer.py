from data.data_loader import Dataset_MTS
from MDTEformer_exp.exp_basic import Exp_Basic
from MDTEformer_models.MDTEformer import MDTEformer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import os
import time
import json
import pickle
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

class Exp_MDTEformer(Exp_Basic):
    def __init__(self, args):
        super(Exp_MDTEformer, self).__init__(args)
    
    def _build_model(self):        
        model = MDTEformer(
            self.args.data_dim, 
            self.args.in_len, 
            self.args.out_len,
            self.args.seg_len,
            self.args.win_size,
            self.args.factor,
            self.args.d_model, 
            self.args.d_ff,
            self.args.n_heads, 
            self.args.e_layers,
            self.args.dropout, 
            self.args.baseline,
            self.device
        ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size;
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size;
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.in_len, args.out_len],  
            data_split = args.data_split,
        )

        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        scale_statistic = {'mean': train_data.scaler.mean, 'std': train_data.scaler.std}
        with open(os.path.join(path, "scale_statistic.pkl"), 'wb') as f:
            pickle.dump(scale_statistic, f)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        
        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            old_i = 0
            old_loss = 0
            train_loss = []
            time_step_list = []
            train_loss_list = []
            
            self.model.train()
            epoch_time = time.time()
            batch_size = self.args.batch_size
            i=0
            while i<len(train_data)-128:
                iter_count += 1
                #print("batch_size:", batch_size)
                
                model_optim.zero_grad()
                batch_x, batch_y = self._get_batch_data(train_data, i, batch_size)
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y)

                loss = criterion(pred[:, :, 0], true[:, :, 0])
                
                train_loss.append(loss.item())
                train_loss_list.append(loss.item())
                time_step_list.append(batch_size)
                
                if (iter_count+1) % 100==0:
                    print("\ti: {0}, epoch: {1} | loss: {2:.7f}| time_steps_avg: {3:.2f}".format(i, epoch + 1, loss.item(), (i- old_i)/iter_count))
                    speed = (time.time()-time_now)/(i-old_i)*100
                    left_time = speed/100 * ((self.args.train_epochs - epoch)*len(train_data) + len(train_data) -i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    old_i = i
                
                loss.backward()
                model_optim.step()

                i = i + batch_size

                if (batch_size < self.args.batch_size_max) and (loss.item() > old_loss):
                    batch_size = int(batch_size *2)
                if ((batch_size > self.args.batch_size_min) and (loss.item() < old_loss)):
                    batch_size = int(batch_size /2)

                old_loss = loss.item()
            
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'checkpoint.pth')

        # # 创建 DataFrame（每个列表作为一行）
        # df = pd.DataFrame([train_loss_list, time_step_list])
        # # 可选：添加行标签（例如行名）
        # df.index = ["train_loss", "time_step"]
        # # 写入 CSV 文件（保留行标签）
        # df.to_csv("output_pandas.csv", encoding="utf-8")
        
        return self.model

    def test(self, setting, save_pred = False, inverse = True):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        metrics_all = []
        instance_num = 0
        
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, inverse)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, inverse = True):

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)

        if inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)

        return outputs, batch_y
    
    def eval(self, setting, save_pred = False, inverse = False):
        #evaluate a saved model
        args = self.args
        data_set = Dataset_MTS(
            root_path=args.root_path,
            data_path=args.data_path,
            flag='test',
            size=[args.in_len, args.out_len],
            data_split = args.data_split,
            scale = True,
            scale_statistic = args.scale_statistic,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)

        self.model.eval()

        preds = []
        trues = []
        metrics_all = []
        instance_num = 0

        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(data_loader):
                pred, true = self._process_one_batch(
                    data_set, batch_x, batch_y, inverse)
                batch_size = pred.shape[0]
                instance_num += batch_size
                batch_metric = np.array(metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)
                if (save_pred):
                    preds.append(data_set.inverse_transform(pred.detach().cpu().numpy()))
                    trues.append(data_set.inverse_transform(true.detach().cpu().numpy()))

        metrics_all = np.stack(metrics_all, axis = 0)
        metrics_mean = metrics_all.sum(axis = 0) / instance_num

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if (save_pred):
            preds = np.concatenate(preds, axis = 0)
            trues = np.concatenate(trues, axis = 0)
            np.save(folder_path+'pred.npy', preds)
            np.save(folder_path+'true.npy', trues)

        return mae, mse, rmse, mape, mspe

    def _get_batch_data(self, train_data, i, batch_size):
        x = torch.zeros((batch_size, self.args.in_len, self.args.data_dim))
        y = torch.zeros((batch_size, self.args.out_len, self.args.data_dim))
        k=0
        for j in range(i, i + batch_size):
            x_data, y_data = train_data[j]
            x[k], y[k] = torch.from_numpy(x_data), torch.from_numpy(y_data)
            k=k+1
        return x,y