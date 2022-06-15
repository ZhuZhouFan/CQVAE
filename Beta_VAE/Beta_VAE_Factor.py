import numpy as np
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from Network import VAE_Factor_Network


class VAE_Factor_Agent():
    def __init__(self, N, T, P, K, f_hidden_dim, model_para, learning_rate, seed, log_dir):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.network = VAE_Factor_Network(N, T, P, K, f_hidden_dim, model_para)
        self.network.train()
        self.optimizer = torch.optim.Adam(self.network.factor_loading_network.parameters(), lr = learning_rate)
        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr = learning_rate)
        self.loss_function = torch.nn.MSELoss(reduction = 'sum')
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.writer = SummaryWriter(log_dir = self.log_dir)
        self.best_score = 1e10
        self.P = P
        self.T = T
        self.N = N
        self.early_stopping_count = 0

    def update_params(self, optim, loss, networks, retain_graph=False, grad_cliping = None):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if grad_cliping:
            for net in networks:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_cliping)
        optim.step()

    def load_data(self, C, r, valid_size, test_size, num_cpu, batch_size):
        print('Begining loading...')
        assert (self.N, self.T, self.P) == C.shape, 'Plz check the shape of C'
        assert (self.N, self.T) == r.shape, 'Plz check the shape of r'
        self.batch_size = batch_size
        index_1 = round(self.T * (1 - valid_size - test_size))
        index_2 = round(self.T * (1 - test_size))
        C = C.swapaxes(0, 1)
        r = r.transpose()
        r = r[:, :, np.newaxis]
        C = np.concatenate((C, r), axis = 2)
        feature_train = torch.Tensor(C)[:index_1, :, :]
        label_train = torch.Tensor(r)[:index_1, :]
        feature_valid = torch.Tensor(C)[index_1: index_2, :, :]
        label_valid = torch.Tensor(r)[index_1: index_2, :]
        self.feature_test = torch.Tensor(C)[index_2:, :, :]
        self.label_test = torch.Tensor(r)[index_2:, :]
        dataset_train = TensorDataset(feature_train, label_train)
        self.train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = num_cpu)
        dataset_valid = TensorDataset(feature_valid, label_valid)
        self.valid_loader = DataLoader(dataset_valid, batch_size = batch_size, shuffle = True, num_workers = num_cpu)
        self.sample_num_train = feature_train.shape[0]
        self.sample_num_valid = feature_valid.shape[0]
        self.sample_num_test = self.feature_test.shape[0]
        print(f'The data contains {self.sample_num_train} training samples, {self.sample_num_valid} validation samples and {self.sample_num_test} test samples')
        print("Complete!")

    def train(self, epoch_num, lam):
        print("Start training VAE Factor Model...")
        self.network.train()
        for i in range(epoch_num):
            start_time = time.time()
            train_losses = list()
            for j, (x, y) in enumerate(self.train_loader):
                x = x.to(self.network.device)
                y = y.to(self.network.device)
                y_hat = self.network.forward(x)
                L1_loss = 0
                for param in self.network.factor_loading_network.parameters():
                    L1_loss += torch.sum(torch.abs(param))
                loss = self.loss_function(y_hat, y) + lam * self.N * self.batch_size * L1_loss
                train_losses.append(loss.item())
                self.update_params(self.optimizer, loss, networks=[self.network.factor_loading_network], retain_graph = False)
            end_time = time.time()
            dtime = end_time - start_time
            train_loss = torch.tensor(train_losses).mean()
            print(f'Epoch {i}/{epoch_num-1} Training Loss {train_loss:.2f} Time Consume {dtime:.3f}')
            self.writer.add_scalar('loss/train', train_loss, i)
            if i % 10 == 0:
                self.evaluate(i, epoch_num, lam)
                # torch.save(self.network.state_dict(), os.path.join(self.log_dir,'VAEF_final.pth'))
            if self.early_stopping_count >= 10:
                break

    def evaluate(self, i, epoch_num, lam):
        self.network.eval()
        with torch.no_grad():
            valid_losses = list()
            for j,(x,y) in enumerate(self.valid_loader):
                x = x.to(self.network.device)
                y = y.to(self.network.device)
                y_hat = self.network.forward(x)
                L1_loss = 0
                for param in self.network.factor_loading_network.parameters():
                    L1_loss += torch.sum(torch.abs(param))
                loss = self.loss_function(y_hat, y) + lam * self.N * self.batch_size * L1_loss
                valid_losses.append(loss.item())
            valid_loss = torch.tensor(valid_losses).mean()
        self.writer.add_scalar('loss/valid', valid_loss.item(), i)
        print("-"*60)
        print(f'Evaluation {i}/{epoch_num-1} Loss {valid_loss.item():.2f}')
        if valid_loss.item() < self.best_score:
            print('Update Model')
            self.best_score = valid_loss.item()
            torch.save(self.network.state_dict(), os.path.join(self.log_dir, 'VAEF_best.pth'))
            self.early_stopping_count = 0
        else:
            self.early_stopping_count += 1
        print("-"*60)
        self.network.train()