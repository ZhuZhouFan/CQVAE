import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from Network import VAE_Network


class Beta_VAE_Agent():
    def __init__(self, input_dim, latent_dim, output_dim, beta, learning_rate, seed, log_dir):
        self.input_dim = input_dim
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.network = VAE_Network(input_dim, latent_dim, output_dim)
        self.network.train()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = learning_rate)
        self.beta = beta
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.writer = SummaryWriter(log_dir = self.log_dir)
        self.best_score = 1e10
        self.early_stopping_count = 0

    def update_params(self, optim, loss, networks, retain_graph = False, grad_cliping = None):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if grad_cliping:
            for net in networks:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_cliping)
        optim.step()

    def loss_function(self, x, x_hat, mean, log_var):
        reproduction_loss = torch.nn.MSELoss(reduction = 'sum')(x_hat, x)
        KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + self.beta * KLD

    def load_data(self, feature, label, valid_size, test_size, num_cpu, batch_size):
        print('Begining loading...')
        self.batch_size = batch_size
        T = feature.shape[0]
        index_1 = round(T * (1 - valid_size - test_size))
        index_2 = round(T * (1 - test_size))
        feature_train = feature[:index_1, :]
        label_train = label[:index_1, :]
        feature_valid = feature[index_1: index_2, :]
        label_valid = label[index_1: index_2, :]
        dataset_train = TensorDataset(feature_train, label_train)
        self.train_loader = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = num_cpu)
        dataset_valid = TensorDataset(feature_valid, label_valid)
        self.valid_loader = DataLoader(dataset_valid, batch_size = batch_size, shuffle = True, num_workers = num_cpu)
        self.sample_num_train = feature_train.shape[0]
        self.sample_num_valid = feature_valid.shape[0]
        print(f'The data contains {self.sample_num_train} training samples and {self.sample_num_valid} valid samples')
        print("Complete!")

    def train(self, epoch_num, lam):
        # print("Start training VAE...")
        self.network.train()
        for i in range(epoch_num):
            train_losses = list()
            for j, (x, _) in enumerate(self.train_loader):
                x = x.to(self.network.device)
                x_hat, mean, log_var = self.network.forward(x)
                L1_loss = 0
                for param in self.network.parameters():
                    L1_loss += torch.sum(torch.abs(param))
                loss = self.loss_function(x, x_hat, mean, log_var) + L1_loss * self.batch_size * lam
                train_losses.append(loss.item())
                self.update_params(self.optimizer, loss, networks=[self.network])
            train_loss = torch.tensor(train_losses).mean()
            # print(f'Epoch {i}/{epoch_num-1} Training Loss {train_loss:.2f} Time Consume {dtime:.3f}')
            self.writer.add_scalar('loss/train', train_loss, i)
            if i % 10 == 0:
                self.evaluate(i, epoch_num)
                torch.save(self.network.state_dict(), os.path.join(self.log_dir, 'Beta_VAE_final.pth'))
            if self.early_stopping_count >= 10:
                break
        print(f'best score:{self.best_score}')

    def evaluate(self, i, epoch_num):
        self.network.eval()
        with torch.no_grad():
            valid_losses = list()
            for j, (x, y) in enumerate(self.valid_loader):
                x = x.to(self.network.device)
                x_hat, mean, log_var = self.network.forward(x)
                loss = torch.nn.MSELoss(reduction = 'sum')(x_hat, x)
                valid_losses.append(loss.item())
            valid_loss = torch.tensor(valid_losses).mean()
        self.writer.add_scalar('loss/valid', valid_loss.item(), i)
        # print("-"*60)
        # print(f'Evaluation {i}/{epoch_num-1} Loss {test_loss.item():.2f}')
        # print("-"*60)
        if valid_loss.item() < self.best_score:
            # print('Update Model')
            self.early_stopping_count = 0
            self.best_score = valid_loss.item()
            torch.save(self.network.state_dict(), os.path.join(self.log_dir, 'Beta_VAE_best.pth'))
        else:
            self.early_stopping_count += 1

        self.network.train()