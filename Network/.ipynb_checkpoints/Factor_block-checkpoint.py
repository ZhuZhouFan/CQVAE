import torch
import torch.nn as nn
from Network import AE_Network, VAE_Network


class Factor_Loading_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, layers):
        super(Factor_Loading_Network, self).__init__()
        self.layers = layers
        self.latent_dim = latent_dim
        self.FC1 = nn.Linear(input_dim, hidden_dim)
        self.FC2 = nn.Linear(hidden_dim, latent_dim)
        self.BN = nn.BatchNorm1d(input_dim)
        self.Relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.BN(x)
        x = torch.transpose(x, 2, 1)
        for i in range(self.layers):
            y = self.Relu(self.FC1(x[:, i, :]))
            y = torch.unsqueeze(self.FC2(y), 1)
            if i == 0:
                y_hat = y
            else:
                y_hat = torch.cat([y_hat, y], axis = 1)
        return y_hat


class AE_Factor_Network(nn.Module):
    def __init__(self, N, T, P, K, f_hidden_dim, model_para):
        super(AE_Factor_Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.factor_loading_network = Factor_Loading_Network(input_dim = P,
                                                             hidden_dim = f_hidden_dim,
                                                             latent_dim = K,
                                                             layers = N).to(self.device)
        self.AE = AE_Network(input_dim = N, 
                             latent_dim = K, 
                             output_dim = N).to(self.device)
        self.AE.load_state_dict(model_para)
        self.Encoder = self.AE.Encoder
        self.P = P

    def forward(self, x):
        x1 = x[:, :, :self.P]
        x2 = x[:, :, self.P:]
        x2 = torch.squeeze(x2, 2)
        factor_loading = self.factor_loading_network(x1)
        factor_return = torch.unsqueeze(self.Encoder(x2), axis = 2)
        return torch.bmm(factor_loading, factor_return)


class VAE_Factor_Network(nn.Module):
    def __init__(self, N, T, P, K, f_hidden_dim, model_para):
        super(VAE_Factor_Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.factor_loading_network = Factor_Loading_Network(input_dim = P,
                                                             hidden_dim = f_hidden_dim,
                                                             latent_dim = K,
                                                             layers = N).to(self.device)
        self.VAE = VAE_Network(input_dim = N, 
                             latent_dim = K, 
                             output_dim = N).to(self.device)
        self.VAE.load_state_dict(model_para)
        self.Encoder = self.VAE.Encoder
        self.P = P

    def forward(self, x):
        x1 = x[:, :, :self.P]
        x2 = x[:, :, self.P:]
        x2 = torch.squeeze(x2, 2)
        factor_loading = self.factor_loading_network(x1)
        mean, log_var = self.Encoder(x2)
        # z = self.VAE.reparameterization(mean, torch.exp(0.5 * log_var)) # log std -> std
        z = mean
        factor_return = torch.unsqueeze(z, axis = 2)
        return torch.bmm(factor_loading, factor_return)


class Quantile_Factor_Loading_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, N, tau_num):
        super(Quantile_Factor_Loading_Network, self).__init__()
        self.N = N
        self.FC1 = nn.Linear(input_dim, hidden_dim)
        self.FC2 = nn.Linear(hidden_dim, latent_dim * tau_num)
        self.BN = nn.BatchNorm1d(input_dim)
        self.Relu = nn.ReLU()
        '''
        the effect of weight initialization is still unknown
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.BN(x)
        x = torch.transpose(x, 2, 1)
        for i in range(self.N):
            y = torch.unsqueeze(self.Relu(self.FC1(x[:, i, :])), 1)
            y = self.FC2(y)
            if i == 0:
                y_hat = y
            else:
                y_hat = torch.cat([y_hat, y], axis = 1)
        return y_hat


class Quantile_AE_Factor_Network(nn.Module):
    def __init__(self, N, T, P, K, f_hidden_dim, tau_num, model_para):
        super(Quantile_AE_Factor_Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.factor_loading_network = Quantile_Factor_Loading_Network(input_dim = P, 
                                                                      hidden_dim = f_hidden_dim, 
                                                                      latent_dim = K, 
                                                                      N = N, 
                                                                      tau_num = tau_num).to(self.device)
        self.AE = AE_Network(input_dim = N, 
                             latent_dim = K, 
                             output_dim = N).to(self.device)
        self.AE.load_state_dict(model_para)
        self.Encoder = self.AE.Encoder
        self.P = P
        self.N = N
        self.tau_num = tau_num
        self.K = K

    def forward(self, x):
        x1 = x[:, :, :self.P]
        x2 = x[:, :, self.P:]
        x2 = torch.squeeze(x2, 2)
        factor_loading = self.factor_loading_network(x1)
        factor_return = torch.unsqueeze(self.Encoder(x2), axis = 2)
        for j in range(self.tau_num):
            if j == 0:
                y = torch.bmm(factor_loading[:, :, :self.K], factor_return)
            else:
                y = torch.cat([y, torch.bmm(factor_loading[:, :, j*self.K: (j+1) * self.K], factor_return)], axis = 2)
        return y


class Quantile_VAE_Factor_Network(nn.Module):
    def __init__(self, N, T, P, K, f_hidden_dim, tau_num, model_para):
        super(Quantile_VAE_Factor_Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.factor_loading_network = Quantile_Factor_Loading_Network(input_dim = P, 
                                                                      hidden_dim = f_hidden_dim, 
                                                                      latent_dim = K, 
                                                                      N = N, 
                                                                      tau_num = tau_num).to(self.device)
        self.AE = VAE_Network(input_dim = N, 
                             latent_dim = K, 
                             output_dim = N).to(self.device)
        self.AE.load_state_dict(model_para)
        self.Encoder = self.AE.Encoder
        self.P = P
        self.N = N
        self.tau_num = tau_num
        self.K = K

    def forward(self, x):
        x1 = x[:, :, :self.P]
        x2 = x[:, :, self.P:]
        x2 = torch.squeeze(x2, 2)
        factor_loading = self.factor_loading_network(x1)
        mean, log_std = self.Encoder(x2)
        # z = self.AE.reparameterization(mean, torch.exp(0.5 * log_std))
        z = mean
        factor_return = torch.unsqueeze(z, axis = 2)
        for j in range(self.tau_num):
            if j == 0:
                y = torch.bmm(factor_loading[:, :, :self.K], factor_return)
            else:
                y = torch.cat([y, torch.bmm(factor_loading[:, :, j*self.K: (j+1) * self.K], factor_return)], axis = 2)
        return y