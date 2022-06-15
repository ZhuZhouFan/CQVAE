import torch
import torch.nn as nn


class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE_Encoder, self).__init__()
        self.FC_mean = nn.Linear(input_dim, latent_dim)
        self.FC_var = nn.Linear(input_dim, latent_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        mean = self.FC_mean(x)
        log_var = self.FC_var(x)
        return mean, log_var


class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(VAE_Decoder, self).__init__()
        self.FC = nn.Linear(latent_dim, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.FC(x)
        return x


class VAE_Network(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(VAE_Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Encoder = VAE_Encoder(input_dim, latent_dim).to(self.device)
        self.Decoder = VAE_Decoder(latent_dim, output_dim).to(self.device)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device) #抽样噪音
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # log std -> std
        x_hat = self.Decoder(z)
        return x_hat, mean, log_var