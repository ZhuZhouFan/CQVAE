import torch
import torch.nn as nn


class AE_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AE_Encoder, self).__init__()
        self.FC = nn.Linear(input_dim, latent_dim)
        self.Relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.Relu(self.FC(x))
        return x


class AE_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(AE_Decoder, self).__init__()
        self.FC = nn.Linear(latent_dim, output_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.FC(x)
        return x


class AE_Network(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(AE_Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Encoder = AE_Encoder(input_dim, latent_dim).to(self.device)
        self.Decoder = AE_Decoder(latent_dim, output_dim).to(self.device)

    def forward(self, x):
        latent = self.Encoder(x)
        y = self.Decoder(latent)
        return y