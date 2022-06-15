import os
import torch
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from Network import Quantile_AE_Factor_Network


def load_mat(factor_name, period, load_path):
    try:
        df = pd.read_csv(f'{load_path}/{factor_name}.csv', index_col = 'date')
        df = df.loc[period[0]:period[1], :]
        return df
    except Exception as e:
        return None


def rank_normalize_C(C):
    for j in range(C.shape[1]):
        tem = C[:, j, :]
        tem_ = pd.DataFrame(tem)
        C[:, j, :] = (2 * tem_.rank()/(tem_.shape[0] + 1) - 1).values
    return C


class QAE_Factor_Inference(object):
    def __init__(self, factor_matrix_path, model_path, K, f_hidden_dim, bandwidth, factor_list, 
                 tau = torch.Tensor([0.1, 0.3, 0.5, 0.7, 0.9]).unsqueeze(0).unsqueeze(1)):
        self.factor_matrix_path = factor_matrix_path
        self.model_path = model_path
        self.K = K
        self.f_hidden_dim = f_hidden_dim
        self.bandwidth = bandwidth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.factor_list = factor_list
        self.tau = tau

    def load_data(self, log_period, permno_list, set_zero = None):
        data_list = Parallel(n_jobs = 10)(delayed(load_mat)(factor_name, log_period, self.factor_matrix_path) for factor_name in self.factor_list)
        self.T = data_list[0].shape[0]
        self.N = len(permno_list)
        self.P = len(data_list)
        trade_index = data_list[0].index
        r = load_mat('RET', log_period, self.factor_matrix_path)[permno_list].values.transpose()
        C = np.zeros((self.N, self.T, self.P))
        for i in range(self.P):
            C[:, :, i] = data_list[i][permno_list].values.transpose()
        C = rank_normalize_C(C)
        if set_zero is not None:
            set_zero_index = self.factor_list.index(set_zero)
            C[:, :, set_zero_index] = 0.0
        C = C.swapaxes(0, 1)
        r = r.transpose()
        r = r[:, :, np.newaxis]
        C = np.concatenate((C, r), axis = 2)
        self.permno_list = permno_list
        self.trade_index = trade_index
        return C, r

    def inference(self, log_period, permno_list, feature = None, set_zero = None):
        if feature is None:
            feature, label = self.load_data(log_period, permno_list, set_zero)
        feature = torch.Tensor(feature)
        period_model_path = f'{self.model_path}/{log_period[0]}-{log_period[1]}'
        model_log_index = os.listdir(period_model_path)
        model_log_index.sort()

        networks = dict.fromkeys(range(len(model_log_index)), 0)
        y_totals = dict.fromkeys(networks.keys(), 0)
        y_preds = dict.fromkeys(networks.keys(), 0)

        with torch.no_grad():
            for j in networks.keys():
                path = os.path.join(period_model_path, model_log_index[j])
                AE_model_para = torch.load(f'{path}/AE_best.pth', map_location = self.device)
                networks[j] = Quantile_AE_Factor_Network(self.N, self.T, self.P, self.K, self.f_hidden_dim, self.tau.shape[-1], AE_model_para, )
                networks[j].load_state_dict(torch.load(f'{path}/QAEF_best.pth', map_location = self.device))
                networks[j].eval()
                y_totals[j] = pd.DataFrame(data = networks[j].forward(feature.to(self.device)).cpu().numpy().mean(axis = 2),
                                           index = self.trade_index,
                                           columns = self.permno_list)
                r_ = feature[:, :, -1]
                c_ = feature[:, :, :-1]
                latent_variable = networks[j].Encoder(r_.to(self.device))
                ma_latent = torch.zeros_like(latent_variable)
                factor_return = ma_latent.unsqueeze(axis = 2)
                for k in range(self.bandwidth, latent_variable.shape[0], 1):
                    ma_latent[k, :] = latent_variable[(k - self.bandwidth):k, :].mean(axis = 0)
                beta = networks[j].factor_loading_network(c_.to(self.device))
                for l in range(self.tau.shape[-1]):
                    if l == 0:
                        y_pred = torch.bmm(beta[:, :, :self.K], factor_return)
                    else:
                        y_pred = torch.cat([y_pred, torch.bmm(beta[:, :, l * self.K: (l + 1) * self.K], factor_return)], axis = 2)
                y_preds[j] = pd.DataFrame(data = y_pred.cpu().numpy().mean(axis = 2),
                                          index = self.trade_index,
                                          columns = self.permno_list)
                y_preds[j].iloc[:self.bandwidth, :] = np.nan
        return y_totals, y_preds