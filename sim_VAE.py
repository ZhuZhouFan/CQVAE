import numpy as np
import os
from datetime import datetime
import argparse
import pandas as pd
import torch
from DGP import DGP_gu, DGP_t3
from Beta_VAE import Beta_VAE_Agent, VAE_Factor_Agent

cuda_dic = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7}

def train_once(repetition_index, bandwidth, N, T, P_f, P_x, P_c, W, linear_index, K, beta, lr, f_hidden_dim, lam, AE_epoch_num, AE_factor_epoch_num, root_log_dir, seed):
    r, C = DGP_gu(N, T, P_f, P_x, P_c, W, linear_index)
    # r, C = DGP_t3(N, T, P_f, P_x, P_c, W, linear_index)
    P = C.shape[2]
    log_dir = f'{root_log_dir}/{repetition_index}'
    portfolio = np.zeros((P, T))
    for t in range(T):
        portfolio[:, t] = np.linalg.inv(C[:, t, :].transpose() @ C[:, t, :]) @ C[:, t, :].transpose() @ r[:, t]
    VAE_agent = Beta_VAE_Agent(input_dim = P,
                    latent_dim = K,
                    output_dim = P,
                    beta = beta, 
                    learning_rate = lr,
                    seed = seed + repetition_index,
                    log_dir = log_dir)
    VAE_feature = VAE_label = torch.Tensor(portfolio.transpose())
    VAE_agent.load_data(feature = VAE_feature, label = VAE_label, valid_size = 1/3, test_size = 1/3, num_cpu = 0, batch_size = 64)
    VAE_agent.train(AE_epoch_num, 0)
    VAE_model_para = torch.load(f'{log_dir}/Beta_VAE_best.pth')
    VAE_factor = VAE_Factor_Agent(N = N,
                            T = T,
                            P = P,
                            K = K,
                            f_hidden_dim = f_hidden_dim,
                            model_para = VAE_model_para,
                            learning_rate = lr, 
                            seed = seed + repetition_index,
                            log_dir = log_dir)
    VAE_factor.load_data(C = C, r = r, valid_size = 1/3, test_size =1/3, batch_size = 64, num_cpu = 0)
    VAE_factor.train(AE_factor_epoch_num, lam)
    VAE_factor.network.load_state_dict(torch.load(f'{log_dir}/VAEF_best.pth'))
    r_hat_total = VAE_factor.network.forward(VAE_factor.feature_test.to(VAE_factor.network.device)).cpu().detach().numpy()
    true_r = VAE_factor.label_test.numpy()
    R_total = 1 - np.sum(np.power(true_r - r_hat_total, 2)) / np.sum(np.power(true_r, 2))
    feature_test = VAE_factor.feature_test
    r_test = feature_test[:, :, -1]
    c_test = feature_test[:, :, :-1]
    mean, log_std = VAE_factor.network.Encoder(r_test.to(VAE_factor.network.device))
    latent_ = mean
    # latent_ = VAE_factor.network.VAE.reparameterization(mean, torch.exp(0.5 * log_std))
    moving_latent = torch.zeros_like(latent_)
    for j in range(bandwidth, latent_.shape[0], 1):
        moving_latent[j, :] = latent_[j - bandwidth:j, :].mean(axis = 0)
    beta = VAE_factor.network.factor_loading_network(c_test.to(VAE_factor.network.device))
    r_hat_pred = torch.bmm(beta[bandwidth:, :, :], torch.unsqueeze(moving_latent[bandwidth:, :], axis = 2)).cpu().detach().numpy()
    R_pred = 1 - np.sum(np.power(true_r[bandwidth:, :, :] - r_hat_pred, 2)) / np.sum(np.power(true_r[bandwidth:, :, :], 2))
    return R_total, R_pred

def run(args):
    # cuda setting
    torch.cuda.set_device(cuda_dic[args.cuda])
    # DGP parameters
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    N = args.N
    T = args.T
    P_x = P_c = 50
    P_f = 3
    W = np.hstack([np.identity(P_f), np.zeros([P_f, P_x - P_f])])
    linear_index = args.L
    # network parameters
    f_hidden_dim = 64
    K = args.K
    lr = 1e-3
    lam = 1e-4
    beta = args.B
    AE_epoch_num = 5000
    AE_factor_epoch_num = 1000
    # the number of repetition and log_dir
    repetition_num = args.R
    bandwidth = 10
    log_name = args.log_name
    time_ = datetime.now().strftime("%Y%m%d-%H%M%S")
    root_log_dir = f'{log_name}_{seed}_{time_}'
    if not os.path.exists(root_log_dir):
        os.makedirs(root_log_dir)
    # loop
    result_tab = pd.DataFrame(columns = ['R_total', 'R_pred'])
    for repetition_index in range(repetition_num):
        R_total, R_pred = train_once(repetition_index, bandwidth, N, T, P_f, P_x, P_c, W, linear_index, K, beta, lr, f_hidden_dim, lam, AE_epoch_num, AE_factor_epoch_num, root_log_dir, seed)
        result_tab.loc[repetition_index, 'R_total'] = R_total
        result_tab.loc[repetition_index, 'R_pred'] = R_pred
        result_tab.to_csv(f'{root_log_dir}/result.csv', index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type = str, default = 'one')
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--N', type = int, default = 200)
    parser.add_argument('--T', type = int, default = 500)
    parser.add_argument('--L', type = bool, default = True)
    parser.add_argument('--R', type = int, default = 100)
    parser.add_argument('--K', type = int, default = 3)
    parser.add_argument('--B', type = float, default = 1.0)
    parser.add_argument('--log_name', type = str, default = 'AE_test')
    args = parser.parse_args()
    run(args)