import numpy as np
import os
from datetime import datetime
import argparse
import pandas as pd
import torch
from DGP import DGP_gu, DGP_t3
from AE import AE_Agent, AE_Factor_Agent

cuda_dic = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7}

def train_once(repetition_index, bandwidth, N, T, P_f, P_x, P_c, W, linear_index, K, lr, f_hidden_dim, lam, AE_epoch_num, AE_factor_epoch_num, root_log_dir, seed):
    # r, C = DGP_gu(N, T, P_f, P_x, P_c, W, linear_index)
    r, C = DGP_t3(N, T, P_f, P_x, P_c, W, linear_index)
    P = C.shape[2]
    log_dir = f'{root_log_dir}/{repetition_index}'
    portfolio = np.zeros((P, T))
    for t in range(T):
        portfolio[:, t] = np.linalg.inv(C[:, t, :].transpose() @ C[:, t, :]) @ C[:, t, :].transpose() @ r[:, t]
    AE_agent = AE_Agent(input_dim = P,
                    latent_dim = K,
                    output_dim = P,
                    learning_rate = lr,
                    seed = seed + repetition_index,
                    log_dir = log_dir)
    AE_feature = AE_label = torch.Tensor(portfolio.transpose())
    AE_agent.load_data(feature = AE_feature, label = AE_label, valid_size = 1/3, test_size = 1/3, num_cpu = 0, batch_size = 64)
    AE_agent.train(AE_epoch_num, 0)
    AE_model_para = torch.load(f'{log_dir}/AE_best.pth')
    AE_factor = AE_Factor_Agent(N = N,
                            T = T,
                            P = P,
                            K = K,
                            f_hidden_dim = f_hidden_dim,
                            model_para = AE_model_para,
                            learning_rate = lr, 
                            seed = seed + repetition_index,
                            log_dir = log_dir)
    AE_factor.load_data(C = C, r = r, valid_size = 1/3, test_size =1/3, batch_size = 64, num_cpu = 0)
    AE_factor.train(AE_factor_epoch_num, lam)
    AE_factor.network.load_state_dict(torch.load(f'{log_dir}/AEF_best.pth'))
    true_r = AE_factor.label_test.numpy()
    r_hat_total = AE_factor.network.forward(AE_factor.feature_test.to(AE_factor.network.device)).cpu().detach().numpy()
    R_total = 1 - np.sum(np.power(true_r - r_hat_total, 2)) / np.sum(np.power(true_r, 2))
    feature_test = AE_factor.feature_test
    r_test = feature_test[:, :, -1]
    c_test = feature_test[:, :, :-1]
    portfolio_test = torch.zeros((r_test.shape[0], P), device = AE_factor.network.device)
    for t in range(r_test.shape[0]):
        portfolio_test[t, :] = torch.inverse(c_test[t, :, :].t() @ c_test[t, :, :]) @ c_test[t, :, :].t() @ r_test[t, :]
    latent_ = AE_factor.network.Encoder(portfolio_test.to(AE_factor.network.device))
    moving_latent = torch.zeros_like(latent_)
    for j in range(bandwidth, latent_.shape[0], 1):
        moving_latent[j, :] = latent_[(j - bandwidth):j, :].mean(axis = 0)
    beta = AE_factor.network.factor_loading_network(c_test.to(AE_factor.network.device))
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
    AE_epoch_num = 5000
    AE_factor_epoch_num = 5000
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
        R_total, R_pred = train_once(repetition_index, bandwidth, N, T, P_f, P_x, P_c, W, linear_index, K, lr, f_hidden_dim, lam, AE_epoch_num, AE_factor_epoch_num, root_log_dir, seed)
        result_tab.loc[repetition_index, 'R_total'] = R_total
        result_tab.loc[repetition_index, 'R_pred'] = R_pred
        result_tab.to_csv(f'{root_log_dir}/result.csv', index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default = 'one')
    parser.add_argument('--seed', type=int, default = 0)
    parser.add_argument('--N', type=int, default = 200)
    parser.add_argument('--T', type=int, default = 500)
    parser.add_argument('--L', type=bool, default = True)
    parser.add_argument('--R', type=int, default = 100)
    parser.add_argument('--K', type=int, default = 3)
    parser.add_argument('--log_name', type=str, default = 'AE_test')
    args = parser.parse_args()
    run(args)