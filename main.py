'''
This file is used to reproduce the simulation results for CAE, CVAE, CQAE and CQVAE models in the Appendix.

The workflow of this file could be abstracted as following:
1. generate the [N, T, P] feature of firm characteristics and [N, T] excess return;
2. Divide the full dataset into training set, validation set and testing set;
3. Train the model with training dataset and validation dataset;
4. Evaluate the model performance on the testing set.

As for the empirical analysis, just replace the generated data in step 1 with the designed feature and excess return based on your dataset.

We recommend to run our code with GPU, since it could be much faster compared to running on CPU.
'''

import numpy as np
import os
from datetime import datetime
import argparse
import pandas as pd
import torch
from DGP import DGP_gu, DGP_t3
from AE import AE_Agent, AE_Factor_Agent, QAE_Factor_Agent
from Beta_VAE import Beta_VAE_Agent, VAE_Factor_Agent, QVAE_Factor_Agent


def simulate(repetition_index, root_log_dir, seed, 
               N, T, P_f, P_x, P_c, W, linear_index, heavy_tail_index, 
               model_type, f_hidden_dim, K, lr, lam, AE_epoch_num, AE_factor_epoch_num,
               bandwidth):
    
    # generate dataset.
    # For empirical analysis, just feed in the designed feature and label here
    if heavy_tail_index:
        r, C = DGP_t3(N, T, P_f, P_x, P_c, W, linear_index)
    else:
        r, C = DGP_gu(N, T, P_f, P_x, P_c, W, linear_index)
    P = C.shape[2]
        
    # build the (variational) autoencoder model and train it
    log_dir = f'{root_log_dir}/{repetition_index}'
    
    if model_type in ['CAE', 'CQAE']:
        AE_agent = AE_Agent(input_dim = P,
                    latent_dim = K,
                    output_dim = P,
                    learning_rate = lr,
                    seed = seed + repetition_index,
                    log_dir = log_dir)
    elif model_type in ['CVAE', 'CQVAE']:
        AE_agent = Beta_VAE_Agent(input_dim = P,
                    latent_dim = K,
                    output_dim = P,
                    beta = 1.0, 
                    learning_rate = lr,
                    seed = seed + repetition_index,
                    log_dir = log_dir)
    else:
        raise ValueError('Model must be choosen from CAE, CVAE, CQAE and CQVAE')
    
    # compute the managed portfolio return 
    portfolio = np.zeros((P, T))
    for t in range(T):
        portfolio[:, t] = np.linalg.inv(C[:, t, :].transpose() @ C[:, t, :]) @ C[:, t, :].transpose() @ r[:, t]
    AE_feature = AE_label = torch.Tensor(portfolio.transpose())
    
    AE_agent.load_data(feature = AE_feature, label = AE_label, valid_size = 1/3, test_size = 1/3, num_cpu = 0, batch_size = 64)
    AE_agent.train(AE_epoch_num, 0)
    
    # after training the autoencoder, build the beta side and train it
    if model_type in ['CAE', 'CQAE']:
        AE_model_para = torch.load(f'{log_dir}/AE_best.pth')
    elif model_type in ['CVAE', 'CQVAE']:
        AE_model_para = torch.load(f'{log_dir}/Beta_VAE_best.pth')
    else:
        raise ValueError('Model must be choosen from CAE, CVAE, CQAE and CQVAE')
    
    if model_type == 'CAE':
        AE_factor = AE_Factor_Agent(N = N,
                            T = T,
                            P = P,
                            K = K,
                            f_hidden_dim = f_hidden_dim,
                            model_para = AE_model_para,
                            learning_rate = lr, 
                            seed = seed + repetition_index,
                            log_dir = log_dir)
    elif model_type == 'CVAE':
        AE_factor = VAE_Factor_Agent(N = N,
                            T = T,
                            P = P,
                            K = K,
                            f_hidden_dim = f_hidden_dim,
                            model_para = AE_model_para,
                            learning_rate = lr, 
                            seed = seed + repetition_index,
                            log_dir = log_dir)
    elif model_type == 'CQAE':
        # tau is the vector of predetermined quantile levels used for approximating the distribution.
        # we recommend to use the following one.
        tau = torch.Tensor([0.1, 0.3, 0.5, 0.7, 0.9]).unsqueeze(0).unsqueeze(1) 
        AE_factor = QAE_Factor_Agent(N = N,
                            T = T,
                            P = P,
                            K = K,
                            f_hidden_dim = f_hidden_dim,
                            tau = tau,  
                            model_para = AE_model_para,
                            learning_rate = lr, 
                            seed = seed + repetition_index,
                            log_dir = log_dir)
    elif model_type == 'CQVAE':
        tau = torch.Tensor([0.1, 0.3, 0.5, 0.7, 0.9]).unsqueeze(0).unsqueeze(1) 
        AE_factor = QVAE_Factor_Agent(N = N,
                            T = T,
                            P = P,
                            K = K,
                            f_hidden_dim = f_hidden_dim,
                            tau = tau,  
                            model_para = AE_model_para,
                            learning_rate = lr, 
                            seed = seed + repetition_index,
                            log_dir = log_dir)
    else:
        raise ValueError('Model must be choosen from CAE, CVAE, CQAE and CQVAE')
    
    AE_factor.load_data(C = C, r = r, valid_size = 1/3, test_size =1/3, batch_size = 64, num_cpu = 0)
    AE_factor.train(AE_factor_epoch_num, lam)
    
    # evaluate the model performance
    if model_type == 'CAE':
        AE_factor.network.load_state_dict(torch.load(f'{log_dir}/AEF_best.pth'))
    elif model_type == 'CVAE':
        AE_factor.network.load_state_dict(torch.load(f'{log_dir}/VAEF_best.pth'))
    elif model_type == 'CQAE':
        AE_factor.network.load_state_dict(torch.load(f'{log_dir}/QAEF_best.pth'))
    elif model_type == 'CQVAE':
        AE_factor.network.load_state_dict(torch.load(f'{log_dir}/QVAEF_best.pth'))
    
    # Compute the R_total on testing set
    true_r = AE_factor.label_test.numpy()
    r_hat_total = AE_factor.network.forward(AE_factor.feature_test.to(AE_factor.network.device)).cpu().detach().numpy()
    if model_type in ['CQAE', 'CQVAE']:
        true_r = AE_factor.label_test.numpy()[:, :, 0]
        r_hat_total = r_hat_total.mean(axis = 2) # since CQAE and CQVAE estimate the J conditional quantiles
    R_total = 1 - np.sum(np.power(true_r - r_hat_total, 2)) / np.sum(np.power(true_r, 2))
    
    feature_test = AE_factor.feature_test
    r_test = feature_test[:, :, -1]
    c_test = feature_test[:, :, :-1]
    
    portfolio_test = torch.zeros((r_test.shape[0], P), device = AE_factor.network.device)
    for t in range(r_test.shape[0]):
        portfolio_test[t, :] = torch.inverse(c_test[t, :, :].t() @ c_test[t, :, :]) @ c_test[t, :, :].t() @ r_test[t, :]
        
    if model_type in ['CAE', 'CQAE']:
        latent_ = AE_factor.network.Encoder(portfolio_test.to(AE_factor.network.device))
    else:
        latent_, _ = AE_factor.network.Encoder(portfolio_test.to(AE_factor.network.device))
            
    moving_latent = torch.zeros_like(latent_)
    for j in range(bandwidth, latent_.shape[0], 1):
        moving_latent[j, :] = latent_[j - bandwidth:j, :].mean(axis = 0)
    beta = AE_factor.network.factor_loading_network(c_test.to(AE_factor.network.device))
    
    if model_type in ['CQAE', 'CQVAE']:
        factor_return = torch.unsqueeze(moving_latent[bandwidth:, :], axis = 2)
        for j in range(AE_factor.network.tau_num):
            if j == 0:
                r_hat_pred = torch.bmm(beta[bandwidth:, :, :K], factor_return)
            else:
                r_hat_pred = torch.cat([r_hat_pred, torch.bmm(beta[bandwidth:, :, j*K: (j+1) * K], factor_return)], axis = 2) 
        r_hat_pred = r_hat_pred.cpu().detach().numpy().mean(axis = 2)
        R_pred = 1 - np.sum(np.power(true_r[bandwidth:, :] - r_hat_pred, 2)) / np.sum(np.power(true_r[bandwidth:, :], 2))  
    else:
        r_hat_pred = torch.bmm(beta[bandwidth:, :, :], torch.unsqueeze(moving_latent[bandwidth:, :], axis = 2)).cpu().detach().numpy()
        R_pred = 1 - np.sum(np.power(true_r[bandwidth:, :, :] - r_hat_pred, 2)) / np.sum(np.power(true_r[bandwidth:, :, :], 2))  
          
    return R_total, R_pred

def run(args):
    # cuda setting
    torch.cuda.set_device(args.cuda)
    # DGP parameters
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    N = args.N
    T = args.T
    P_x = P_c = 50
    P_f = 3
    W = np.hstack([np.identity(P_f), np.zeros([P_f, P_x - P_f])])
    linear_index = args.linear
    heavy_tail_index = args.heavy_tail
    # network parameters
    model_type = args.model
    f_hidden_dim = args.hidden
    K = args.K
    lr = args.lr
    lam = args.lam
    AE_epoch_num = 5000
    AE_factor_epoch_num = 1000
    # the number of repetition and log_dir
    repetition_num = args.R
    bandwidth = args.bandwidth
    log_name = args.save_folder
    time_ = datetime.now().strftime("%Y%m%d-%H%M%S")
    root_log_dir = f'{log_name}_{seed}_{time_}'
    if not os.path.exists(root_log_dir):
        os.makedirs(root_log_dir)
    # loop
    result_tab = pd.DataFrame(columns = ['R_total', 'R_pred'])
    for repetition_index in range(repetition_num):
        R_total, R_pred = simulate(repetition_index, root_log_dir, seed, 
               N, T, P_f, P_x, P_c, W, linear_index, heavy_tail_index, 
               model_type, f_hidden_dim, K, lr, lam, AE_epoch_num, AE_factor_epoch_num,
               bandwidth)
        result_tab.loc[repetition_index, 'R_total'] = R_total
        result_tab.loc[repetition_index, 'R_pred'] = R_pred
        result_tab.to_csv(f'{root_log_dir}/result.csv', index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parameters for simulation setting
    parser.add_argument('--R', type=int, default = 3,
                        help='Number of repetitions in simulation')
    parser.add_argument('--seed', type=int, default = 42, 
                        help = 'Random seed.')
    # parameters for data generating process
    parser.add_argument('--N', type=int, default = 1000,
                        help = 'Number of individuals')
    parser.add_argument('--T', type=int, default = 400,
                        help = 'Number of time observations')
    parser.add_argument('--linear', action='store_true', default=False,
                    help='Whether use linear beta function in DGP')
    parser.add_argument('--heavy-tail', action='store_true', default=False,
                    help='Whether use t_3 distribution for latent factor')
    # parameters for network
    parser.add_argument('--cuda', type=int, default = 0, 
                        help='Number of GPU device.')
    parser.add_argument('--model', type=str, default = 'CAE', 
                        help='Which model used for prediction. Must be choosed from CAE, CVAE, CQAE and CQVAE')
    parser.add_argument('--K', type=int, default = 3, 
                        help='Number of estimated latent factor')
    parser.add_argument('--hidden', type=int, default = 64, 
                        help='Number of hidden units in FC layer')
    parser.add_argument('--lr', type=float, default = 1e-2, 
                        help='Learning rate for ADAM algorithm')
    parser.add_argument('--lam', type=float, default = 0, 
                        help='Tuning parameter for L_1 penalty')
    parser.add_argument('--bandwidth', type = int, default = 10, 
                    help='Bandwidth for predicting the latent factor')
    parser.add_argument('--save-folder', type=str, default = 'logs',
                        help='Where to save the trained model and evluation results')
    args = parser.parse_args()
    run(args)