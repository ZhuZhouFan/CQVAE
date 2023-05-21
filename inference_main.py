import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
import torch
from AE import AE_Factor_Agent, QAE_Factor_Agent
from Beta_VAE import VAE_Factor_Agent, QVAE_Factor_Agent

def load_mat(factor_name, start_time, end_time, load_path):
    try:
        factor_matrix = pd.read_csv(f'{load_path}/{factor_name}.csv', index_col = 'date')
        factor_matrix = factor_matrix.loc[start_time:end_time, :]
        return factor_matrix
    except Exception as e:
        return None
    
def build_feature_and_label(start_time,
                             end_time,
                             permno_list,
                             matrix_path):
    factor_list = ['absacc','acc','aeavol' ,'age' ,'agr' ,'baspread' ,'beta' ,'betasq',
               'bm','bm_ia' ,'cash','cashdebt','cashpr','cfp','cfp_ia' ,'chatoia','chcsho' ,'chempia',
               'chinv','chmom' ,'chpmia' ,'chtx','cinvest','convind' ,'currat' ,'depr' ,'divi' ,
               'divo','dolvol' ,'dy','ear','egr','ep','gma','grcapx','grltnoa','herf','hire',
               'idiovol','ill','indmom','invest','lev','lgr','maxret','mom12m','mom1m','mom36m',
               'mom6m','ms','mvel1','mve_ia','nincr','operprof','orgcap','pchcapx_ia','pchcurrat',
               'pchdepr','pchgm_pchsale','pchquick','pchsale_pchinvt','pchsale_pchrect', 'pchsale_pchxsga', 
               'pchsaleinv', 'pctacc', 'pricedelay', 'ps', 'quick', 'rd', 'rd_mve', 'rd_sale', 
               'realestate', 'retvol', 'roaq', 'roavol', 'roeq', 'roic', 'rsup', 'salecash', 
               'saleinv', 'salerec', 'secured', 'securedind', 'sgr', 'sin', 'sp', 'std_dolvol', 
               'std_turn', 'stdacc', 'stdcf', 'tang', 'tb', 'turn', 'zerotrade']
    label_matrix = load_mat('RET', start_time, end_time, matrix_path)[permno_list]
    selected_date_list = label_matrix.index.values
    N, T = label_matrix.T.shape
    P = len(factor_list)
    
    C = np.zeros([N, T, P])
    factor_matrix_list = Parallel(n_jobs = 20)(delayed(load_mat)
                                  (factor, start_time, end_time, matrix_path) for factor in tqdm(factor_list, desc = 'Loading factor matrices'))
    for p, factor_matrix in enumerate(factor_matrix_list):
        factor_matrix = factor_matrix[permno_list]
        factor_matrix = 2 * factor_matrix.rank(axis = 1) / (N + 1) - 1
        C[:, :, p] = factor_matrix.T.values
        
    r = np.zeros([N, T])
    r[:, :] = label_matrix.T.values
        
    return C, r, selected_date_list

def inference(args):
    
    train_start_time = args.train_start_time
    train_end_time = args.train_end_time
    test_end_time = args.test_end_time
    matrix_path = args.matrix_path
    model_log_path = args.model_path
    seed = args.seed
    model_type = args.model
    f_hidden_dim = args.hidden
    lam = args.lam
    lr = args.lr
    K = args.K
    bandwidth = args.bandwidth
    
    trained_model_path = f'{model_log_path}/{model_type}/{seed}_{train_start_time}_{train_end_time}_{lam}_{lr}_{f_hidden_dim}'
    permno_list = np.load(f'{trained_model_path}/permno_list.npy', allow_pickle = True)
    C, r, selected_date_list = build_feature_and_label(train_start_time,
                             test_end_time,
                             permno_list,
                             matrix_path)
    
    N, T, P = C.shape
    
    portfolio = np.zeros((P, T))
    for t in range(T):
        try:
            portfolio[:, t] = np.linalg.inv(C[:, t, :].transpose() @ C[:, t, :]) @ C[:, t, :].transpose() @ r[:, t]
        except Exception:
            portfolio[:, t] = np.linalg.pinv(C[:, t, :].transpose() @ C[:, t, :]) @ C[:, t, :].transpose() @ r[:, t]
            
    if model_type in ['CAE', 'CQAE']:
        AE_model_para = torch.load(f'{trained_model_path}/AE_best.pth')
    elif model_type in ['CVAE', 'CQVAE']:
        AE_model_para = torch.load(f'{trained_model_path}/Beta_VAE_best.pth')
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
                        seed = seed,
                        log_dir = trained_model_path)
        AE_factor.network.load_state_dict(torch.load(f'{trained_model_path}/AEF_best.pth'))
    elif model_type == 'CVAE':
        AE_factor = VAE_Factor_Agent(N = N,
                            T = T,
                            P = P,
                            K = K,
                            f_hidden_dim = f_hidden_dim,
                            model_para = AE_model_para,
                            learning_rate = lr, 
                            seed = seed,
                            log_dir = trained_model_path)
        AE_factor.network.load_state_dict(torch.load(f'{trained_model_path}/VAEF_best.pth'))
    elif model_type == 'CQAE':
        tau = torch.Tensor([0.1, 0.3, 0.5, 0.7, 0.9]).unsqueeze(0).unsqueeze(1) 
        AE_factor = QAE_Factor_Agent(N = N,
                            T = T,
                            P = P,
                            K = K,
                            f_hidden_dim = f_hidden_dim,
                            tau = tau,  
                            model_para = AE_model_para,
                            learning_rate = lr, 
                            seed = seed,
                            log_dir = trained_model_path)
        AE_factor.network.load_state_dict(torch.load(f'{trained_model_path}/QAEF_best.pth'))
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
                            seed = seed,
                            log_dir = trained_model_path)
        AE_factor.network.load_state_dict(torch.load(f'{trained_model_path}/QVAEF_best.pth'))
    else:
        raise ValueError('Model must be choosen from CAE, CVAE, CQAE and CQVAE')
    
    if model_type in ['CAE', 'CQAE']:
        latent_ = AE_factor.network.Encoder(torch.Tensor(portfolio.T).to(AE_factor.network.device))
    else:
        latent_, _ = AE_factor.network.Encoder(torch.Tensor(portfolio.T).to(AE_factor.network.device))
    moving_latent = torch.zeros_like(latent_)
    for j in range(bandwidth, latent_.shape[0], 1):
        moving_latent[j, :] = latent_[j - bandwidth:j, :].mean(axis = 0)
    
    beta = AE_factor.network.factor_loading_network(torch.Tensor(C).transpose(0, 1).to(AE_factor.network.device))
    
    if model_type in ['CAE', 'CVAE']:
        r_hat_total = torch.bmm(beta, torch.unsqueeze(latent_, axis = 2)).cpu().detach().numpy()
        r_hat_pred = torch.bmm(beta, torch.unsqueeze(moving_latent, axis = 2)).cpu().detach().numpy()

        mu_total_df = pd.DataFrame(columns=permno_list, index = selected_date_list)
        mu_total_df.index.name = 'date'
        mu_total_df.loc[:, :] = r_hat_total.squeeze(-1)
        mu_total_df = mu_total_df.loc[train_end_time:test_end_time, :]
        mu_total_df.to_csv(f'{trained_model_path}/mu_total.csv')

        mu_pred_df = pd.DataFrame(columns=permno_list, index = selected_date_list)
        mu_pred_df.index.name = 'date'
        mu_pred_df.loc[:, :] = r_hat_pred.squeeze(-1)
        mu_pred_df = mu_pred_df.loc[train_end_time:test_end_time, :]
        mu_pred_df.to_csv(f'{trained_model_path}/mu_pred.csv')
    else:
        tau_list = [0.1, 0.3, 0.5, 0.7, 0.9]
        latent_ = torch.unsqueeze(latent_, axis = 2)
        moving_latent = torch.unsqueeze(moving_latent, axis = 2)
        for j, tau in enumerate(tau_list):
            quantile_total = torch.bmm(beta[:, :, j*K: (j+1) * K], latent_).cpu().detach().numpy()
            quantile_pred = torch.bmm(beta[:, :, j*K: (j+1) * K], moving_latent).cpu().detach().numpy()

            qunatile_total_df = pd.DataFrame(columns=permno_list, index = selected_date_list)
            qunatile_total_df.index.name = 'date'
            qunatile_total_df.loc[:, :] = quantile_total.squeeze(-1)
            qunatile_total_df = qunatile_total_df.loc[train_end_time:test_end_time, :]
            qunatile_total_df.to_csv(f'{trained_model_path}/{tau}_total.csv')

            qunatile_pred_df = pd.DataFrame(columns=permno_list, index = selected_date_list)
            qunatile_pred_df.index.name = 'date'
            qunatile_pred_df.loc[:, :] = quantile_pred.squeeze(-1)
            qunatile_pred_df = qunatile_pred_df.loc[train_end_time:test_end_time, :]
            qunatile_pred_df.to_csv(f'{trained_model_path}/{tau}_total.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # specify the network
    parser.add_argument('--model', type=str, default = 'CAE', 
                        help='Which model used for prediction. Must be choosed from CAE, CVAE, CQAE and CQVAE')
    parser.add_argument('--seed', type=int, default = 42, 
                        help='Random Seed')
    parser.add_argument('--K', type=int, default = 3, 
                        help='Number of estimated latent factor')
    parser.add_argument('--hidden', type=int, default = 128, 
                        help='Number of hidden units in FC layer')
    parser.add_argument('--lr', type=float, default = 1e-3, 
                        help='Learning rate for ADAM algorithm')
    parser.add_argument('--lam', type=float, default = 0, 
                        help='Tuning parameter for L_1 penalty')
    parser.add_argument('--bandwidth', type = int, default = 10, 
                    help='Bandwidth for predicting the latent factor')
    # specify the in-sample period and out-of-sample period
    parser.add_argument('--train-start-time', type=str, default = '1957-01-01', 
                        help='The starting time of insample period')
    parser.add_argument('--train-end-time', type=str, default = '1987-01-01', 
                        help='The ending time of insample period')
    parser.add_argument('--test-end-time', type=str, default = '1992-01-01', 
                        help='The ending time of insample period')
    # specify the saving directory
    parser.add_argument('--matrix-path', type=str, default = 'Using the fillna_matrix_path in the preprocess_data.py',
                        help='The path of preprocessed factor matrices')
    parser.add_argument('--model-path', type=str, default = 'Using the save_dir in the empirical_main.py',
                        help='The path of trained model')
    args = parser.parse_args()
    inference(args)
    