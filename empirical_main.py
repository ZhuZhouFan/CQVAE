import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from datetime import datetime
import argparse
import torch
from AE import AE_Agent, AE_Factor_Agent, QAE_Factor_Agent
from Beta_VAE import Beta_VAE_Agent, VAE_Factor_Agent, QVAE_Factor_Agent

def load_mat(factor_name,
             start_time,
             end_time,
             load_path):
    try:
        factor_matrix = pd.read_csv(f'{load_path}/{factor_name}.csv', index_col = 'date')
        factor_matrix = factor_matrix.loc[start_time:end_time, :]
        return factor_matrix
    except Exception as e:
        return None

def obtain_permno_list(factor_name, start_time, end_time, load_path, na_percent = 0.75):
    df = load_mat(factor_name, start_time, end_time, load_path)
    indice = (df.isna().sum(axis = 0) < df.shape[0] * na_percent)
    permno_list = indice.loc[indice].index.values
    return permno_list

def build_feature_and_label(start_time,
                            end_time,
                            matrix_path,
                            na_matrix_path):
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
    permno_list = obtain_permno_list('RET', start_time, end_time, na_matrix_path)
    label_matrix = load_mat('RET', start_time, end_time, matrix_path)[permno_list]
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
    
    return C, r, permno_list

def run(args):
    # cuda setting
    torch.cuda.set_device(args.cuda)
    # DGP parameters
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # network parameters
    model_type = args.model
    f_hidden_dim = args.hidden
    K = args.K
    lr = args.lr
    lam = args.lam
    AE_epoch_num = 5000
    AE_factor_epoch_num = 1000
    
    # specify other parameters
    start_time = args.start_time
    end_time = args.end_time
    log_name = args.save_folder
    matrix_path = args.matrix
    na_matrix_path = args.namatrix
    log_dir = f'{log_name}/{model_type}/{seed}_{start_time}_{end_time}_{K}_{lam}_{lr}_{f_hidden_dim}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    C, r, permno_list = build_feature_and_label(start_time, end_time, matrix_path, na_matrix_path)
    np.save(f'{log_dir}/permno_list.npy', permno_list)
    
    N, T, P = C.shape
    
    if model_type in ['CAE', 'CQAE']:
        AE_agent = AE_Agent(input_dim = P,
                    latent_dim = K,
                    output_dim = P,
                    learning_rate = lr,
                    seed = seed,
                    log_dir = log_dir)
    elif model_type in ['CVAE', 'CQVAE']:
        AE_agent = Beta_VAE_Agent(input_dim = P,
                    latent_dim = K,
                    output_dim = P,
                    beta = 1.0, 
                    learning_rate = lr,
                    seed = seed,
                    log_dir = log_dir)
    else:
        raise ValueError('Model must be choosen from CAE, CVAE, CQAE and CQVAE')
    
    # compute the managed portfolio return 
    portfolio = np.zeros((P, T))
    for t in range(T):
        try:
            portfolio[:, t] = np.linalg.inv(C[:, t, :].transpose() @ C[:, t, :]) @ C[:, t, :].transpose() @ r[:, t]
        except Exception:
            portfolio[:, t] = np.linalg.pinv(C[:, t, :].transpose() @ C[:, t, :]) @ C[:, t, :].transpose() @ r[:, t]
    
    AE_feature = AE_label = torch.Tensor(portfolio.transpose())
    AE_agent.load_data(feature = AE_feature, label = AE_label, valid_size = 1/3, test_size = 0, num_cpu = 0, batch_size = 64)
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
                            seed = seed,
                            log_dir = log_dir)
    elif model_type == 'CVAE':
        AE_factor = VAE_Factor_Agent(N = N,
                            T = T,
                            P = P,
                            K = K,
                            f_hidden_dim = f_hidden_dim,
                            model_para = AE_model_para,
                            learning_rate = lr, 
                            seed = seed,
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
                            seed = seed,
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
                            seed = seed,
                            log_dir = log_dir)
    else:
        raise ValueError('Model must be choosen from CAE, CVAE, CQAE and CQVAE')
    
    AE_factor.load_data(C = C, r = r, valid_size = 1/3, test_size = 0, batch_size = 64, num_cpu = 0)
    AE_factor.train(AE_factor_epoch_num, lam)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parameters for simulation setting
    parser.add_argument('--seed', type=int, default = 42, 
                        help = 'Random seed.')
    # parameters for specify training sample and validation sample
    parser.add_argument('--start-time', type=str, default = '1957-01-01', 
                        help = 'The starting time of insample period.')
    parser.add_argument('--end-time', type=str, default = '1987-01-01', 
                        help = 'The ending time of insample period.')
    # parameters for network
    parser.add_argument('--cuda', type=int, default = 0, 
                        help='Number of GPU device.')
    parser.add_argument('--model', type=str, default = 'CAE', 
                        help='Which model used for prediction. Must be choosed from CAE, CVAE, CQAE and CQVAE')
    parser.add_argument('--K', type=int, default = 3, 
                        help='Number of estimated latent factor')
    parser.add_argument('--hidden', type=int, default = 128, 
                        help='Number of hidden units in FC layer')
    parser.add_argument('--lr', type=float, default = 1e-3, 
                        help='Learning rate for ADAM algorithm')
    parser.add_argument('--lam', type=float, default = 1e-3, 
                        help='Tuning parameter for L_1 penalty')
    # parameters for specifying data storage
    parser.add_argument('--save-folder', type=str, default = 'Specify the path to save your model',
                        help='Where to save the trained model and evluation results')
    parser.add_argument('--matrix', type=str, default = 'Using the fillna_matrix_path in the preprocess_data.py',
                        help='The path for preprocessed factor matrices')
    parser.add_argument('--namatrix', type=str, default = 'Using the withna_matrix_path in the preprocess_data.py',
                        help='The path for factor matrices with nans')
    args = parser.parse_args()
    run(args)