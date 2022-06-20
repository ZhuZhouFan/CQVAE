import numpy as np
import pandas as pd
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
from Beta_VAE import Beta_VAE_Agent, VAE_Factor_Agent


def load_mat(factor_name, period, load_path):
    try:
        df = pd.read_csv(f'{load_path}/{factor_name}.csv', index_col = 'date')
        df = df.loc[period[0]:period[1], :]
        return df
    except Exception as e:
        return None


def obtain_permno_list(factor_name, period, load_path, na_percent = 0.75):
    df = load_mat(factor_name, period, load_path)
    indice = (df.isna().sum(axis = 0) < df.shape[0] * na_percent)
    permno_list = indice.loc[indice].index.values
    return permno_list


def rank_normalize_C(C):
    for j in range(C.shape[1]):
        tem = C[:, j, :]
        tem_ = pd.DataFrame(tem)
        C[:, j, :] = (2 * tem_.rank()/(tem_.shape[0] + 1) - 1).values
    return C


def run(args):
    cuda_dic = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7}
    torch.cuda.set_device(cuda_dic[args.cuda])
    factor_matrix_path = '/data/QAE/xiu_factor_matrix'
    K = args.K
    seed = args.seed
    AE_lr = 1e-4
    AEF_lr = 1e-3
    AE_epoch_num = 5000
    AE_factor_epoch_num = 5000
    f_hidden_dim = 64
    lam = 1e-5
    log_name = args.log_name
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
    time_set = []
    for i in range(1957, 2018, 5):
        time_set.append(int(str(i)+'0101'))
    period_set = []
    for i in range(len(time_set) - 6):
        period_set.append((time_set[i], time_set[i + 6]))
    period_set = period_set[:-1]
    log_set = []
    for i in range(len(time_set) - 1):
        log_set.append((time_set[i], time_set[i + 1]))
    log_set = log_set[6:]

    if not os.path.exists(log_name):
        os.mkdir(log_name)

    for i in range(len(period_set)):
        train_period = period_set[i]
        log_period = log_set[i]
        data_list = Parallel(n_jobs = 10)(delayed(load_mat)(factor_name, train_period, factor_matrix_path) for factor_name in tqdm(factor_list, desc = 'Loading factor matrixs'))
        permno_list = obtain_permno_list('RET', train_period, '/data/QAE/xiu_factor_matrix_with_na')
        T = data_list[0].shape[0]
        N = len(permno_list)
        P = len(data_list)
        r = load_mat('RET', train_period, factor_matrix_path)[permno_list].values.transpose()
        C = np.zeros((N, T, P))
        for i in range(P):
            C[:, :, i] = data_list[i][permno_list].values.transpose()
        C = rank_normalize_C(C)
        time_ = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(os.path.join(log_name, '%s-%s'%(log_period[0],log_period[1]))):
            os.mkdir(os.path.join(log_name, '%s-%s'%(log_period[0],log_period[1])))
        log_dir = os.path.join(log_name, '%s-%s'%(log_period[0], log_period[1]), str(args.seed)+'-'+str(time_))
        portfolio = np.zeros((P, T))
        for t in range(T):
            portfolio[:, t] = np.linalg.inv(C[:, t, :].transpose() @ C[:, t, :]) @ C[:, t, :].transpose() @ r[:, t]
        AE_agent = Beta_VAE_Agent(input_dim = N, 
                                  latent_dim = K, 
                                  output_dim = N, 
                                  learning_rate = AE_lr, 
                                  beta = 1, 
                                  seed = seed, 
                                  log_dir = log_dir)
        AE_feature = AE_label = torch.Tensor(r.transpose())
        AE_agent.load_data(feature = AE_feature, label = AE_label, valid_size = 1/3, test_size = 0, num_cpu = 0, batch_size = 32)
        AE_agent.train(AE_epoch_num, 0)
        AE_model_para = torch.load(f'{log_dir}/Beta_VAE_best.pth')
        AE_factor = VAE_Factor_Agent(N = N,
                                     T = T,
                                     P = P,
                                     K = K,
                                     f_hidden_dim = f_hidden_dim,
                                     model_para = AE_model_para,
                                     learning_rate = AEF_lr, 
                                     seed = seed,
                                     log_dir = log_dir)
        AE_factor.load_data(C = C, r = r, valid_size = 1/3, test_size = 0, batch_size = 32, num_cpu = 0)
        AE_factor.train(AE_factor_epoch_num, lam)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type = str, default = 'zero')
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--log_name', type = str, default = 'test')
    parser.add_argument('--K', type = int, default = 3)
    args = parser.parse_args()
    run(args)