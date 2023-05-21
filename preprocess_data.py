
'''
This code is used to preprocess the downloaded data from Prof. Xiu's homepage.

The workflow of this file could be abstracted as the following steps:
1. Extract the factor matrices from the raw data, where each factor matrix is a [T, N] matrix;
2. Impute the missing values of factor matrices with their cross-sectional medians.

Before executing the code, kindly specify the directory of the downloaded data and the directories where the factor matrices will be saved.

'''
import os
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np

downloaded_data_path = ''
withna_matrix_path = ''
fillna_matrix_path = ''

if not os.path.exists(withna_matrix_path):
    os.mkdir(withna_matrix_path)
    
if not os.path.exists(fillna_matrix_path):
    os.mkdir(fillna_matrix_path)

full_df = pd.read_csv(downloaded_data_path)
full_df['DATE'] = pd.to_datetime(full_df['DATE'], format='%Y%m%d').apply(lambda x: f'{x.year}-{x.month:02d}-{x.day:02d}')
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

def re_matrix(factor, df, save_path):
    permno_list = df['permno'].unique()
    permno_list.sort()
    date_list = df['DATE'].unique()
    date_list = date_list[(date_list >= '1957-01-01') * (date_list <= '2017-01-01')]
    date_list.sort()
    
    df_matrix = pd.DataFrame(columns = permno_list, index = date_list)
    df_matrix.index.name = 'date'
    
    for permno in permno_list:
        df_ = df.loc[df['permno'] == permno, ['DATE', factor]]
        df_.set_index('DATE', inplace = True)
        df_.sort_index(inplace = True)
        df_matrix[permno] = df_[factor]
    
    df_matrix.to_csv(f'{save_path}/{factor}.csv')
    
for factor in tqdm(factor_list + ['RET']):
    re_matrix(factor, full_df, withna_matrix_path)
    
def fill_na(factor, load_path, save_path):
    factor_matrix = pd.read_csv(f'{load_path}/{factor}.csv', index_col = 'date')
    factor_matrix[np.isinf(factor_matrix)] = np.nan
    date_list = factor_matrix.index.values
    
    for date in date_list:
        not_na_sum = factor_matrix.loc[date, :].notna().sum()
        if not_na_sum > 0:
            cross_median = np.nanmedian(factor_matrix.loc[date, :].values)
            factor_matrix.loc[date, :] = factor_matrix.loc[date, :].fillna(cross_median)

    factor_matrix.fillna(method = 'ffill', inplace = True, axis = 0)
    factor_matrix.fillna(method = 'bfill', inplace = True, axis = 0)
    factor_matrix.to_csv(f'{save_path}/{factor}.csv')
    
Parallel(n_jobs = 20)(delayed(fill_na)
                      (factor, withna_matrix_path, fillna_matrix_path) for factor in tqdm(factor_list + ['RET']))