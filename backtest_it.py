import pandas as pd
import numpy as np


def backtest_it(vwap_mat:pd.DataFrame,
                factor_mat:pd.DataFrame,
                start_time:int,
                end_time:int,
                T_hold:int,
                tc:float = 0.003):

    factor_mat.sort_index(inplace = True)
    score_rank_mat = factor_mat.rank(axis = 1, method = "first", pct = True).copy()
    date_all_list = factor_mat.loc[start_time:end_time,:].index.values
    date_trade_list = date_all_list[::T_hold]
    df_backtest = pd.DataFrame(columns=[f'pnl{i}' for i in range(1,11)] + ['pnl_bench','turnoverD10'],index=date_trade_list,dtype=object)
    df_backtest['turnoverD10'] = 0
    df_backtest['cost'] = 0
    dict_port_list = [dict()] * 10
    dict_port_pre = dict()
    dict_port_short = dict()
    for idx_date,date in enumerate(date_trade_list):
        pick_cond = (~np.isnan(factor_mat.loc[date,:]))
        stocktransac = pick_cond[pick_cond==True].index
        if len(stocktransac) == 0:
            continue
        if len(stocktransac)>=400:
            dict_port_list = [dict()] * 10
            s_rank = score_rank_mat.loc[date,stocktransac]
            for decile in range(1,11):
                lb = (decile-1)*0.1 if decile>1 else -0.0000001
                ub = decile*0.1
                stock_in_port = s_rank[(s_rank>lb) & (s_rank<=ub)].index.values
                try:
                    dict_port_list[decile-1] = dict(zip(stock_in_port,np.zeros_like(stock_in_port)+1.0/len(stock_in_port)))
                except ZeroDivisionError:
                    print(stock_in_port)
        if idx_date < len(date_trade_list)-1:
            date_cp = date_trade_list[idx_date + 1]
        else:
            date_cp = date_all_list[-1]
        for idex_decile, dict_port in enumerate(dict_port_list):
            if not dict_port:
                continue
            decile = idex_decile + 1
            price0_port = vwap_mat.loc[date,dict_port.keys()]
            price1_port = vwap_mat.loc[date_cp,dict_port.keys()]
            ret_port = price1_port/price0_port-1
            weight_port = np.array(list(dict_port.values()))
            df_backtest.loc[date_cp,f'pnl{decile}'] = np.sum(ret_port*weight_port)
            if idx_date > 0 and decile==10:
                turnover = 0
                stock_new = set(dict_port.keys()) - set(dict_port_pre.keys())
                for k in stock_new:
                    turnover += dict_port[k]
                stock_keep = set(dict_port.keys()) & set(dict_port_pre.keys())
                for k in stock_keep:
                    turnover += max(0,dict_port[k]-dict_port_pre[k])
                df_backtest.loc[date_cp,'turnoverD10'] = turnover
                df_backtest.loc[date_cp,'costD10'] = tc*turnover
            if decile==10:
                weigth_port_cp = weight_port*price1_port/price0_port
                weigth_port_cp /= np.sum(weigth_port_cp)
                dict_port_pre = dict(zip(dict_port.keys(),weigth_port_cp))
            if idx_date > 0 and decile==1:
                turnover_short = 0
                stock_new = set(dict_port.keys()) - set(dict_port_short.keys())
                for k in stock_new:
                    turnover_short += dict_port[k]
                stock_keep = set(dict_port.keys()) & set(dict_port_short.keys())
                for k in stock_keep:
                    turnover_short += max(0,dict_port[k] - dict_port_short[k])
                df_backtest.loc[date_cp,'turnoverD1'] = turnover_short
                df_backtest.loc[date_cp,'costD1'] = tc*turnover_short
            if decile == 1:
                weigth_port_cp_short = weight_port*price1_port/price0_port
                weigth_port_cp_short /= np.sum(weigth_port_cp_short)
                dict_port_short = dict(zip(dict_port.keys(), weigth_port_cp_short))
            
    for decile in range(1,11):
        df_backtest[f'nav{decile}'] = (1 + df_backtest[f'pnl{decile}']).cumprod()
    df_backtest['pnl_long'] = df_backtest['pnl10'] - df_backtest['costD10']
    df_backtest['nav_long'] = (1 + df_backtest['pnl_long']).cumprod()
    df_backtest['pnl_long_short'] = df_backtest['pnl10'] - df_backtest['pnl1'] - df_backtest['costD10']  - df_backtest['costD1'] 
    df_backtest['nav_long_short'] = (1 + df_backtest['pnl_long_short']).cumprod()
    return df_backtest