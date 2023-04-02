import pandas as pd
import numpy as np
import yfinance as yf
import os


"""
    standart deviation ve log return un ve mean in 
    window rolling sma ve ema ortalamalri ile fiyati 
    ve urunu analiz etmeye calisiyor !!!!!!!!!!!!!!!!!!!!
"""

def create_close_df_yf(symbols: list, start: str, end: str) -> pd.DataFrame():
    """
    df must have PRice and Return columns
    """
    close = pd.DataFrame()
    for i in symbols:
       temp = yf.download(i, start, end) 
       temp.index = pd.to_datetime(temp.index).date
       close[i] = temp.Close
    return close

def norm_close_df(df: pd.DataFrame()) -> pd.DataFrame():
    norm = pd.DataFrame()
    norm = df.div(df.iloc[0]).mul(100)
    return norm

def compound_annual_growth_rate(df: pd.DataFrame()) -> float:
    norm = df.Price[-1]/df.Price[0]
    cagr = (norm)**(1/((df.index[-1] - df.index[0]).days / 365.25)) - 1
    return cagr

def multiple_investment(df: pd.DataFrame(), returns: str) -> float:
    return (1 + df[returns]).prod()

def norm_compound_returns(df: pd.DataFrame(), col: str):
    return df[col].add(1).cumprod()

def norm_cumulative_returns(df: pd.Dataframe(), log_ret:str):
    return df[log_ret].cumsum().apply(np.exp)

def log_cumulative_returns(df: pd.DataFrame(), log_returns: str):
    return np.exp(df[log_returns].sum())

def geo_mean_return(df: pd.DataFrame()) -> float:
    multiple = multiple_investment(df)
    n = df.Returns.count()
    geo_mean = multiple**(1/n) - 1
    return geo_mean

def calc_eff_ann_rate(pv, r, n, m=1):
    """
    pv: your investment
    fv: effective annual rate
    m: quarter how many? - monthly=12
    """
    fv = pv *(1+r/m)**n*m
    return (fv/pv)**(1/n)-1

def calc_eff_ann_rate_exp(r, m=1):
    return np.exp(r) - 1

def create_log_return(df: pd.DataFrame(), close: str) -> None:
    df["log_ret"] = np.log(df[close] / df[close].shift())
    
