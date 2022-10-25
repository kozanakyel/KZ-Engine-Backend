import pandas as pd
import matplotlib.pyplot as plt
import os
pd.options.display.float_format = '{:.4f}'.format
plt.style.use("seaborn")


def make_backtest(df, range) -> dict:
    result_list = []
    write_bt_crossover(df, 'sma', bt_crossover, range, result_list)
    write_bt_band(df, range, result_list)
    write_bt_crossover(df, 'dema', bt_crossover, range, result_list)
    write_bt_crossover(df, 'ema', bt_crossover, range, result_list)
    write_bt_crossover(df, 'kama', bt_crossover, range, result_list)
    write_bt_crossover(df, 't3', bt_crossover, range, result_list)
    write_bt_crossover(df, 'tema', bt_crossover, range, result_list)
    write_bt_crossover(df, 'TRIMA', bt_crossover, range, result_list)
    write_bt_crossover(df, 'wma', bt_crossover, range, result_list)
    write_bt_band(df, range, result_list, up='dmi_up', low='dmi_down')
    write_bt_crossover(df, 'cmo', bt_crossover, range, result_list)
    write_bt_range(df, 'cci', bt_threshold, 200, -150, range, result_list)
    write_bt_range(df, 'rsi', bt_threshold, 80, 20, range, result_list)
    write_bt_range(df, 'wllr', bt_threshold, -20, -80, range, result_list)
    write_bt_range(df, 'mfi', bt_threshold, 90, 10, range, result_list)
    write_bt_macd(df, bt_band_range, result_list)
    result_list.sort(reverse=True)
    return result_list


def bt_crossover(df: pd.DataFrame(), upper_thresh, 
                lower_thresh, start_budget=1000) -> pd.DataFrame():
    sample = df.copy()
    entry_time = []
    exit_time = []

    entry_price = []
    exit_price = []

    trade_taken = False

    for index,datetime in enumerate(sample.index):
        current_datetime = datetime
        upper = df[upper_thresh].iloc[index]
        lower = df[lower_thresh].iloc[index]
        close = sample['Close'].iloc[index]
        
        
        if (upper > lower) and (trade_taken == True):
            continue
            
        if ((upper > lower) and (trade_taken == False)):
            trade_taken = True

            entry_time.append(current_datetime)
            entry_price.append(close)

        elif trade_taken and (lower >= upper):
            trade_taken = False
            exit_time.append(current_datetime)
            exit_price.append(close)

        elif (index == (len(sample) - 1)) and (trade_taken != False):
            trade_taken = False
            exit_time.append(current_datetime)
            exit_price.append(close)
    
    #print(len(entry_time), len(exit_time), len(entry_price), len(exit_price))
    if len(entry_time) != len(exit_time):
        entry_time.pop()
        entry_price.pop()
    #print(len(entry_time), len(exit_time), len(entry_price), len(exit_price))
    trade_sheet = pd.DataFrame({"entry_time" :entry_time,
                               "exit_time" : exit_time,
                               "entry_price" : entry_price,
                               "exit_price" : exit_price})

    # calculating pnl from trade sheet
    start_budget = 1000
    for i in range(len(trade_sheet['entry_time'])):
        trade_sheet.loc[i, "pnl_percent"] = (trade_sheet.loc[i,'exit_price'] - trade_sheet.loc[i,'entry_price'])/trade_sheet.loc[i,'entry_price']
        trade_sheet.loc[i, "pnl_cash"] = 1000 * trade_sheet.loc[i, "pnl_percent"] + start_budget - start_budget*0.001
        start_budget = trade_sheet.loc[i, "pnl_cash"]
    
    return trade_sheet

def bt_plot_crossover(df: pd.DataFrame(), trade_sheet: pd.DataFrame(), param1, param2):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,6))
    ax1.plot(df.index, df['Close'], label='Close')
    ax1.plot(df.index, df[param1], label=param1)
    ax1.plot(df.index, df[param2], label=param2)
    ax1.scatter(trade_sheet['entry_time'], trade_sheet['entry_price'], marker='o', color='green', label='buy')
    ax1.scatter(trade_sheet['exit_time'], trade_sheet['exit_price'], marker='x', color='red', label='sell')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.scatter(trade_sheet['entry_time'], trade_sheet['entry_price'], marker='o', color='green', label='buy')
    ax2.scatter(trade_sheet['exit_time'], trade_sheet['exit_price'], marker='x', color='red', label='sell')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price')
    ax2.legend()

    ax3.plot(trade_sheet['entry_time'], trade_sheet['pnl_cash'], label='buy')
    ax3.set_title('pnl_cash')
    ax3.legend()

def bt_plot_ind(df: pd.DataFrame(), trade_sheet: pd.DataFrame(), param1):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,6))
    ax1.plot(df.index, df['Close'], label='Close')
    ax1.scatter(trade_sheet['entry_time'], trade_sheet['entry_price'], marker='o', color='green', label='buy')
    ax1.scatter(trade_sheet['exit_time'], trade_sheet['exit_price'], marker='x', color='red', label='sell')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend()

    ax2.plot(df.index, df[param1], label=param1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel(param1)
    ax2.legend()

    ax3.plot(trade_sheet['entry_time'], trade_sheet['pnl_cash'], label='buy')
    ax3.set_title('pnl_cash')
    ax3.legend()



def bt_band_range(df: pd.DataFrame(), upper_thresh, 
                lower_thresh, start_budget=1000) -> pd.DataFrame():
    sample = df.copy()
    entry_time = []
    exit_time = []

    entry_price = []
    exit_price = []

    trade_taken = False

    for index,datetime in enumerate(sample.index):
        current_datetime = datetime
        upper = df[upper_thresh].iloc[index]
        lower = df[lower_thresh].iloc[index]
        close = sample['Close'].iloc[index]
        
        if (lower < close) and (upper > close) and (trade_taken == True):
            continue
            
        if ((lower >= close) and (trade_taken == False)):
            trade_taken = True

            entry_time.append(current_datetime)
            entry_price.append(close)
            #print(current_datetime, close, 'burda')

        elif trade_taken and (upper <= close):
            trade_taken = False
            exit_time.append(current_datetime)
            exit_price.append(close)

        elif (index == (len(sample) - 1)) and (trade_taken != False):
            trade_taken = False
            exit_time.append(current_datetime)
            exit_price.append(close)
    
    #print(len(entry_time), len(exit_time), len(entry_price), len(exit_price))
    if len(entry_time) != len(exit_time):
        entry_time.pop()
        entry_price.pop()
    #print(len(entry_time), len(exit_time), len(entry_price), len(exit_price))
    trade_sheet = pd.DataFrame({"entry_time" :entry_time,
                               "exit_time" : exit_time,
                               "entry_price" : entry_price,
                               "exit_price" : exit_price})

    # calculating pnl from trade sheet
    start_budget = 1000
    for i in range(len(trade_sheet['entry_time'])):
        trade_sheet.loc[i, "pnl_percent"] = (trade_sheet.loc[i,'exit_price'] - trade_sheet.loc[i,'entry_price'])/trade_sheet.loc[i,'entry_price']
        trade_sheet.loc[i, "pnl_cash"] = 1000 * trade_sheet.loc[i, "pnl_percent"] + start_budget - start_budget*0.001
        start_budget = trade_sheet.loc[i, "pnl_cash"]

    return trade_sheet



def bt_threshold(df: pd.DataFrame(), indicator, upper_thresh, 
                lower_thresh, start_budget=1000) -> pd.DataFrame():
    sample = df.copy()
    entry_time = []
    exit_time = []

    entry_price = []
    exit_price = []

    trade_taken = False

    for index,datetime in enumerate(sample.index):
        current_datetime = datetime
        indicat = df[indicator].iloc[index]
        close = sample['Close'].iloc[index]
        
        if (lower_thresh < indicat) and (upper_thresh > indicat) and (trade_taken == True):
            continue
            
        if ((lower_thresh >= indicat) and (trade_taken == False)):
            trade_taken = True

            entry_time.append(current_datetime)
            entry_price.append(close)
            #print(current_datetime, close, 'burda')

        elif trade_taken and (upper_thresh <= indicat):
            trade_taken = False
            exit_time.append(current_datetime)
            exit_price.append(close)

        elif (index == (len(sample) - 1)) and (trade_taken != False):
            trade_taken = False
            exit_time.append(current_datetime)
            exit_price.append(close)
    
    #print(len(entry_time), len(exit_time), len(entry_price), len(exit_price))
    if len(entry_time) != len(exit_time):
        entry_time.pop()
        entry_price.pop()
    #print(len(entry_time), len(exit_time), len(entry_price), len(exit_price))
    trade_sheet = pd.DataFrame({"entry_time" :entry_time,
                               "exit_time" : exit_time,
                               "entry_price" : entry_price,
                               "exit_price" : exit_price})

    # calculating pnl from trade sheet
    start_budget = 1000
    for i in range(len(trade_sheet['entry_time'])):
        trade_sheet.loc[i, "pnl_percent"] = (trade_sheet.loc[i,'exit_price'] - trade_sheet.loc[i,'entry_price'])/trade_sheet.loc[i,'entry_price']
        trade_sheet.loc[i, "pnl_cash"] = 1000 * trade_sheet.loc[i, "pnl_percent"] + start_budget - start_budget*0.001
        start_budget = trade_sheet.loc[i, "pnl_cash"]

    return trade_sheet


def write_bt_crossover(df: pd.DataFrame, ind: str, func_bt, range: list, result: list) -> None:
    res = {} 
    for i in range:
        for k in range:
            if k < i:
                temp_bctest_df = func_bt(df, ind+'_'+str(k), ind+'_'+str(i))
                if 'pnl_cash' in temp_bctest_df.columns:
                    res[ind+'_'+str(k)+'_'+str(i)] = temp_bctest_df['pnl_cash'].iloc[-1], len(temp_bctest_df)               

    if(len(res) > 0):
        res = max(res.values()), max(res, key=res.get)
        result.append(res)
    else:
        print('no entry')

def write_bt_range(df: pd.DataFrame, ind: str, func_bt, upper_thresh, lower_thresh, range: list, result: list) -> None:
    res = {} 
    for i in range:
        temp_bctest_df = func_bt(df, ind+'_'+str(i), upper_thresh, lower_thresh)
        if 'pnl_cash' in temp_bctest_df.columns:
            res[ind+'_'+str(i)] = temp_bctest_df['pnl_cash'].iloc[-1], len(temp_bctest_df)
   
    if(len(res) > 0):
        res = max(res.values()), max(res, key=res.get)
        result.append(res)
    else:
        print(f'no entry {ind}_{str(i)}')

def write_bt_band(df: pd.DataFrame, range: list, result: list, up='upband', low='lowband') -> None:
    result_band = {}
    for i in range:
        temp_band_df = bt_band_range(df, up+'_'+str(i), low+'_'+str(i))
        if 'pnl_cash' in temp_band_df.columns:
            result_band['band_'+str(i)] = temp_band_df['pnl_cash'].iloc[-1], len(temp_band_df)

    if(len(result_band) > 0):
        res = max(result_band.values()), max(result_band, key=result_band.get)
        result.append(res)
    else:
        print(f'no entry band {str(i)}')

def write_bt_macd(df: pd.DataFrame, func_bt, result: list, up='macd', low='macdsignal') -> None:
    result_band = {}
    temp_band_df = func_bt(df, up, low)
    if 'pnl_cash' in temp_band_df.columns:
        result_band['macd_'] = temp_band_df['pnl_cash'].iloc[-1], len(temp_band_df)

    if(len(result_band) > 0):
        res = max(result_band.values()), max(result_band, key=result_band.get)
        result.append(res)
    else:
        print('no entry macd')


def write_backtest_result(df, symbol, period, interval, result_list=None):
    path_df = './outputs/data_ind/'+symbol
    if result_list == None:
        result_list = make_backtest(df, range)
    file_res_indicator = f'res_ind_{symbol}_{period}_{interval}.csv'
    with open(os.path.join(path_df, file_res_indicator), mode='w+') as res_file:
        for i in result_list:
            pnl = i[0]
            res_file.write(f'{pnl[0]},{pnl[1]},{i[1]},\n')
 