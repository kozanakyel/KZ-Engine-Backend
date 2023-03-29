from KZ_project.strategy_kz.tradingviewstrategy import TradingViewStrategy
from KZ_project.technical_analysis.backtest_kz import *

#screener="turkey"
#exchange="BIST"
#path_symbol = './data/symbol_data/ana_market.csv'
#index = 'Order'
screener="crypto"
exchange="BINANCE"
path_symbol = './data/symbol_data/usdtcoins.csv'
index = 'Order'

def test_trading_filter_strategyu_hisse_result(interval: str):
    st_main_market = TradingViewStrategy(exchange, screener, path_symbol)
    filter_list = st_main_market.get_st_result_list(interval)
    
    filtre = []
    if screener == 'crypto':
        for i in filter_list:
            r = i[-4:]
            r = f'{i[:-4]}-{r[:-1]}'
            filtre.append(r)
    elif screener == 'turkey':
        for i in filter_list:
            filtre.append(f'{i}.IS')
    return filter_list

def test_listen_live_signal():
    st_main_market = TradingViewStrategy(exchange, screener, path_symbol)
    st_main_market.listen_live_signal('1h')
    
    

if __name__ == '__main__':

    #result_list = test_trading_filter_strategyu_hisse_result('1h')
    #print(result_list)
    
    test_listen_live_signal()