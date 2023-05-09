from KZ_project.ml_pipeline.services.tradingview_service.tradingview_service import TradingViewService

#screener="turkey"
#exchange="BIST"
#path_symbol = './data/symbol_data/ana_market.csv'
#index = 'Order'
screener="crypto"
exchange="BINANCE"
path_symbol = './data/symbol_data/usdtcoins.csv'
index = 'Order'

def trading_filter_strategyu_hisse_result(interval: str):
    st_main_market = TradingViewService(exchange, screener, path_symbol)
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

def listen_live_signal():
    st_main_market = TradingViewService(exchange, screener, path_symbol)
    st_main_market.listen_live_signal('1h')
    
    

if __name__ == '__main__':    
    listen_live_signal()