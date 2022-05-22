import time
import pandas as pd
from datetime import datetime
from jqdatasdk import *
from sqlalchemy import create_engine


start_day = '2022-01-27'
end_day = '-'.join([str(time.localtime().tm_year), str(time.localtime().tm_mon), str(time.localtime().tm_mday)])

auth('18510984298', 'Lmy123***')
stock_list = get_index_stocks('000300.XSHG')
engine = create_engine(
    "mysql+pymysql://root:zmq261317@localhost:3306/shares?charset=utf8"
)
fields = ['open', 'close', 'high', 'low', 'volume', 'money']
all_days = list(get_trade_days(end_date=end_day, count=100))
count = len(all_days) - all_days.index(datetime.strptime(start_day, '%Y-%m-%d').date()) - 1
for stock in stock_list:
    q = query(
        valuation.capitalization,
        valuation.circulating_cap,
        valuation.market_cap,
        valuation.circulating_market_cap,
        valuation.turnover_ratio,
        valuation.pe_ratio,
        valuation.pe_ratio_lyr,
        valuation.pb_ratio,
        valuation.ps_ratio,
        valuation.pcf_ratio
    ).filter(valuation.code.in_([stock]))
    data1 = get_fundamentals_continuously(q, count=count, end_date=end_day, panel=False)
    if data1.shape[0] == count:
        data1.drop([0], inplace=True)
        data1.index = range(data1.shape[0])
    data2 = get_price(stock, start_date=start_day, end_date=end_day, fields=fields, panel=False)
    data2.index = range(data2.shape[0])
    data2.drop([0, data2.shape[0] - 1], inplace=True)
    data2.index = range(data2.shape[0])
    data = pd.concat([data1, data2], axis=1)
    data = data[['day', 'code', 'open', 'close', 'high', 'low', 'volume', 'money',
                 'capitalization', 'circulating_cap', 'market_cap', 'circulating_market_cap', 'turnover_ratio',
                 'pe_ratio', 'pe_ratio_lyr', 'pb_ratio', 'ps_ratio', 'pcf_ratio']]
    data.to_sql(name='xshe' + stock[:6], con=engine, if_exists='append', index=False)

