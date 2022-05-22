import time
import numpy as np
import pandas as pd
from jqdatasdk import *
from sqlalchemy import create_engine

today = '-'.join([str(time.localtime().tm_year), str(time.localtime().tm_mon), str(time.localtime().tm_mday)])

auth('18510984298', 'Lmy123***')
stock_list = get_index_stocks('000300.XSHG')
engine = create_engine(
    "mysql+pymysql://root:zmq261317@localhost:3306/shares?charset=utf8"
)
count = 5000
fields = ['open', 'close', 'high', 'low', 'volume', 'money']

# all_days = get_trade_days(end_date=today, count=5000)

for stock in stock_list[:1]:
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
    data1 = get_fundamentals_continuously(q, end_date=today, count=count, panel=False)
    data2 = get_price(stock, count=count, end_date=today, fields=fields, panel=False)
    data2.dropna(axis=0, how='all', inplace=True)
    data2.index = range(data2.shape[0])
    data2.drop([data2.shape[0] - 1], inplace=True)
    data = pd.concat([data1, data2], axis=1)
    data = data[['day', 'code', 'open', 'close', 'high', 'low', 'volume', 'money',
                 'capitalization', 'circulating_cap', 'market_cap', 'circulating_market_cap', 'turnover_ratio',
                 'pe_ratio', 'pe_ratio_lyr', 'pb_ratio', 'ps_ratio', 'pcf_ratio']]
    data.to_sql(name='xshe' + stock[:6], con=engine, if_exists='replace', index=False)