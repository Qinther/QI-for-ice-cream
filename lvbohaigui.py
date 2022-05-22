import time
import threading
import pandas as pd
from jqdatasdk import *
from bayes_opt import BayesianOptimization


today = '-'.join([str(time.localtime().tm_year), str(time.localtime().tm_mon), str(time.localtime().tm_mday)])
count = 500  # 获取的数据量
auth('18981215602', 'Zmq261317')
share_data = get_price("300288.XSHE", count=count, end_date=today, fields=['open', 'close', 'high', 'low'],
                       panel=False)
date_index = list(share_data.index)  # 所有交易日日期与时间
start_day = date_index[301]
total = 10000  # 总金
result_pool = []  # 储存优化结果


def turtle_trade(atr, up, down, start_date=start_day):
    """
    海龟交易函数，将交易信息保存在 trade_inf 中
    :param atr: ATR 平均波动幅度
    :param up: 通道上轨
    :param down: 通道下轨
    :param start_date: 交易起始日期
    :return: 收益率
    """

    # 每只股票的交易信息，包含起始时间，结束时间，每次交易的时间，仓位（Uint），收益率
    trade_inf = {
        "start-date": None,
        "end-date": None,
        "trade-date": [],
        "position": 0,
        "price": 0,
        "profit": 0,
    }
    j = date_index.index(start_date)
    while j < count:
        if (trade_inf["position"] == 0) & (share_data.iloc[j, 0] > up):
            trade_inf["position"] += 1
            trade_inf["price"] += share_data.iloc[j, 0]
            trade_inf["start-date"] = date_index[j]
            trade_inf["trade-date"].append(date_index[j])
        while trade_inf["position"] > 0:
            if share_data.iloc[j, 0] > share_data.loc[trade_inf["trade-date"][-1], "open"] + 0.5 * atr:
                trade_inf["position"] += 1
                trade_inf["price"] += share_data.iloc[j, 0]
                trade_inf["trade-date"].append(date_index[j])
            if share_data.iloc[j, 0] < share_data.loc[trade_inf["trade-date"][-1], "open"] - 2.0 * atr:
                trade_inf["profit"] = (trade_inf["position"] * share_data.iloc[j, 1] - trade_inf["price"]) / trade_inf[
                    "price"]
                trade_inf["position"] = 0
                trade_inf["end-date"] = date_index[j]
                return trade_inf["profit"]
            if share_data.iloc[j, 0] < down:
                trade_inf["profit"] = (trade_inf["position"] * share_data.iloc[j, 1] - trade_inf["price"]) / trade_inf[
                    "price"]
                trade_inf["position"] = 0
                trade_inf["end-date"] = date_index[j]
                return trade_inf["profit"]
            j += 1
        j += 1
    else:
        trade_inf["profit"] = (trade_inf["position"] * share_data.iloc[j - 1, 1] - trade_inf["price"]) / trade_inf[
            "price"]
        return trade_inf["profit"] if trade_inf["profit"] == trade_inf["profit"] else 0


def bayes_optimization(atr, up, down):
    """
    贝叶斯优化函数
    对传入的三个参数进行优化，优化范围为-10% ~ +10%
    :param atr: ATR
    :param up: Up
    :param down: Down
    :return: 无
    """
    bayes = BayesianOptimization(
        turtle_trade,
        {"atr": (atr - 0.1 * atr, atr + 0.1 * atr),
         "up": (up - 0.1 * up, up + 0.1 * up),
         "down": (down - 0.1 * down, down + 0.1 * down)},
    )
    bayes.maximize()
    result_pool.append(bayes.max)


# b, a = signal.butter(2, 0.1, btype='lowpass', analog=False, output='ba')
# dic = {}
# for i in ['open', 'close', 'high', 'low']:
#     x = share_data.loc[:, i]
#     y = signal.filtfilt(b, a, x, method='pad')
#     dic[i] = y
# share_data = pd.DataFrame(dic, index=index)

share_data['tr'] = None  # TR
share_data['atr'] = None  # ATR
share_data['up'] = None  # Up
share_data['down'] = None  # Down
share_data['unit'] = None  # Unit

# 计算TR，ATR，Up，Down，Uint
for i in range(1, share_data.shape[0]):
    # TR
    share_data.iloc[i, 4] = max([share_data.iloc[i, 2] - share_data.iloc[i, 3],
                                 share_data.iloc[i, 2] - share_data.iloc[i - 1, 1],
                                 share_data.iloc[i - 1, 1] - share_data.iloc[i, 3]])

    # ATR
    if i < 20:
        share_data.iloc[i, 5] = sum(share_data.iloc[1:i + 1, 4]) / i
    else:
        share_data.iloc[i, 5] = sum(share_data.iloc[i - 19:i + 1, 4]) / 20

    # Up & Down
    if i < 10:
        share_data.iloc[i, 6] = max(share_data.iloc[:i, 2])
        share_data.iloc[i, 7] = min(share_data.iloc[:i, 3])
    else:
        share_data.iloc[i, 6] = max(share_data.iloc[i - 10:i, 2])
        share_data.iloc[i, 7] = min(share_data.iloc[i - 10:i, 3])

    # UNIT
    share_data.iloc[i, 8] = 0.01 * total / share_data.iloc[i, 5]

a = turtle_trade(share_data.iloc[301, 5], share_data.iloc[301, 6], share_data.iloc[301, 7])
# b = bayes_optimization(share_data.iloc[100, 5], share_data.iloc[100, 6], share_data.iloc[100, 7])

# 优化的参数初始值
# param_dict = [
#     {
#         'atr': 0.3,
#         'up': 17.3,
#         'down': 16.1
#     },
#     {
#         'atr': 0.4,
#         'up': 17.2,
#         'down': 16.2
#     },
# ]
# # 做两次迭代优化
# for i in range(2):
#     threads = [threading.Thread(target=bayes_optimization, kwargs=item) for item in
#                param_dict]  # 依据param_dict内有几组参数创建几个线程
#     for thread in threads:
#         thread.start()
#     for thread in threads:
#         thread.join()
#     result_pool.sort(key=lambda i: i['target'], reverse=True)  # 将每个线程优化的结果保留并排序
#     param_dict = [item['params'] for item in result_pool[:2]]  # 挑选排名靠前的几个参数进行下一次迭代优化

# 保存优化的得到的最优参
# final_result = pd.DataFrame(result_pool[0])
# final_result.to_csv('D:\\py_learn\\量化投资\\海龟.csv')
