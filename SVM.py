import time
import pymysql
import numpy as np
import pandas as pd
from jqdatasdk import *
from sklearn import tree, svm, preprocessing
from multiprocessing.dummy import Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def get_stock_data(stock, start_day):
    """
    获取某一个股票在某日期之前的数据

    :param stock: 股票代码  ‘000001.XSGH’ str
    :param start_day: 终止日期  ‘2021-10-15’ str
    :return: 返回股票数据  DataFrame

    两种方式：
    直接利用聚宽获取 -0
    从本地数据库获取 -1
    """
    # q = query(
    #     valuation.market_cap,
    #     valuation.turnover_ratio,
    #     valuation.pe_ratio,
    #     valuation.pe_ratio_lyr,
    #     valuation.pb_ratio,
    #     valuation.ps_ratio,
    #     valuation.pcf_ratio
    # ).filter(valuation.code.in_([stock]))
    # data = get_fundamentals_continuously(q, end_date=today, count=3000, panel=False)

    data = pd.read_sql("select * from %s where day <= '%s' order by day desc limit 2000" % ('xshe' + stock[:6], start_day),
                       con=con)
    data = data.iloc[::-1]
    data.index = range(len(data))
    return data


def clean_data(data):
    """
    数据清洗 —— 缺失值填充、标准化

    :param data: 股票数据  DataFrame
    :return: 清洗结果：训练集、验证集以及预测使用数据  DataFrame
    """
    data = data.fillna(method='bfill')
    data = data.fillna(method='ffill')
    data = preprocessing.scale(data.iloc[:, 2:])
    data = pd.DataFrame(data)
    data_0 = data.iloc[:0 - time_step, :]
    data_1 = data.iloc[1:0 - time_step + 1, :]
    data_1.index = range(len(data_1))
    data_15 = data.iloc[time_step:, :]
    data_15.index = range(len(data_15))
    data_x = data_1 - data_0
    data_y = round((data_15 - data_0) / data_0, 3)
    x = np.array(data_x.iloc[:, 1:])
    y = np.array([int(i*1000) for i in data_y.iloc[:, 0]])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    x_now = np.array(data.iloc[-2, 1:] - data.iloc[-1, 1:])
    np.reshape(x_now, (1, len(x_now)))
    return x_train, x_test, y_train.T, y_test.T, x_now


def model_train(x_train, x_test, y_train, y_test, x_now):
    """
    训练模型，预测

    :param x_train: 训练集因子
    :param x_test: 验证集因子
    :param y_train: 训练集标签
    :param y_test: 验证集标签
    :param x_now: 预测所用因子数据
    :return: 返回list[损失，预测结果（增长率）]
    """
    svm_tool = svm.SVC(kernel='linear')
    svm_tool.fit(x_train, y_train)
    y_predict = svm_tool.predict(x_test)
    result = np.array(y_test) - np.array(y_predict)
    success_rate = sum([1 if i != 0 else 0 for i in result]) / len(result)
    loss = mean_squared_error(y_test, y_predict)
    print('损失为', loss)

    predict = svm_tool.predict(x_now.reshape(1, len(x_now)))
    print("预测结果为", predict)

    return [loss, predict[0]]


def main():
    """
    主程序
    以每个月第一个交易日为期，对所有股票进行模型训练，并预测。将结果排序，取最优20只股，保存到本地。

    :return: 无
    """
    final_result = {}
    for day in mouth_days[:1]:
        result_dic[day] = []
        for stock in stock_list:
            print("获取股票数据；")
            data = get_stock_data(stock, day)
            if len(data) < 100:
                break
            print("处理股票数据；")
            xtrain, xtest, ytrain, ytest, x_now = clean_data(data)
            print("开始预测；", stock)
            r = model_train(xtrain, xtest, ytrain, ytest, x_now)
            result_dic[day].append([stock] + r)
            print("预测完成！")

        result_dic[day].sort(key=lambda i: i[1])
        success_result = result_dic[day][:100]
        success_result.sort(key=lambda i: i[2], reverse=True)
        final_result[day] = [i[0] for i in success_result[:20]]
    final_result = pd.DataFrame(final_result).T
    print(final_result)
    # final_result.to_csv('D:\\py_learn\\量化投资\\svm_result.csv')



# today = '-'.join([str(time.localtime().tm_year), str(time.localtime().tm_mon), str(time.localtime().tm_mday)])
today = '2021-10-15'
time_step = 15
con = pymysql.connect(host='localhost', port=3306, user='root', passwd='zmq261317', db='shares')
cur = con.cursor()

result_dic = {}
final_result = {}
# auth('18611980865', 'Th503221')
stock_list = get_index_stocks('000300.XSHG')

all_days = get_trade_days(end_date=today, count=1000)
df = pd.DataFrame(all_days)
df.index = pd.to_datetime(all_days)
day_list = list(df.resample('m'))
mouth_days = [str(i[1].iloc[0, 0]) for i in day_list[1:]]
mouth_days = mouth_days[::-1]

main()

# pool = Pool(30)
# for i in stock_list:
#     pool.apply_async(main, (i,))
# pool.close()
# pool.join()




