import time
import pymysql
import pandas as pd
import numpy as np
from jqdatasdk import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.losses import mean_squared_error



class my_lsml():
    """
    定义一个LSTM模型的类
    固定某些参数
    """
    def __init__(self, shape_1, shape_2):
        """
        定义

        :param shape_1: 训练集列，因子数
        :param shape_2: 训练集行，时间跨度——50

        以及其他参数设置：
        """
        self.epoch = 3
        self.batch_size = 16
        self.model = Sequential()
        self.model.add(LSTM(units=100, return_sequences=True, input_dim=shape_1, input_length=shape_2))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit(self, x_train, y_train):
        """
        模型训练

        :param x_train: 训练集——因子
        :param y_train: 训练集——标签
        :return: 无
        """
        self.model.fit(x_train, y_train, epochs=self.epoch, batch_size=self.batch_size, verbose=1)

    def predict(self, x_valid):
        """
        模型预测

        :param x_valid: 预测所用数据，因子
        :return: 返回预测结果
        """
        return self.model.predict(x_valid)


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

    data = pd.read_sql("select * from %s where day <= '%s' order by day desc limit 2000" % ('x' + stock[:6], start_day),
                       con=con)
    data = data.iloc[::-1]
    data.index = range(len(data))

    return data


def clean_data(raw):
    """
    数据清洗 —— 缺失值填充、标准化

    :param raw: 股票数据  DataFrame
    :return: 清洗结果：训练集、验证集以及预测使用数据  DataFrame
    """
    raw = raw.set_index('day').drop('code', axis=1)
    raw = raw.fillna(method='bfill')
    raw = raw.fillna(method='ffill')
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(raw)
    X = np.array([data[i - time_step:i] for i in range(time_step, len(data) - time_space)])  # 因子数据
    Y = np.array([data[i, 0] for i in range(time_step + time_space, len(data))])  # 标签数据
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    x_now = np.array([np.array(data[-50:])])

    return x_train, x_test, y_train, y_test, x_now


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
    lstm = my_lsml(x_train.shape[-1], x_train.shape[1])
    lstm.fit(x_train, y_train)
    y_predict = lstm.predict(x_test)
    loss = sum(mean_squared_error(y_test, y_predict))  # 损失函数——均方差
    print('损失为', loss)
    predict = lstm.predict(x_now)
    print("预测结果为", predict[0][0])

    return [loss, (predict[0][0] - x_now[0, 49, 0]) / x_now[0, 49, 0]]


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
    # final_result.to_csv('D:\\py_learn\\量化投资\\LSML_result_20.csv', encoding='utf-8')


# today = '-'.join([str(time.localtime().tm_year), str(time.localtime().tm_mon), str(time.localtime().tm_mday)])
today = '2021-10-15'
time_step = 50
result_dic = {}

time_space = 15
con = pymysql.connect(host='localhost', port=3306, user='root', passwd='zmq261317', db='shares')
cur = con.cursor()
auth('18611980865', 'Th503221')
stock_list = get_index_stocks('000300.XSHG')

all_days = get_trade_days(end_date=today, count=1000)
df = pd.DataFrame(all_days)
df.index = pd.to_datetime(all_days)
day_list = list(df.resample('m'))
mouth_days = [str(i[1].iloc[0, 0]) for i in day_list[1:]]
mouth_days = mouth_days[::-1]

main()



