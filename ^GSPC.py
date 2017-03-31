import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas_datareader import data

end = '2013-12-31'
start = '1999-12-31'
GSPC = data.DataReader('^GSPC', 'yahoo', start, end)

#print(GSPC)

#S&P500每日收盘价和成交量
f1 = plt.figure()
GSPC['Adj Close'].plot(legend=True)
f2 = plt.figure()
GSPC['Volume'].plot(legend=True)

#S&P500每日收益率
simple_ret = GSPC['Adj Close'].pct_change()
log_ret = np.log(1+simple_ret)
#log_ret = np.log(GSPC['Adj Close'] / GSPC['Adj Close'].shift(1))
f3 = plt.figure('S&P500 Daily return')
log_ret.plot()

#S&P500每日价格范围
daily_range = GSPC['High'] - GSPC['Low']
f4 = plt.figure('S&P500 daily range')
daily_range.plot()

#21天滚动波动率
f5 = plt.figure('S&P500 21day rolling volatility')
(np.sqrt(252) * pd.rolling_std(log_ret, 21)).plot()

#14年波动率
t = 1
for year in range(2000, 2014):
    print("year", year, "had vol of", format(np.std(log_ret[t:t+252])*np.sqrt(252), '0.2%') )
    t=t+252

plt.show()