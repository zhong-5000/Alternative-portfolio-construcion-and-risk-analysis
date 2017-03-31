import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bs
from pandas_datareader import data
from datetime import date
from scipy.stats import norm
from pandas_datareader._utils import RemoteDataError

site = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
page = urllib.request.urlopen(site)
soup = bs(page.read(), "html5lib")  # soup中存储了整个网页的源码
table = soup.find('table', {'class': 'wikitable'})  # table中存储了网页中SP500表格的源码

SD = dict()  # 字典用来存放ticker和所在行业
for row in table.findAll('tr'):  # 'tr'是HTML语言中的行
    col = row.findAll('td')  # 'td'是HTML中的列
    if len(col) > 0:
        ticker = str(col[0].string.strip())
        sector = str(col[3].string.strip()).lower()
        SD[ticker] = sector

SP500 = pd.Series(SD)  # 把字典转成series格式
SP500.to_csv('/Users/wuqianzhong/Desktop/NYU/Risk Management Maymin/SP500 tickers.csv')

start = date(2005, 1, 1)
end = date(2015, 1, 1)

# 顺序存储SP500的所有ticker
SP500_Tickers = []
SP500_csv = pd.read_csv('/Users/wuqianzhong/Desktop/NYU/Risk Management Maymin/SP500 tickers.csv', header=None)
for item in SP500_csv[0]:
    SP500_Tickers.append(item)

# 得到所有ticker的价量信息
'''
for name in SP500_Tickers:
    try:
        vars()[name] = data.DataReader(name, 'yahoo', start, end)  # vars()[name]存储该股票价量信息
        print(name)
    except RemoteDataError:
        print(name, "data unavailable")  # 若yahoo finance无相关数据
'''


# # # 新project # # #
#######################################################################
GSPC = data.DataReader('^GSPC', 'yahoo', start, end)
AAPL = data.DataReader('AAPL', 'yahoo', start, end)
GOOG = data.DataReader('GOOG', 'yahoo', start, end)


def VaR(value, CI, mu, sigma):
    return (norm.ppf(CI)*sigma-mu) * value  # VARIANCE_COVARIANCE APPROACH


def CVaR(value, CI, mu, sigma):
    return ((1-CI)**-1 * norm.pdf(norm.ppf(1-CI))*sigma-mu) * value
# VaR和CVaR公式来源 http://www.quantatrisk.com/2016/12/08/conditional-value-at-risk-normal-student-t-var-model-python/


# # # Tangency portfolio（maximize portfolio's sharp ratio）
AAPL['log_ret'] = np.log(AAPL['Adj Close'] / AAPL['Adj Close'].shift(1))
GOOG['log_ret'] = np.log(GOOG['Adj Close'] / GOOG['Adj Close'].shift(1))
AAPL['mov_mean'] = pd.rolling_mean(AAPL['log_ret'], window=252)
GOOG['mov_mean'] = pd.rolling_mean(GOOG['log_ret'], window=252)
AAPL['mov_vol'] = pd.rolling_std(AAPL['log_ret'], window=252)
GOOG['mov_vol'] = pd.rolling_std(GOOG['log_ret'], window=252)
AAPL['VaR'] = VaR(1, 0.95, AAPL['mov_mean'], AAPL['mov_vol'])
GOOG['VaR'] = VaR(1, 0.95, GOOG['mov_mean'], GOOG['mov_vol'])
Mov_Cov = pd.rolling_cov(AAPL['log_ret'], GOOG['log_ret'], window=252)
Mov_Corr = pd.rolling_corr(AAPL['log_ret'], GOOG['log_ret'], window=252)
AAPL['mov_var'] = pd.rolling_var(AAPL['log_ret'], window=252)
GOOG['mov_var'] = pd.rolling_var(GOOG['log_ret'], window=252)
Rf = 0.02 / 252  # risk free rate

# Tgcy_portfolio 权重公式来源  https://business.missouri.edu/yanx/fin333/lectures/Riskyportfolio%20short.pdf
Tgcy_wgt_AAPL = ((AAPL['log_ret'] - Rf) * GOOG['mov_var'] - (GOOG['log_ret'] - Rf) * Mov_Cov) / \
                ((AAPL['log_ret'] - Rf) * GOOG['mov_var'] - (GOOG['log_ret'] - Rf) * AAPL['mov_var'] - \
                 (AAPL['log_ret'] - Rf + GOOG['log_ret'] - Rf) * Mov_Cov)  # 计算两个资产权重的公式

Tgcy_wgt_AAPL[Tgcy_wgt_AAPL<0] = 0   # assume long only
Tgcy_wgt_AAPL[Tgcy_wgt_AAPL>1] = 1
Tgcy_wgt_GOOG = 1 - Tgcy_wgt_AAPL
Tgcy_portfolio_ret = AAPL['log_ret'] * Tgcy_wgt_AAPL + GOOG['log_ret'] * Tgcy_wgt_GOOG

Tgcy_portfolio_mean = pd.rolling_mean(Tgcy_portfolio_ret, window=252)
Tgcy_portfolio_std = pd.rolling_std(Tgcy_portfolio_ret, window=252)
Tgcy_portfolio_VaR = VaR(1, 0.95, Tgcy_portfolio_mean, Tgcy_portfolio_std)
Tgcy_portfolio_CVaR = CVaR(1, 0.95, Tgcy_portfolio_mean, Tgcy_portfolio_std)
Tgcy_portfolio = pd.DataFrame({'Return': Tgcy_portfolio_ret,
                              'VaR': Tgcy_portfolio_VaR,
                              'CVaR': Tgcy_portfolio_CVaR,
                              'Cum_Return': Tgcy_portfolio_ret.cumsum()})
Tgcy_portfolio = Tgcy_portfolio[['Return', 'Cum_Return', 'VaR', 'CVaR']]  # 重新排列按该顺序输出，默认是按字母顺序


# # # Equal weighted portfolio
Eql_portfolio_ret = AAPL['log_ret'] * 0.5 + GOOG['log_ret'] * 0.5

Eql_portfolio_mean = pd.rolling_mean(Eql_portfolio_ret, window=252)
Eql_portfolio_std = pd.rolling_std(Eql_portfolio_ret, window=252)
Eql_portfolio_VaR = VaR(1, 0.95, Eql_portfolio_mean, Eql_portfolio_std)
Eql_portfolio_CVaR = CVaR(1, 0.95, Eql_portfolio_mean, Eql_portfolio_std)
Eql_portfolio = pd.DataFrame({'Return': Eql_portfolio_ret,
                              'VaR': Eql_portfolio_VaR,
                              'CVaR': Eql_portfolio_CVaR,
                              'Cum_Return': Eql_portfolio_ret.cumsum()})
Eql_portfolio = Eql_portfolio[['Return', 'Cum_Return', 'VaR', 'CVaR']]

# # # Risk parity weighted portfolio
# article http://people.umass.edu/kazemi/An%20Introduction%20to%20Risk%20Parity.pdf
RskPrty_wgt_AAPL = GOOG['mov_vol'] / (GOOG['mov_vol']+AAPL['mov_vol'])
RskPrty_wgt_GOOG = 1 - RskPrty_wgt_AAPL
RskPrty_portfolio_ret = AAPL['log_ret'] * RskPrty_wgt_AAPL + GOOG['log_ret'] * RskPrty_wgt_GOOG

RskPrty_portfolio_mean = pd.rolling_mean(RskPrty_portfolio_ret, window=252)
RskPrty_portfolio_std = pd.rolling_std(RskPrty_portfolio_ret, window=252)
RskPrty_portfolio_VaR = VaR(1, 0.95, RskPrty_portfolio_mean, Eql_portfolio_std)

# VaRR 代表用variance/covariance method算出的组合VaR
RskPrty_portfolio_VaRR = np.sqrt(RskPrty_wgt_AAPL**2 * AAPL['VaR']**2 + RskPrty_wgt_GOOG**2 * GOOG['VaR']**2 \
                                 +2*RskPrty_wgt_AAPL*RskPrty_wgt_GOOG*AAPL['VaR']*GOOG['VaR']\
                                 *Mov_Corr)
RskPrty_portfolio_CVaR = CVaR(1, 0.95, RskPrty_portfolio_mean, Eql_portfolio_std)
RskPrty_portfolio = pd.DataFrame({'Return': RskPrty_portfolio_ret,
                              'VaR': RskPrty_portfolio_VaR,
                              'VaRR': RskPrty_portfolio_VaRR,
                              'CVaR': RskPrty_portfolio_CVaR,
                              'Cum_Return': RskPrty_portfolio_ret.cumsum()})
RskPrty_portfolio = RskPrty_portfolio[['Return', 'Cum_Return', 'VaR', 'VaRR', 'CVaR']]

print(RskPrty_portfolio)




