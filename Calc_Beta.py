import pandas as pd
from pandas_datareader import data
from datetime import date
import numpy as np

# Grab time series data for 10-year history for AAPL and for S&P500
start = date(2006, 1, 1)
end = date(2016, 1, 1)
GSPC = data.DataReader('^GSPC', 'yahoo', start, end)
AAPL = data.DataReader('AAPL', 'yahoo', start, end)

# pd.DataFrame usage example
'''
df2 =pd.DataFrame({'rol1' : [1, 2, 3],
                  'rol2' : [4, 5, 6]},
                  index = ['a', 'b', 'c'])
print(df2)

   rol1  rol2
a     1     4
b     2     5
c     3     6
'''

# compute log returns
df = pd.DataFrame({'GSPC_adjclose': GSPC['Adj Close'],
                   'AAPL_adjclose': AAPL['Adj Close']},
                  index = GSPC.index)

df[['AAPL_log_ret', 'GSPC_log_ret']] =np.log(df[['AAPL_adjclose', 'GSPC_adjclose']] /
                                             df[['AAPL_adjclose', 'GSPC_adjclose']].shift(1))
df = df.dropna()

#compute stock beta alpha
covmat = np.cov(df['AAPL_log_ret'], df['GSPC_log_ret'])
beta = covmat[0, 1] / covmat[1, 1]
alpha = np.mean(df['AAPL_log_ret']) - beta * np.mean(df['GSPC_log_ret'])

#compute r_squared
ypred = alpha + beta * df['GSPC_log_ret']
SS_res = np.sum(np.power(ypred - df['AAPL_log_ret'], 2))
SS_tol = covmat[0, 0] * len(df -1)
r_squared = 1. - SS_res / SS_tol

#daily volatility
volatility = np.sqrt(covmat[0, 0])

#annualized the numbers
prd = 252 # used daily returns; 252 days to annualize
alpha = alpha * prd
volatility = volatility * np.sqrt(prd)

print(df['AAPL_log_ret'][0:5])

