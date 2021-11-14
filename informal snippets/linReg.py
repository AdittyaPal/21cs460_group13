import pandas as pd
import numpy as np
from pylab import mpl, plt
raw=pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
data=pd.DataFrame(raw['close'][0:10000])
lags=5
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi']=300
mpl.rcParams['font.family']='serif'

dataLags=np.lib.stride_tricks.sliding_window_view(data['close'], lags)[:-1,:]
regParams=np.linalg.lstsq(dataLags, data['close'][5:], rcond=None)[0]
#print(regParams)

data.loc[:,'predict']=pd.Series(np.dot(dataLags, regParams)).shift(lags)
data.dropna(inplace=True)
data['returns']=np.log(data['close']/data['close'].shift(1))
data['logPredict']=np.log(data['predict']/data['predict'].shift(1))
data['strategy']=np.sign(data['logPredict'])*data['returns']
print(data[['returns', 'strategy']].sum().apply(np.exp))
data[['returns', 'strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()
#data[['close', 'predict']][0:200].plot(figsize=(10, 6))
#plt.show()

data['returns']=np.log(data['close']/data['close'].shift(1))
data.dropna(inplace=True)
dataLagReturn=np.lib.stride_tricks.sliding_window_view(data['returns'], lags)[:-1,:]
'''
regReturns=np.linalg.lstsq(dataLagReturn, data['returns'][5:], rcond=None)[0]

data.loc[:,'logPredict']=pd.Series(np.dot(dataLagReturn, regReturns)).shift(lags)
#data[['returns', 'logPredict']].plot(figsize=(10, 6))
#plt.show()
hits=np.sign(data['returns']*data['logPredict']).value_counts()
print(hits)
'''

regReturns=np.linalg.lstsq(dataLagReturn, np.sign(data['returns'][5:]), rcond=None)[0]
#print(regReturns)
data.loc[:,'signPredict']=pd.Series(np.sign(np.dot(dataLagReturn, regReturns))).shift(lags)
hits=np.sign(data['returns']*data['signPredict']).value_counts()
print(hits)
'''
data['strategy1']=np.sign(data['logPredict'])*data['returns']
data['strategy2']=data['signPredict']*data['returns']
print(data[['returns', 'strategy1', 'strategy2']].sum().apply(np.exp))
data[['returns', 'strategy1', 'strategy2']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()
'''
