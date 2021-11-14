import pandas as pd
import numpy as np
from pylab import mpl, plt
raw=pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
print(raw.info())

data=pd.DataFrame(raw['close'][0:5000])
data['SMA1']=data['close'].rolling(50).mean()
data['SMA2']=data['close'].rolling(300).mean()
#print(data.tail())
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi']=300
mpl.rcParams['font.family']='serif'
#data.plot(title='EUR/USD | 50 and 300 mins SMAs', figsize=(12, 8))
data['position']=np.where(data['SMA1']>data['SMA2'], 1, -1)
data.dropna(inplace=True)
#data['position'].plot(ylim=[-1.1, 1.1], title='Go Long vs. Go Short', figsize=(10, 6))
data['returns']=np.log(data['close']/data['close'].shift(1))
#data['returns'].hist(bins=100, figsize=(10, 6))
data['perform']=data['position'].shift(1)*data['returns']
#print(data[['returns', 'perform']].sum())
#print(data[['returns', 'perform']].sum().apply(np.exp))
#data[['returns', 'perform']].cumsum().apply(np.exp).plot(figsize=(10, 6))
print(data[['returns', 'perform']].mean()*300)
print(np.exp(data[['returns', 'perform']].mean()*300)-1)
print(data[['returns', 'perform']].std()*300**0.5)
print((data[['returns', 'perform']].apply(np.exp)-1).std()*300**0.5)
data['cumReturn']=data['perform'].cumsum().apply(np.exp)
data['cumMax']=data['cumReturn'].cummax()
data[['cumReturn', 'cumMax']].dropna().plot(figsize=(10, 6))
drawdown=data['cumMax']-data['cumReturn']
print(drawdown.max())
temp=drawdown[drawdown==0]
periods=(temp.index[1:]-temp.index[:-1])
print(periods.max())
plt.show()

