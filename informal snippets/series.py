import pandas as pd
import numpy as np
from pylab import mpl, plt
from statsmodels.graphics.tsaplots import plot_acf

raw=pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
data=pd.DataFrame(raw[['time','close']][0:1000])
data['mean1']=data['close'].rolling(window=10).mean()
data['logs']=np.log(data['close'])
data['mean2']=data['logs'].rolling(window=10).mean()
data['returns']=np.log(data['close']/data['close'].shift(1))
data['mean3']=data['returns'].rolling(window=10).mean()
data['std3']=data['returns'].rolling(window=10).std()
data.dropna(inplace=True)

plt.style.use('seaborn')
mpl.rcParams['savefig.dpi']=300
mpl.rcParams['font.family']='serif'

fig, axes=plt.subplots(nrows=4, ncols=1, figsize=(10, 20))

data[['close', 'mean1']].dropna().plot(ax=axes[0], title='Raw Time Series')
data[['logs', 'mean2']].dropna().plot(ax=axes[1], title='Log Time Series')
data[['returns', 'mean3', 'std3']].dropna().plot(ax=axes[2], title='Log Difference')
plot_acf(data['returns'], ax=axes[3], lags=40)
plt.show()
