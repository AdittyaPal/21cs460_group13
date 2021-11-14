import pandas as pd
import numpy as np
from pylab import mpl, plt
import pmdarima as pm

raw=pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
data=pd.DataFrame(raw[['time','close']][0:3000])
data['logs']=np.log(data['close'])
data['returns']=np.log(data['close']/data['close'].shift(1))
data.dropna(inplace=True)

plt.style.use('seaborn')
mpl.rcParams['savefig.dpi']=300
mpl.rcParams['font.family']='serif'

model=pm.auto_arima(data['logs'][0:2000], start_p=0, strat_q=0, start_P=0, start_Q=0, d=0, max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False)
print(model.summary())

end=2000
predict=np.array([])

while(end<2470):
	model.fit(data['logs'][end-500:end])
	predict=np.append(predict, model.predict(n_periods=30, return_conf_int=False))
	end+=10

end=2000
index=0
abscissa=np.arange(2000, 2500)
plt.plot(abscissa, data['logs'][2000:2500], c='g')	
while(end<2470):
	plt.plot(abscissa[end-2000:end+30-2000], predict[index:index+30], c='b')
	end+=10
	index+=30

plt.show()

