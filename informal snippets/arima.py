import pandas as pd
import numpy as np
from pylab import mpl, plt
import pmdarima as pm

raw=pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
data=pd.DataFrame(raw[['time','close']][0:5000])
data['logs']=np.log(data['close'])
data['returns']=np.log(data['close']/data['close'].shift(1))
data.dropna(inplace=True)

plt.style.use('seaborn')
mpl.rcParams['savefig.dpi']=300
mpl.rcParams['font.family']='serif'

model=pm.auto_arima(data['logs'], start_p=0, strat_q=0, start_P=0, start_Q=0, d=0, max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False)

model.plot_diagnostics(figsize=(16, 12))
plt.show()

'''
end=2000
predict=np.zeros(99)

while(end<2099):
	model.fit(data['logs'][0:end])
	predict[end-2000]=model.predict(n_periods=1, return_conf_int=False)
	#print('Actual: %.8f, Predicted: %.8f'%(data['logs'][end+1], predict[end-200]))
	end+=1
	
predict=np.flip(predict)

plt.plot(np.arange(99), predict)
plt.plot(np.arange(99), data['logs'][2000:])
#plt.fill_between(np.arange(50), conf_int[:, 0], conf_int[:, 1], alpha=0.1, color='b')
plt.show()

#validate=pm.model_selection.SlidingWindowForecastCV(window_size=5)
#preds=pm.model_selection.cross_val_predict(model, data['logs'], cv=validate, averaging='median')
preds, conf_int=model.predict_in_sample(start=100, end=1999, dynamic=False, return_conf_int=True)
data['predict']=np.concatenate((data['logs'][0:99], preds))
data['predictReturns']=data['predict']-data['predict'].shift(1)
data['strategy']=np.sign(data['predictReturns'])*data['returns']
print(data[['returns', 'strategy']].sum().apply(np.exp))
data[['returns', 'strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()
'''

