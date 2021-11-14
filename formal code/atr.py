'''
python script to calculate the average true returns 
for a time series data
'''
import pandas as pd
import numpy as np
from pylab import mpl, plt

#formatting of the plot
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi']=300
mpl.rcParams['font.family']='serif'

#read the data
raw=pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
data=pd.DataFrame(raw[['time','close']][0:5000])
window=30
data['min']=data['close'].rolling(window).min()           #rolling minimum
data['max']=data['close'].rolling(window).max()           #rolling maximum
data['range']=data['max']-data['min']                     #difference between the rolling maximum and minimum
data['maxToday']=abs(data['max']-data['close'].shift(1))  #difference between rolling maximum and previous day's price
data['minToday']=abs(data['min']-data['close'].shift(1))  #difference between rolling minimum and previous day's price
data['atr']=np.maximum(data['range'], data['maxToday'])   #calculate the maximum of the max-min range and max-price difference
data['atr']=np.maximum(data['atr'], data['minToday'])     #and min-price difference
data['atr%']=data['atr']/data['close']                    #determine the relaive returns (scaled by closing prices)
data[['atr', 'atr%']].plot(subplots=True, figsize=(10, 6))#plot the absolute and relaive true returns
plt.show()
