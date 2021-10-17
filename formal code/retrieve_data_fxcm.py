import fxcmpy
import datetime as dt
from pylab import mpl, plt
import pandas as pd
 
TOKEN='#############################'
con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='debug', log_file='./logFile.txt', server='demo')
print(con.get_instruments())
start=dt.datetime(2021, 8, 11)
end=dt.datetime(2021, 8, 21)
data=con.get_candles('EUR/USD', period='m1', start=start, end=end)
print("Received")
print(data.head())
print(data.tail())
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi']=500
mpl.rcParams['font.family']='serif'
data['askclose'].plot(title='EUR/USD', figsize=(20, 12));
plt.show()
con.close()

data.to_pickle('./data2.pkl')
extract=pd.read_pickle('./data2.pkl')




