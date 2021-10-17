
#from pylab import mpl, plt
import pandas as pd

data=pd.read_csv('OneMonthData.csv')
#data.to_csv('./OneMonthData.csv')

print(data.info())
'''
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi']=300
mpl.rcParams['font.family']='serif'
data.plot(title='EUR/USD', figsize=(20, 12))
plt.show()
'''