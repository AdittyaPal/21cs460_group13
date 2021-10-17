import pandas as pd
from pylab import mpl, plt

fullData1=pd.read_csv('./One_Month_Data.csv')
fullData=pd.read_csv('./OneMonthData.csv')
fullData2=fullData.set_index('time')
data_plot1=pd.DataFrame(fullData1['askhigh'])
data_plot2=pd.DataFrame(fullData2['high'])
#PLOT
# convert time in seconds into the datetime format
#plot_data['time']=pd.to_datetime(plot_data['time'])
# display ticks on the chart

plt.style.use('seaborn')
mpl.rcParams['savefig.dpi']=300
mpl.rcParams['font.family']='serif'
data_plot1.plot(title='EUR/USD', figsize=(20, 12), linewidth=0.5)
data_plot2.plot(title='EUR/USD', figsize=(20, 12), linewidth=0.5)
plt.show()

