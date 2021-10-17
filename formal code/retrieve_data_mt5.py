from datetime import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import MetaTrader5 as mt5
# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)
 
# import the 'pandas' module for displaying data obtained in the tabular form
import pandas as pd
pd.set_option('display.max_columns', 500) # number of columns to be displayed
pd.set_option('display.width', 1500)      # max table width to display
# import pytz module for working with time zone
import pytz
 
# establish connection to MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()
 
# now connect to another trading account specifying the password
account = #########
authorized = mt5.login(account, password="#########", server="MetaQuotes-Demo")
if authorized:
    # set time zone to UTC
	timezone = pytz.timezone("Etc/UTC")
	# create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
	utc_from = datetime(2021, 8, 31, tzinfo=timezone)
	utc_to = datetime(2021, 9, 30, tzinfo=timezone)
	# get bars from EURUSD M5 within the interval of 2020.01.10 00:00 - 2020.01.11 13:00 in UTC time zone
	rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M1, utc_from, utc_to)
else:
    print("failed to connect at account #{}, error code: {}".format(account, mt5.last_error()))
 
# shut down connection to the MetaTrader 5 terminal
mt5.shutdown()
 
# display each element of obtained data in a new line
print("Display obtained data 'as is'")
counter=0
for rate in rates:
    counter+=1
    if counter<=10:
        print(rate)
 
# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(rates)
# convert time in seconds into the 'datetime' format
rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
 
# display data
print("\nDisplay dataframe with data")
print(rates_frame.head(10))

rates_frame.to_pickle('./data.pkl')
extracted=pd.read_pickle('./data.pkl')

#PLOT
# display ticks on the chart
plt.plot(extracted['time'], extracted['high'], 'r-', label='high')
plt.plot(extracted['time'], extracted['low'], 'b-', label='low')
 
# display the legends
plt.legend(loc='upper left')
 
# add the header
plt.title('EURUSD ticks')
 
# display the chart
plt.show()
