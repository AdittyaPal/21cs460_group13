import pandas as pd
import numpy as np
import zmq            #import the python wrapper package for the ZeroMQ library
import time
import random

#read the data that would be sent by the server over sockets
raw=pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
data=pd.DataFrame(raw[['time','close']][:8000])
#array whose elements would be sent
data_send=np.array(data['close'][5000:12000])

context = zmq.Context()            #instaniate a Context object for socket communication
socket = context.socket(zmq.PUB)   #define the socket to be of publisher type
socket.bind('tcp://0.0.0.0:5555')  #bind the socket to the local IP address(for Linux) with port number 5555

tickval=0
#loop to send the elements of the array sequentially
while tickval<7000:
    tick='{} {}'.format('close', data_send[tickval])    #generate the text value to be sent
    print(tickval)
    print(tick)                                         #print he mesage that would be sent
    socket.send_string(tick)                            #send the message via the socket created
    time.sleep(2+random.random())                       #pause the loop o simulate the random arrival of data
    tickval+=1                                          #update he ick value to be sent

