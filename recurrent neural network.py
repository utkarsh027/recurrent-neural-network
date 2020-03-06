#data preprocessing
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training training set
dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=dataset_train.iloc[:,1:2].values#that craeat numpy array

#featurpree scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)

#creating a data structure with 60 timesteps and 1 output
#looks at 60 before and 60 after of time t

x_train=[]
y_train=[]
for i in range (60,1258):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)

#reshaping
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#part2
#building the rnn

#importing the keras libraries and packages
from keras.models import Sequential
from keras .layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initialising the rnn
regressor=Sequential()

#Adding the first LSTM layer and some dropout regularisation
#we add drpout layer to avoidoverfitting
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))

#adding a second LSTm layer and some drpoout regularization
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

#adding a third LSTM layer and some dropout regulrization
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

#adding afourth LSTM layer and some dropout regularization
#since its the last layer we are not going to return any sequence so return_sequences=false
regressor.add(LSTM(units=50))
regressor.add(Dropout=(0.2))

#adding the output layer
regressor.add(Dense(units=1))

#compiling the rnn
regressor.compile(optimizer='adam',loss='mean_squared_error')
#making the prediction and visualising the result

#fitting the rnn to the training set
regressor.fit(x_train,y_train,epochs=100,batch_size=32)
