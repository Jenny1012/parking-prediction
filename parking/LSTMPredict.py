from stock.Workbook_Init import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import pandas as pd
#import pandas_datareader.data as web
from datetime import datetime, date, timedelta
import datetime as dt
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import os

Epoch = 25
Batch_size = 32
Units = 50 ##
predictionDays = 7
layers = 1
dropout = 0.2
val = 0.2

def LSTMModel(train, test):
    # Prepare Data
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled_data = scaler.fit_transform(train.values.reshape(-1, 1)) 

    x_train = []
    y_train = []

    for i in range(predictionDays, len(scaled_data)):
        x_train.append(scaled_data[i-predictionDays : i, 0]) 
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train) 
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build The Model
    model = Sequential()
    model.add(LSTM(units = Units, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(dropout))
    model.add(Dense(units = 1)) 
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_squared_error', 'mean_absolute_error'])
    history = model.fit(x_train, y_train, epochs = Epoch, batch_size = Batch_size, validation_split = val)
    #print(model.summary())

    # Test The Model Accuracy on Existing Data
    # Load Test Data
    actual_prices = test.values
    total_dataset = pd.concat((train, test), axis = 0)
    model_inputs = total_dataset[len(total_dataset) - len(test) - predictionDays:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Make Prediction on Test Data
    x_test = [] 

    for i in range(predictionDays, len(model_inputs)):
        x_test.append(model_inputs[i-predictionDays: i, 0])

    x_test = np.array(x_test)
    y_test = x_test
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_occupancy = model.predict(x_test)
    predicted_occupancy_1 = scaler.inverse_transform(predicted_occupancy)
    
    MSE = str(mean_squared_error(actual_prices, predicted_occupancy_1))
    #print("Mean Squared Error : " + MSE)
    MAE = str(mean_absolute_error(actual_prices, predicted_occupancy_1))
    #print("Mean Absolute Error : " + MAE)

    return MAE, MSE, predicted_occupancy_1