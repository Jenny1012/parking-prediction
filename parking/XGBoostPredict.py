from stock.Workbook_Init import *
import numpy as np
from numpy import absolute
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime, date, timedelta
import datetime as dt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from xgboost import XGBRegressor
import os
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA


predictionDays = 7
n_estimators = 100
predictionDays = 90
learning_rate = 0.05
early_stopping_rounds = 5

def XGBModel(train, test):
    # Prepare Data
    scaled_data = train.values.reshape(-1, 1)

    x_train = []
    y_train = []

    for i in range(predictionDays, len(scaled_data)):
        x_train.append(scaled_data[i-predictionDays : i, 0]) 
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train) 

    # Test The Model Accuracy on Existing Data
    # Load Test Data
    actual_prices = test.values
    total_dataset = pd.concat((train, test), axis = 0)
    model_inputs = total_dataset[len(total_dataset) - len(test) - predictionDays:].values
    model_inputs = model_inputs.reshape(-1, 1)

    # Make Prediction on Test Data
    x_test = [] 
    y_test = []
    for i in range(predictionDays, len(model_inputs)):
        x_test.append(model_inputs[i-predictionDays:i, 0])
        y_test.append(model_inputs[i, 0])

    # Build The Model
    eval_set = [(x_test, y_test)]
    model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate = learning_rate)
    #model.fit(x_train, y_train, eval_metric=mean_squared_error)
    model.fit(x_train, y_train, eval_metric=mean_absolute_error)

    x_test = np.array(x_test)
    predicted_occupancy = model.predict(x_test)
    MSE = str(mean_squared_error(predicted_occupancy, y_test))
    #print("Mean Squared Error : " + MSE)
    MAE = str(mean_absolute_error(predicted_occupancy, y_test))
    #print("Mean Absolute Error : " + MAE)

    return MAE, MSE, predicted_occupancy