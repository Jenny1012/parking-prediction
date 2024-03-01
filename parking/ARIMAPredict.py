# Import libraries and custom functions defined in Workbook_Init.py
from Workbook_Init import *
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
from numpy import absolute
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from LSTMPredict import *
import LSTMPredict
from XGBoostPredict import *
import XGBoostPredict

def ARIMAModel(train, test):
    arima_model = ARIMA(train, order=(18, 1, 0))
    result_AR = arima_model.fit()
    y_pred_AR = result_AR.forecast(steps=len(test))
    y_pred_AR.index = test.index
    #y_pred_AR = y_pred_AR.fillna(0)
    """print('-'*77)
    print('ARIMA Model Metrics on Test Data - ', Roadname)
    print('='*77)"""
    ExplainedVariance, mae = report_metrics(test.squeeze(), y_pred_AR.squeeze())

    return ExplainedVariance, mae, y_pred_AR