import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
import seaborn as sns
##https://www.kaggle.com/code/ludovicocuoghi/electric-production-forecast-lstm-sarima-mape-2-5
import torch
import torch.nn as nn
from torch.autograd import Variable

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


from math import sqrt

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlesize=14, titlepad=10)

from fbprophet import Prophet

df = pd.read_csv('Electric_Production.csv')
df.columns = ['date','value']
df['date'] = pd.to_datetime(df['date'],infer_datetime_format=True)
df = df.set_index(['date'])



df_pr = df.copy()
df_pr = df.reset_index()

df_pr.columns = ['ds','y'] # To use prophet column names should be like that

train_data_pr = df_pr.iloc[:len(df)-12]
test_data_pr = df_pr.iloc[len(df)-12:]


m = Prophet()
m.fit(train_data_pr)
future = m.make_future_dataframe(periods=12,freq='D')
prophet_pred = m.predict(future)


prophet_pred.tail()

prophet_pred = pd.DataFrame({"Date" : prophet_pred[-12:]['ds'], "Pred" : prophet_pred[-12:]["yhat"]})

prophet_pred = prophet_pred.set_index("Date")

prophet_pred.index.freq = "D"

test_data_pr["Prophet_Predictions"] = prophet_pred['Pred'].values


plt.figure(figsize=(16,5))
ax = sns.lineplot(x= test_data_pr.index, y=test_data_pr["y"])
sns.lineplot(x=test_data_pr.index, y = test_data_pr["Prophet_Predictions"]);

from statsmodels.tools.eval_measures import rmse
prophet_rmse_error = rmse(test_data_pr['y'], test_data_pr["Prophet_Predictions"])
prophet_mse_error = prophet_rmse_error**2
mean_value = df['value'].mean()

print(f'MSE Error: {prophet_mse_error}\nRMSE Error: {prophet_rmse_error}\nMean: {mean_value}')