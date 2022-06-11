import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('train_E1GspfA.csv')

###group by date, rather than hourly
df_2 = df.groupby(['date'])['demand'].sum().reset_index()

##plot the time seres

df_2.set_index('date',inplace=True)


# df_2.plot()
# plt.show()

###check for stationality
from statsmodels.tsa.stattools import adfuller
def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")

adfuller_test(df_2['demand'])