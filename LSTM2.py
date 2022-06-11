import pandas as pd
import numpy as np
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



df = pd.read_csv('Electric_Production.csv')
df.columns = ['date','value']
df['date'] = pd.to_datetime(df['date'],infer_datetime_format=True)
df = df.set_index(['date'])

"""
The first thing we must do is to properly shape the input data. 
When modeling a time series by LSTM RNN, it is crucial to to properly shape the 
input data in a sliding windows format. In this application, 
the data is given as monthly data. So, for example, 
we can use a 12 steps prediction window. This means that we use 12 samples of 
data (data of an entire year) to predict the 13th sample.
"""
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
prediction_window=12

n_input=12 #Use 12 months data to predict 13 th month data
n_features=1 # we are dealing with an univariate time series, so n_features should be set to 1.
#In case of a multivariate time series, n_features should be set to a proper value higher than 1.


train = df.copy()


"""
########################
## rescale data
########################
"""
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)


"""
Now we can formally create the train set. As stated before, the train set will have 
a "sliding window" shape. This means that we have to shape the data in such a way that
 the RNN will predict the 13th sample starting from the previous from 12 samples.
"""
def sliding_windows(data, n_input):
    X_train=[]
    y_train=[]
    for i in range(n_input,len(data)):
        X_train.append(data[i-n_input:i])
        y_train.append(data[i])
    return np.array(X_train), np.array(y_train)


x, y = sliding_windows(scaled_train, prediction_window)

print(f'Given the Array: \n {x[0].flatten()}')
print(f'Predict this value: \n {y[0]}')


train_size = int(len(train) - prediction_window*3)
val_size = len(train) - train_size

"""
########################
## convert to tensor
########################
"""
dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

X_train = Variable(torch.Tensor(np.array(x[:train_size])))
y_train = Variable(torch.Tensor(np.array(y[:train_size])))

X_valid = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
y_valid = Variable(torch.Tensor(np.array(y[train_size:len(y)])))


# dataX.to(device)
# dataY.to(device)
# X_train.to(device)
# y_train.to(device)
# X_valid.to(device)
# y_valid.to(device)

"""
########################
## LSTM Modelling
########################
"""


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, 40)
        self.fc2 = nn.Linear(40, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        _, (h_out, _) = self.lstm(x, (h0, c0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc2(self.relu(self.fc1(h_out)))

        return out


"""
########################
## Parameters for training
########################
"""

EPOCHS = 2000
LEARNING_RATE = 0.008

INPUT_SIZE = n_features
HIDDEN_SIZE = 100
NUM_LAYERS = 1

model = LSTMNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
model.to(device)
print(model)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

"""
########################
## training 
########################
"""

early_stopping_patience = 150
early_stopping_counter = 0

valid_loss_min = np.inf

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    model.train()
    output = model(X_train)

    train_loss = criterion(output, y_train)

    train_loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        output_val = model(X_valid)
        valid_loss = criterion(output_val, y_valid)

        if valid_loss <= valid_loss_min:
            torch.save(model.state_dict(), './state_dict.pt')
            print(
                f'Epoch {epoch + 0:01}: Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ...')
            valid_loss_min = valid_loss
            early_stopping_counter = 0  # reset counter if validation loss decreases
        else:
            print(f'Epoch {epoch + 0:01}: Validation loss did not decrease')
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_patience:
            print('Early stopped at epoch :', epoch)
            break

        print(f'\t Train_Loss: {train_loss:.4f} Val_Loss: {valid_loss:.4f}  BEST VAL Loss: {valid_loss_min:.4f}\n')

"""
########################
## LSTM Prediction
########################
"""

# Loading the best model
model.load_state_dict(torch.load('./state_dict.pt'))

valid_predict = model(X_valid)
y_pred_scaled = valid_predict.data.numpy()
y_pred = scaler.inverse_transform(y_pred_scaled)

df_pred = train.iloc[-24:]
df_pred['prediction'] = y_pred

# mape_lstm = mape(df_pred["value"], df_pred["prediction"])
# print(f"MAPE OF LSTM MODEL : {mape_lstm:.2f} %")

rmse_lstm = mean_squared_error(df_pred["value"], df_pred["prediction"], squared=False)
print(f"RMSE OF LSTM MODEL : {rmse_lstm:.2f}")


"""
    ########################
    ## LSTM Forecasting
    ########################
"""

test_predictions = []

first_eval_batch = Variable(torch.Tensor(scaled_train[-n_input:]))  # use the previous 12 samples to predict the 13th
current_batch = first_eval_batch.reshape((1, n_input, n_features))  # reshape the data into (1,12,1)
for i in range(len(scaled_train[-n_input:])):
    # get the prediction value for the first batch
    current_pred = model(current_batch)
    # append the prediction into the array
    test_predictions.append(current_pred)

    # use the prediction to update the batch and remove the first value
    current_batch = torch.cat((current_batch[:, 1:, :], current_pred.reshape(1, 1, 1)), 1)

forec_vals = [val.flatten().item() for val in test_predictions]
forec_vals = np.array(forec_vals).reshape(-1,1)
forec_vals = scaler.inverse_transform(forec_vals)

date_offset=12
forecast_dates =  (train.index + pd.DateOffset(months=date_offset))[-date_offset:]
forecast_dates


df_forecast=pd.DataFrame({'date': forecast_dates})
df_forecast.set_index('date', inplace=True)
df_forecast['prediction'] = forec_vals
df_forecast.head(12)

df_full=df_pred.append(df_forecast)