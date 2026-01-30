import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf

df = yf.download('NFLX', start='2015-01-01')
print(df.head())

##Time Series Plot of Netflix Closing Stock Prices

plt.figure(figsize=(12, 6))
plt.plot(df['Close'], color='red')
plt.title('Netflix Stock Price History')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show(block=False) 
plt.pause(3)

## Data Preprocessing Pipeline for Time Series Forecasting

# 1. Handling Missing Values (Fill any gaps with the previous day's price)
df = df.fillna(method='ffill')

# 2. Selecting only the 'Close' price for our prediction
data = df[['Close']].values

# 3. Scaling: Shrink prices to be between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 4. Time-Based Split: Use 80% for training, 20% for testing
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

print(f"Total days: {len(scaled_data)}")
print(f"Training days: {len(train_data)}")
print(f"Testing days: {len(test_data)}")

## Training the dataset and Predicting with SARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. Defining the model
# (1,1,1) are standard starting parameters for Trend
# (1,1,1,5) assumes a small weekly seasonal pattern (5 trading days)
model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))

# 2. Training the model
model_fit = model.fit(disp=False)

# 3. Making a prediction for the 'test_data' period
predictions = model_fit.forecast(steps=len(test_data))

# 4. Calculating Evaluation Metrics
mae = mean_absolute_error(test_data, predictions)
rmse = np.sqrt(mean_squared_error(test_data, predictions))

print(f"SARIMA MAE: {mae:.4f}")
print(f"SARIMA RMSE: {rmse:.4f}")

##Preparing Sequences and Building the LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Creating Windows (Sequences)
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Creating sequences for training
X_train, y_train = create_sequences(train_data)
# Reshape to 3D: [Batch Size, Time Steps, 1 Feature]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# 2. Building the LSTM Model
model_lstm = Sequential([
    LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)),
    Dense(units=1) # The output: 1 single number (tomorrow's price)
])

# 3. Compile and Train
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

X_test, y_test = create_sequences(test_data)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

lstm_predictions = model_lstm.predict(X_test)

##Creating "Lag" and "Time"

# 1. Creating a Lag Feature (Yesterday's Price)
# This helps the model see what happened exactly one day ago.
df['Lag_1'] = df['Close'].shift(1)

# 2. Creating Time-Based Features
# This tells the model if it is a Monday (0) or Friday (4).
df['DayOfWeek'] = df.index.dayofweek

# 3. Droping the very first row
# Because 'Yesterday' doesn't exist for the first day, it will be empty (NaN).
# We must remove it so the models don't crash.
df.dropna(inplace=True)

print("Feature Engineering Complete!")
print(df[['Close', 'Lag_1', 'DayOfWeek']].head())

## Comparative performance analysis using tables of RMSE, MAE, and MAPE.

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. Function to calculate MAPE
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 2. Assume 'test_data' is our truth and we have predictions

# For SARIMA:
mae_sarima = mean_absolute_error(test_data, predictions)
rmse_sarima = np.sqrt(mean_squared_error(test_data, predictions))
mape_sarima = calculate_mape(test_data, predictions)

print(f"SARIMA - MAE: {mae_sarima:.2f}, RMSE: {rmse_sarima:.2f}, MAPE: {mape_sarima:.2f}%")

##Attention-Based Model
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class SimpleAttention(Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(SimpleAttention, self).build(input_shape)

    def call(self, x):
        
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et) 
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1) 
    

from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model

# 1. Defining the Architecture
inputs = Input(shape=(X_train.shape[1], 1))

lstm_out = LSTM(units=64, return_sequences=True)(inputs)

attention_out = SimpleAttention()(lstm_out)

outputs = Dense(1)(attention_out)

model_attention = Model(inputs, outputs)

# 2. Compiling and Training
model_attention.compile(optimizer='adam', loss='mean_squared_error')
print("Training Advanced Attention Model...")
model_attention.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

## Manual "Tuning" Approach
neuron_options = [32, 64]
best_mae = float('inf')

for n in neuron_options:
    print(f"Testing model with {n} neurons...")
    
att_predictions = model_attention.predict(X_test)
mae_att = mean_absolute_error(y_test, att_predictions)
rmse_att = np.sqrt(mean_squared_error(y_test, att_predictions))
mape_att = calculate_mape(y_test, att_predictions)

X_test, y_test = create_sequences(test_data)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 2. Get the model's guesses
lstm_predictions = model_lstm.predict(X_test)
att_val_predictions = model_attention.predict(X_test)

# 3. Define the missing variables that the error is complaining about
mae_lstm = mean_absolute_error(y_test, lstm_predictions)
rmse_lstm = np.sqrt(mean_squared_error(y_test, lstm_predictions))
mape_lstm = calculate_mape(y_test, lstm_predictions)

print("LSTM Metrics defined successfully!")

## Creating a comparison table
print("\n" + "="*45)
print("FINAL PERFORMANCE COMPARISON TABLE")
print("="*45)
# Using the variables we just defined above
results = [
    ["SARIMA (Baseline)", mae_sarima, rmse_sarima, mape_sarima],
    ["Simple LSTM", mae_lstm, rmse_lstm, mape_lstm],
    ["Attention Model", mae_att, rmse_att, mape_att]
]

print(f"{'Model':<20} | {'MAE':<8} | {'RMSE':<8} | {'MAPE':<8}")
print("-" * 55)
for m, mae_v, rmse_v, mape_v in results:
    print(f"{m:<20} | {mae_v:.4f} | {rmse_v:.4f} | {mape_v:.2f}%")