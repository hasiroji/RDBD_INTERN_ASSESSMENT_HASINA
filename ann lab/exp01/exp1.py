import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(r"c:\Users\M R Computer\OneDrive\Desktop\ann lab\exp01\stock_1.csv")

# Use Stock_1 column
data = df["Stock_1"].values.reshape(-1, 1)

# ==========================
# Data Normalization
# ==========================

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)


# ==========================
# Create Sequences
# ==========================

time_step = 5

X = []
y = []

for i in range(time_step, len(data_scaled)):
    X.append(data_scaled[i-time_step:i, 0])
    y.append(data_scaled[i, 0])

X = np.array(X)
y = np.array(y)

# Reshape for RNN
X = X.reshape((X.shape[0], X.shape[1], 1))

print("Input Shape :", X.shape)
print("Output Shape:", y.shape)

# ==========================
# Train-Test Split
# ==========================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# ==========================
# Build RNN Model
# ==========================
from tensorflow.keras import Input

model = Sequential()

model.add(Input(shape=(X_train.shape[1], 1)))
model.add(SimpleRNN(50, activation='tanh'))
model.add(Dense(1))

model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# ==========================
# Train Model
# ==========================

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=4,
    verbose=0
)

# ==========================
# Prediction
# ==========================

predicted = model.predict(X_test)

predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# ==========================
# Plot Result
# ==========================

plt.figure(figsize=(10,5))

plt.plot(actual,
         label="Actual Price")

plt.plot(predicted,
         label="Predicted Price")

plt.title("Stock Price Prediction using RNN")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)

plt.show()

# ==========================
# Next Day Prediction
# ==========================

last_5_days = data_scaled[-5:]
last_5_days = last_5_days.reshape(1, 5, 1)

next_price = model.predict(last_5_days)

next_price = scaler.inverse_transform(next_price)

print("\nPredicted Next Day Stock Price:")
print(next_price[0][0])