import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# ====================================
# Load Dataset
# ====================================

df = pd.read_csv(r'C:\Users\M R Computer\OneDrive\Desktop\ann lab\exp02\rainfall.csv')

# Rainfall column
data = df["Rainfall"].values.reshape(-1, 1)

# ====================================
# Normalize Data
# ====================================

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# ====================================
# Create Sequences
# Previous 12 months -> Next month
# ====================================

time_step = 6

X = []
y = []

for i in range(time_step, len(data_scaled)):
    X.append(data_scaled[i-time_step:i, 0])
    y.append(data_scaled[i, 0])

X = np.array(X)
y = np.array(y)

# Reshape for RNN
X = X.reshape(X.shape[0], X.shape[1], 1)

print("X Shape:", X.shape)
print("Y Shape:", y.shape)

# ====================================
# Build RNN Model
# ====================================

model = Sequential()

model.add(
    SimpleRNN(
        units=32,
        activation='tanh',
        input_shape=(X.shape[1], 1)
    )
)

model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# ====================================
# Train Model
# ====================================

history = model.fit(
    X,
    y,
    epochs=100,
    batch_size=2,
    verbose=1
)

# ====================================
# Predict Rainfall
# ====================================

predicted = model.predict(X)

predicted = scaler.inverse_transform(predicted)

actual = scaler.inverse_transform(
    y.reshape(-1, 1)
)

# ====================================
# Display Actual vs Predicted
# ====================================

print("\nActual vs Predicted Rainfall\n")

for a, p in zip(actual, predicted):
    print(
        f"Actual: {a[0]:.2f} mm\tPredicted: {p[0]:.2f} mm"
    )

# ====================================
# Rainfall Prediction Graph
# ====================================

plt.figure(figsize=(10, 5))

plt.plot(
    actual,
    marker='o',
    label='Actual Rainfall'
)

plt.plot(
    predicted,
    marker='x',
    label='Predicted Rainfall'
)

plt.title("Monthly Rainfall Prediction using RNN")
plt.xlabel("Month")
plt.ylabel("Rainfall (mm)")
plt.legend()
plt.grid(True)

plt.show()

# ====================================
# Training Loss Graph
# ====================================

plt.figure(figsize=(8, 5))

plt.plot(history.history['loss'])

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.grid(True)

plt.show()

# ====================================
# Predict Next Month Rainfall
# ====================================

last_12_months = data_scaled[-12:]
last_12_months = last_12_months.reshape(1, 12, 1)

next_month = model.predict(last_12_months)

next_month = scaler.inverse_transform(
    next_month
)

print(
    "\nPredicted Next Month Rainfall:",
    round(next_month[0][0], 2),
    "mm"
)