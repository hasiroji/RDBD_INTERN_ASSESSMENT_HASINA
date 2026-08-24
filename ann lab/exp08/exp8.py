import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Dataset

df = pd.read_csv(r"C:\Users\M R Computer\OneDrive\Desktop\ann lab\exp08\rent.csv")

print("Dataset Preview:")
print(df.head())

# Features and Target
X = df[['Area','Bedrooms','Bathrooms','Floor','LocationScore']]
y = df['Rent']

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build ANN Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

history = model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=2,
    validation_data=(X_test, y_test),
    verbose=1
)
# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", round(mse,2))
print("R² Score:", round(r2,4))

# Actual vs Predicted
comparison = pd.DataFrame({
    'Actual Rent': y_test.values,
    'Predicted Rent': y_pred.flatten().round(2)
})

print("\nPrediction Comparison:")
print(comparison)

# Loss Graph
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('ANN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Sample House Prediction
sample_house = np.array([
    [1800, 5, 4, 5, 10]
])

sample_house = scaler.transform(sample_house)

predicted_rent = model.predict(sample_house)

print(
    "\nPredicted House Rent:",
    round(predicted_rent[0][0],2),
    "BDT"
)