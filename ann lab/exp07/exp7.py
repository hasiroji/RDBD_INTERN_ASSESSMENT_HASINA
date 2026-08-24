import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load Dataset

df = pd.read_csv(r"C:\Users\M R Computer\OneDrive\Desktop\ann lab\exp07\student.csv")


print("Dataset Preview:")
print(df.head())

# Features and Target
X = df[['Attendance','Marks','Assignment','Quiz','StudyHours']]
y = df['Pass']

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build Feedforward Neural Network
model = Sequential()

model.add(Dense(
    units=10,
    activation='relu',
    input_shape=(X_train.shape[1],)
))

model.add(Dense(
    units=6,
    activation='relu'
))

model.add(Dense(
    units=1,
    activation='sigmoid'
))

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=4,
    validation_data=(X_test, y_test),
    verbose=1
)

# Prediction
y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5).astype(int)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy*100,2), "%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Loss Graph
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Test Prediction
sample_student = np.array([[10, 80, 78, 75, 40]])

sample_student = scaler.transform(sample_student)

prediction = model.predict(sample_student)

if prediction[0][0] > 0.5:
    print("\nPredicted Result: PASS")
else:
    print("\nPredicted Result: FAIL")