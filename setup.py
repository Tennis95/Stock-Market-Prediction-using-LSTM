import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import openpyxl

# Load dataset (replace with your file path)
df = pd.read_excel("Build/Processed_Stock_Data.xlsx")

# Preprocess dataset
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Feature Engineering
df['Close_Lag1'] = df['Close'].shift(1)
df['Close_Lag7'] = df['Close'].shift(7)
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()
df.fillna(method='bfill', inplace=True)

# Scaling data
scaler = MinMaxScaler()
scaled_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Close_Lag1', 'Close_Lag7', 'MA_7', 'MA_30']
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

# Define target and features
target_col = 'Close'
feature_cols = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close_Lag1', 'Close_Lag7', 'MA_7', 'MA_30']

X = df[feature_cols].values
y = df[target_col].values

# Train-Test Split (80%-20%)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Convert to LSTM sequences
window_size = 30

def create_lstm_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
        y_seq.append(y[i + window_size])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_lstm_sequences(X_train, y_train, window_size)
X_test_seq, y_test_seq = create_lstm_sequences(X_test, y_test, window_size)

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(window_size, len(feature_cols))),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
history = model.fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq), epochs=10, batch_size=32)

# Plot training loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Predict and plot results
y_pred = model.predict(X_test_seq)

plt.figure(figsize=(10, 5))
plt.plot(y_test_seq, label="Actual Price")
plt.plot(y_pred, label="Predicted Price")
plt.legend()
plt.show()
# Evaluate model performance
loss, mae = model.evaluate(X_test_seq, y_test_seq)
print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test MAE: {mae:.4f}")
from sklearn.metrics import mean_squared_error

# Predictions on test set
y_pred = model.predict(X_test_seq)

# Compute RMSE
rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred))
print(f"Test RMSE: {rmse:.4f}")
from sklearn.metrics import r2_score

# Compute R² Score
r2 = r2_score(y_test_seq, y_pred)
print(f"Test R² Score: {r2:.4f}")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_test_seq, label="Actual Price", color='blue')
plt.plot(y_pred, label="Predicted Price", color='red', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Normalized Price")
plt.legend()
plt.title("Actual vs Predicted Stock Prices")
plt.show()
mape = np.mean(np.abs((y_test_seq - y_pred) / y_test_seq)) * 100
print(f"Test MAPE: {mape:.2f}%")
