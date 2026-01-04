# -*- coding: utf-8 -*-
"""
Project: US Eastern Energy Load Forecasting (PJME)
Description: 
    This script trains an LSTM model to predict hourly energy consumption.
    It includes a comparative analysis of model performance between 
    short training (e.g., 150 epochs) and long training (e.g., 500 epochs) 
    to demonstrate overfitting and diminishing returns.

Dataset: PJM Hourly Energy Consumption Data (PJME_hourly.csv)
Region: Eastern United States Interconnection

Author: reikiridvan @industthree Youtube Channel
Date: 2024
Environment: TensorFlow, Pandas, Matplotlib, Scikit-Learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import math
import os
import random

# ==========================================
# 1. CONTROL PANEL
# ==========================================
CONFIG = {
    # Updated Data Source:
    'DATA_URL': 'https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv',
    'LOOK_BACK': 24,       # Look back 24 hours
    'FORECAST_STEPS': 48,  # Forecast 48 hours ahead
    'EPOCHS': 15,          # Number of training epochs (Adjust this for experiments)
    'BATCH_SIZE': 32,
    'NEURONS': 64,
    'SEED': 42,
    'DATA_LIMIT': 5000     # Use the last 5000 data points
}

# ==========================================
# 2. SYSTEM SETUP
# ==========================================
os.environ['PYTHONHASHSEED'] = str(CONFIG['SEED'])
random.seed(CONFIG['SEED'])
np.random.seed(CONFIG['SEED'])
tf.random.set_seed(CONFIG['SEED'])

print("="*50)
print("🚀 STARTING FORECASTING SYSTEM (USA REAL DATA)")
print("="*50)

# ==========================================
# 3. LOAD DATA FROM SOURCE
# ==========================================
print("[1/6] Downloading Real Data from Server (USA PJM)...")

try:
    # Read directly from the URL
    df = pd.read_csv(CONFIG['DATA_URL'], parse_dates=['Datetime'], index_col='Datetime')
    df = df.sort_index() # Sort by datetime
    
    # Take the most recent data
    df_small = df.tail(CONFIG['DATA_LIMIT'])
    data = df_small['PJME_MW'].values
    
    print(f"   -> ✅ Success! Using PJM East (USA) data.")
    print(f"   -> Total Data: {len(data)} Hours")
    
except Exception as e:
    print(f"   -> ❌ Download failed: {e}")
    print("   -> ⚠️ Please check your internet connection!")
    # Fallback data if internet is down
    t = np.arange(0, 3000)
    data = (np.sin(t * 0.05) * 20) + 500 + np.random.normal(0, 2, 3000)

# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(data.reshape(-1, 1))

# ==========================================
# 4. PREPROCESSING
# ==========================================
print("[2/6] Preparing Sequences...")

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

X, y = create_dataset(dataset_scaled, CONFIG['LOOK_BACK'])
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

train_size = int(len(X) * 0.8)
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# ==========================================
# 5. MODEL ARCHITECTURE
# ==========================================
print("[3/6] Building LSTM Architecture...")
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(CONFIG['NEURONS'], return_sequences=True, input_shape=(1, CONFIG['LOOK_BACK'])),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(CONFIG['NEURONS']),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')

# ==========================================
# 6. TRAINING
# ==========================================
print(f"[4/6] Training AI ({CONFIG['EPOCHS']} Epochs)...")
history = model.fit(
    X_train, y_train, 
    epochs=CONFIG['EPOCHS'], 
    batch_size=CONFIG['BATCH_SIZE'], 
    validation_data=(X_test, y_test),
    verbose=1
)

# ==========================================
# 7. EVALUATION
# ==========================================
print("[5/6] Calculating Accuracy...")
test_predict = model.predict(X_test, verbose=0)
test_predict_real = scaler.inverse_transform(test_predict)
y_test_real = scaler.inverse_transform([y_test])

rmse = math.sqrt(mean_squared_error(y_test_real[0], test_predict_real[:,0]))
mape = mean_absolute_percentage_error(y_test_real[0], test_predict_real[:,0])

print("\n" + "="*40)
print(f"📊 FINAL REPORT (PJM DATA - USA)")
print(f"✅ RMSE: {rmse:.4f} MW")
print(f"✅ MAPE: {mape*100:.2f}% (Error Rate)")
print("="*40)

# ==========================================
# 8. VISUALIZATION
# ==========================================
print("[6/6] Generating Comprehensive Plots...")

# Future Forecasting
last_window = dataset_scaled[-CONFIG['LOOK_BACK']:]
current_batch = last_window.reshape((1, 1, CONFIG['LOOK_BACK']))
future_predictions = []

for i in range(CONFIG['FORECAST_STEPS']):
    pred = model.predict(current_batch, verbose=0)[0]
    future_predictions.append(pred)
    current_batch = np.append(current_batch[:, :, 1:], [[pred]], axis=2)

future_predictions = scaler.inverse_transform(future_predictions)

# Plotting
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

# Chart 1: Prediction vs Actual (Zoomed)
ax1 = fig.add_subplot(gs[0])
zoom_start = len(dataset_scaled) - 200
ax1.plot(np.arange(zoom_start, len(dataset_scaled)), 
         scaler.inverse_transform(dataset_scaled[zoom_start:]), 
         label='Actual Data (Real)', color='#1f77b4', linewidth=2)
future_range = np.arange(len(dataset_scaled), len(dataset_scaled) + CONFIG['FORECAST_STEPS'])
ax1.plot(future_range, future_predictions, 
         label='AI Prediction (Future)', color='#ff7f0e', linestyle='--', linewidth=3, marker='o', markersize=4)
ax1.set_title(f'FORECASTING: PJM Electricity Load (USA) | Error: {mape*100:.2f}%', fontsize=14, fontweight='bold')
ax1.set_ylabel('Load (MW)', fontsize=12)
ax1.legend()
ax1.axvline(x=len(dataset_scaled), color='green', linestyle=':', label='Current Time')

# Chart 2: Model Loss
ax2 = fig.add_subplot(gs[1])
ax2.plot(history.history['loss'], label='Training Loss', color='blue')
ax2.plot(history.history['val_loss'], label='Validation Loss', color='red')
ax2.set_title('Model Learning Curve (Convergence Check)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss')
ax2.legend()

# Chart 3: Overview
ax3 = fig.add_subplot(gs[2])
ax3.plot(scaler.inverse_transform(dataset_scaled), color='grey', alpha=0.5)
ax3.set_title(f'Full Data Overview (Last {CONFIG["DATA_LIMIT"]} Hours)', fontsize=12, fontweight='bold')
ax3.set_ylabel('MW')

plt.tight_layout()
plt.show()

print("✅ DONE! Check the 'Plots' tab.")