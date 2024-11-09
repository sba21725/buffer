import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import keras_tuner as kt
import matplotlib.pyplot as plt

# Load your dataset (replace with your actual dataset path)
data = stockprice_merged_df
features = ['Open', 'High', 'Low', 'Close', 'Volume']
data = data[features]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare the data for LSTM
def create_dataset(data, time_step=60, forecast_day=1):
    X, y = [], []
    for i in range(time_step, len(data) - forecast_day):
        X.append(data[i - time_step:i, :])  # Use the past 'time_step' days of data
        y.append(data[i + forecast_day - 1, 3])  # Predict 'Close' price at 'forecast_day' in the future
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step, forecast_day=1)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Step 1: Define a function to build the LSTM model for hyperparameter tuning
def build_model(hp):
    model = Sequential()

    # Hyperparameter for the number of LSTM units
    model.add(LSTM(units=hp.Int('units', min_value=50, max_value=200, step=50), 
                   return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
    
    # Hyperparameter for the dropout rate to prevent overfitting
    model.add(Dense(units=1))  # Output: single value (close price)
    
    # Hyperparameter for optimizer learning rate
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
                  loss='mean_squared_error')
    
    return model

# Step 2: Use Keras Tuner to find the best hyperparameters
tuner = kt.Hyperband(build_model, 
                     objective='val_loss', 
                     max_epochs=10, 
                     hyperband_iterations=2, 
                     directory='my_dir', 
                     project_name='lstm_tuning')

# Step 3: Perform the hyperparameter search
tuner.search(X, y, epochs=10, validation_split=0.2, batch_size=32)

# Step 4: Get the best hyperparameters
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best Hyperparameters:")
print(best_hyperparameters)

# Step 5: Train the model with the best hyperparameters
best_model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Step 6: Make predictions with the best model
predicted_price = best_model.predict(X)

# Inverse transform the predicted data back to the original scale
predicted_price = scaler.inverse_transform(np.hstack((np.zeros((predicted_price.shape[0], 4)), predicted_price)))

# Step 7: Visualize the results
actual_price = scaler.inverse_transform(np.hstack((scaled_data[time_step:], np.zeros((scaled_data.shape[0] - time_step, 4)))))

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(actual_price[:, 3], label='Actual Close Price')  # Column 3 is 'Close'
plt.plot(predicted_price[:, 3], label='Predicted Close Price')
plt.title('LSTM Model with Hyperparameter Tuning - Predicting Close Price')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.show()
