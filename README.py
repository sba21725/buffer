for ticker in tickers[:1]:
    # Retrieve stockprice for a Company from MySQL
    query = """
    SELECT 
        Stockprice.ticker,
        Stockprice.Date,
        Stockprice.Open,
        Stockprice.High,
        Stockprice.Low,
        Stockprice.Close,
        Stockprice.AdjClose,
        Stockprice.Volume
    FROM 
        Stockprice
    WHERE
        Stockprice.ticker = '{}'
    """.format(ticker)
    print(ticker)

    # Fetch data into a pandas DataFrame
    stockprice_df = pd.read_sql(query, engine)

    stockprice_df['Date'] = pd.to_datetime(stockprice_df['Date'], format='%Y-%m-%d')

    # Rename column Date to perform left join
    stockprice_df.rename(columns={'Date': 'date'}, inplace=True)

    # Performing a left join on both 'ticker' and 'date'
    stockprice_merged_df = pd.merge(stockprice_df, stocktweet_grouped_df, on=['ticker', 'date'], how='left')

    stockprice_merged_df.fillna(0, inplace=True)

    stockprice_merged_df = stockprice_merged_df.drop(columns=['ticker'])

    stockprice_merged_df.set_index('date', inplace=True)





    # Selecting the features 
    data = stockprice_merged_df[['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume', 'sentiment_score']]

    # Scale the data 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Define a function to create a dataset with time steps and multiple outputs (1-day, 3-days, 7-days)
    def create_dataset(data, time_step=60, forecast_days=[1, 3, 7]):
        X, y = [], []
        for i in range(time_step, len(data) - max(forecast_days)):
            X.append(data[i-time_step:i, :])  # Input features from previous 'time_step' days
            # Target: Close price for 1-day, 3-days, and 7-days in the future
            y.append([data[i + forecast_day, 3] for forecast_day in forecast_days])
        return np.array(X), np.array(y)

    # Create the dataset
    time_step = 60  # Look back 60 days to predict the next day
    forecast_days = [1, 3, 7]  # Predict 1-day, 3-days, and 7-days ahead for Close price
    X, y = create_dataset(scaled_data, time_step, forecast_days)

    # Split the dataset into training and testing sets (80% training, 20% testing)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape input data for LSTM [samples, time_steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Build the LSTM model
    model = Sequential()

    # Adding the first LSTM layer with Dropout
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))

    # Adding a second LSTM layer with Dropout
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Adding the output layer with 3 neurons for 1 day, 3 days, and 7 days predictions
    model.add(Dense(units=3))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=64)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Inverse transform the predicted and actual Close prices to get back to the original scale
    predictions_transformed = []
    for i in range(predictions.shape[1]):  # For each forecast (1, 3, 7 days)
        predictions_transformed.append(
            scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], scaled_data.shape[1] - 1)), 
                                                     predictions[:, i].reshape(-1, 1)), axis=1))[:, 3]
        )

    # Inverse transform actual data
    y_test_transformed = []
    for i in range(y_test.shape[1]):  # For each forecast (1, 3, 7 days)
        y_test_transformed.append(
            scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], scaled_data.shape[1] - 1)), 
                                                     y_test[:, i].reshape(-1, 1)), axis=1))[:, 3]
        )

    # Evaluate the model performance using RMSE for 1-day, 3-day, and 7-day predictions
    rmse_1day = np.sqrt(mean_squared_error(y_test_transformed[0], predictions_transformed[0]))
    rmse_3day = np.sqrt(mean_squared_error(y_test_transformed[1], predictions_transformed[1]))
    rmse_7day = np.sqrt(mean_squared_error(y_test_transformed[2], predictions_transformed[2]))

    print(f'Root Mean Squared Error (1-day): {rmse_1day}')
    print(f'Root Mean Squared Error (3-day): {rmse_3day}')
    print(f'Root Mean Squared Error (7-day): {rmse_7day}')

    doc = {"ticker": ticker,
         "1D": predictions_transformed[0][0],
         "3D": predictions_transformed[1][0],
         "7D": predictions_transformed[2][0]}

    print(doc)
    
    # Insert forecast into MongoDB Collection
    # result = lstm_coll.insert_one(doc)

    # Plot the actual vs predicted Close prices for each forecast horizon
    plt.figure(figsize=(10, 8))

    # Plot for 1-day ahead prediction
    plt.subplot(3, 1, 1)
    plt.plot(y_test_transformed[0], label='Actual Close Price (1-day ahead)')
    plt.plot(predictions_transformed[0], label='Predicted Close Price (1-day ahead)')
    plt.title('1-Day Ahead Close Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()

    # Plot for 3-day ahead prediction
    plt.subplot(3, 1, 2)
    plt.plot(y_test_transformed[1], label='Actual Close Price (3-days ahead)')
    plt.plot(predictions_transformed[1], label='Predicted Close Price (3-days ahead)')
    plt.title('3-Days Ahead Close Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()

    # Plot for 7-day ahead prediction
    plt.subplot(3, 1, 3)
    plt.plot(y_test_transformed[2], label='Actual Close Price (7-days ahead)')
    plt.plot(predictions_transformed[2], label='Predicted Close Price (7-days ahead)')
    plt.title('7-Days Ahead Close Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()

    plt.tight_layout()
    plt.show()
