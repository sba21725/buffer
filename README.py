# Function to evaluate ARIMA model with given order
def evaluate_arima_model(order):
    try:
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test))
        error = mean_squared_error(test, predictions)
        return error
    except:
        return float("inf")

for ticker in tickers:
    # Retrieve stockprice for a Company from MySQL
    query = """
        SELECT 
            Stockprice.Date,
            Stockprice.Close
        FROM 
            Stockprice
        WHERE
            Stockprice.ticker = '{}';
    """.format(ticker)

    # Fetch data into a pandas DataFrame
    stockprice_df = pd.read_sql(query, engine)

    # stockprice_df.fillna(0, inplace=True)
    stockprice_df.set_index('Date', inplace=True)
    print(ticker)


    # ----------------------------------------------------------------------
    # Hyperparameter Tuning
    # ----------------------------------------------------------------------
    
    # Define the range of p, d, and q values to try
    p_values = range(0, 5)
    d_values = range(0, 5)
    q_values = range(0, 5)

    # Generate all combinations of p, d, q
    pdq_combinations = list(itertools.product(p_values, d_values, q_values))

    # Train-test split
    train_size = int(len(stockprice_df) * 0.8)
    train, test = stockprice_df[:train_size], stockprice_df[train_size:]

    # Hyperparameter tuning
    best_score, best_order = float("inf"), None

    for order in pdq_combinations:
        error = evaluate_arima_model(order)
        if error < best_score:
            best_score, best_order = error, order
        print(f"ARIMA{order} MSE={error:.3f}")

    print(f"Best ARIMA{best_order} MSE={best_score:.3f}")

    # Fit and forecast using the best order
    model = ARIMA(train, order=best_order)
    model_fit = model.fit()


    # Forecast for 1 day
    forecast_1d = model_fit.forecast(steps=1)
    
    # Forecast for 3 days
    forecast_3d = model_fit.forecast(steps=3)

    # Forecast for 7 days
    forecast_7d = model_fit.forecast(steps=7)

    print("FORECAST 1D ")
    print(forecast_1d.iloc[0])
    print("FORECASTS 3D ")
    print(forecast_3d.iloc[2])
    print("FORECASTS 7D ")
    print(forecast_7d.iloc[6])
    
    
    doc = {"ticker": ticker,
         "1D": forecast_1d.iloc[0],
         "3D": forecast_3d.iloc[0],
         "7D": forecast_7d.iloc[0]}

    # Insert forecast into MongoDB Collection
    result = arima_coll.insert_one(doc)

    # Evaluate the model on test data
    forecast_values = model_fit.forecast(steps=len(test))
    mse = mean_squared_error(test, forecast_values)
    print(f"Mean Squared Error: {mse}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(stockprice_df.index, stockprice_df['Close'], label='Close price')
    plt.plot(test.index, forecast_values, label='Forecasted Close price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Close price')
    plt.legend()
    plt.show()
