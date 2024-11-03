# Define the data types for specific columns
dtype_dict = {
    'id': 'int64',            # Integer type
    'ticker': 'str',          # Categorical type
    'tweet': 'str'
}

# Load CSV file with specified data types
df = pd.read_csv('./data/stocktweet/stocktweet.csv', dtype=dtype_dict, parse_dates=['date'])
