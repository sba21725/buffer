# Create the SQLAlchemy engine for MySQL
engine = create_engine(f"mysql+pymysql://{username}:{password}@{hostname}:{port}/{database_name}")

df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

table_name="stocktweet"

# Insert the data into the MySQL table
df.to_sql(table_name, engine, if_exists='append', index=False)

print("Data loaded into MySQL table successfully.")
