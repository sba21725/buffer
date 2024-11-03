# Define the SQL query for creating the table
create_table_query = """
CREATE TABLE IF NOT EXISTS stocktweet (
    id INT PRIMARY KEY,
    date DATE,
    ticker VARCHAR(6),
    tweet TEXT
);
"""

# Execute the query to create the table
try:
    with connection.cursor() as cursor:
        cursor.execute(create_table_query)
        print("Table 'example_table' created successfully.")
finally:
    # Close the connection
    connection.close()
