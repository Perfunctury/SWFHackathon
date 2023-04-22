import pandas as pd

# Read the CSV file and store it in a pandas DataFrame
data = pd.read_csv('Tuolumne_Meadows.csv')

# Convert date column to pandas DateTimeIndex
data['day_of_year'] = pd.to_datetime(data['day_of_year'], format='%d-%b-%y')
data.set_index('day_of_year', inplace=True)

# Resample the data on a daily basis
daily_data = data.resample('D').mean()

# Check for missing days
missing_days = daily_data.isnull().any(axis=1)

# Print the missing days
print("Missing days:")
print(missing_days[missing_days == True])
