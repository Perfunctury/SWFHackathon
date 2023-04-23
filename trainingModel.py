import pandas
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

#Load the training data
unfiltered_data = pandas.read_csv('SWE_Training.csv')

#Filter out any rows with -9999
unfiltered_data = unfiltered_data.drop(unfiltered_data[unfiltered_data==-9999].dropna(how='all').index)

# Convert the date column to datetime format
unfiltered_data['date'] = pandas.to_datetime(unfiltered_data['date'], format='%d-%b-%y')

# Extract year and cumulative day of the year as new columns
unfiltered_data['year'] = unfiltered_data['date'].dt.year
unfiltered_data['day_of_year'] = unfiltered_data['date'].dt.dayofyear

# Save as data to be trained with
unfiltered_data.to_csv('Modified_SWE_Training.csv', index=False)

# Load training data
training_data = pandas.read_csv('Modified_SWE_Training.csv')

# Separate independent variables and dependent variable
X = training_data[['year', 'day_of_year', 'precip_cumulative', 'tmean']].values # Features
y = training_data['swe'].values # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
rf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared:', r2)

# Calculate Residuals
residuals = y_test - y_pred

# Calculate the standard deviation of the residuals
std_dev = np.std(residuals)

# Load the suspect data
suspect_data = pandas.read_csv('Suspect_SWE_Data.csv')

# Process the new data
# Convert the date column to datetime format
suspect_data['date'] = pandas.to_datetime(suspect_data['date'], format='%d-%b-%y')

# Extract year and cumulative day of the year as new columns
suspect_data['year'] = suspect_data['date'].dt.year
suspect_data['day_of_year'] = suspect_data['date'].dt.dayofyear

# Drop any rows with -9999
suspect_data = suspect_data.drop(suspect_data[suspect_data==-9999].dropna(how='all').index)

# Make predictions on the new data
X_Suspect = suspect_data[['year', 'day_of_year', 'precip_cumulative', 'tmean']].values # Features
y_Suspect = suspect_data['swe'].values # Target
y_suspect_pred = rf.predict(X_Suspect)

# Filter out entries with residuals greater than 2 standard deviations
suspect_measurements = suspect_data[np.abs(y_suspect_pred - y_Suspect) >= 2 * std_dev]

# Save the filtered data to a new file
suspect_measurements.to_csv('suspect_measurements.csv', index=False)

# Save the original data, except for the filtered entries, to a new file
filtered_data = suspect_data[np.abs(y_suspect_pred - y_Suspect) < 2 * std_dev]
filtered_data.to_csv('filtered_data.csv', index=False)