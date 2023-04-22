import pandas as pd              # Library for data manipulation and analysis
import numpy as np               # Library for numerical operations
from sklearn.model_selection import train_test_split  # Function for splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler      # Class for feature scaling (standardization)
from sklearn.linear_model import LinearRegression     # Class for linear regression
from sklearn.metrics import mean_squared_error, r2_score  # Functions for calculating performance metrics
import matplotlib.pyplot as plt  # Library for data visualization


# Read the CSV file and store it in a pandas DataFrame
data = pd.read_csv('Tuolumne_Meadows.csv')

# Check for missing values
# Print the number of missing values in each column
print(data.isnull().sum())

# Drop rows with missing values (if necessary)
# Remove rows with missing values from the DataFrame
data = data.dropna()

# Convert date to a numerical feature (e.g., day of the year)
# Convert the date column to datetime objects
# Add a new column with the day of the year for each date
data['date'] = pd.to_datetime(data['date'])
data['day_of_year'] = data['date'].dt.dayofyear

# Select the features and target variable
# Features matrix
X = data[['day_of_year', 'precipitation', 'snow_water_equivalent']]
y = data['mean_daily_air_temperature'] # Target vector

# Split the data into training and testing sets
# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Instantiate a StandardScaler object
scaler = StandardScaler()
# Fit the scaler to the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)
# Transform the testing data using the fitted scaler
X_test_scaled = scaler.transform(X_test)

# Instantiate a LinearRegression object
model = LinearRegression()
# Train the linear regression model using the scaled training data
model.fit(X_train_scaled, y_train)


# Make predictions on the scaled testing data
y_pred = model.predict(X_test_scaled)

# Calculate performance metrics
# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate the R2 score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")


# Create a scatter plot of actual vs. predicted temperatures
plt.scatter(y_test, y_pred)

# Set the x-axis label
plt.xlabel('Actual Temperatures')
# Set the y-axis label
plt.ylabel('Predicted Temperatures')
# Set the plot title
plt.title('Actual vs. Predicted Mean Daily Air Temperatures')
plt.show()


