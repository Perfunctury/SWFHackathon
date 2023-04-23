import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
#
# Read the CSV file and store it in a pandas DataFrame
data = pd.read_csv('valid_dataset.csv')

# Convert date to a numerical feature (e.g., day of the year)
data['day_of_year'] = pd.to_datetime(data['day_of_year'], format='%d-%b-%y').dt.dayofyear

# Select the features and target variable
X = data[['day', 'year', 'day_of_year', 'precipitation', 'temperature']]
y = data['swe']

skipped_rows = []
y_skipped = []

# Split the data into training and testing sets
for index, row in X.iterrows():
    if np.isin(row.values, '-9999').any():
        skipped_rows.append(index)
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=100, random_state=42)

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a random forest regression model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test_scaled)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    # Store predicted values for non-skipped rows
    y_skipped.append(y_pred)

    # Plot actual vs. predicted snow water equivalent for non-skipped rows
    plt.scatter(y_test, y_pred)
    plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), color='red')
    plt.xlabel('Actual Snow Water Equivalent')
    plt.ylabel('Predicted Snow Water Equivalent')
    plt.title('Actual vs. Predicted Snow Water Equivalent')
    plt.show()

# Plot actual vs. predicted snow water equivalent for skipped rows
if skipped_rows:
    y_skipped = np.concatenate(y_skipped)
    y_skipped = y_skipped.reshape(-1, 1)
    y_test_skipped = np.zeros(y_skipped.shape[0])
    plt.scatter(y_test_skipped, y_skipped)
    plt.xlabel('Actual Snow Water Equivalent')
    plt.ylabel('Predicted Snow Water Equivalent')
    plt.title('Actual vs. Predicted Snow Water Equivalent (Skipped Rows)')
    plt.show()

print(f"Skipped Rows: {skipped_rows}")
