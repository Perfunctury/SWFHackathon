
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Read the CSV file and store it in a pandas DataFrame
data = pd.read_csv('valid_dataset.csv')

# Convert date to a numerical feature (e.g., day of the year)
data['day_of_year'] = pd.to_datetime(data['day_of_year'], format='%d-%b-%y').dt.dayofyear

# Select the features and target variable
X = data[['day', 'year', 'day_of_year', 'precipitation', 'temperature']]
y = np.where(data['swe'] > data['swe'].mean(), 1, 0)

# Create a list to store the skipped rows
skipped_rows = []

# Split the data into training and testing sets
for index, row in X.iterrows():
    if row['temperature'] == -9999:
        skipped_rows.append(index)
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=100, random_state=42)

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Transform the features to higher degree polynomials
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    # Train a logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_poly, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test_poly)

    # Calculate performance metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy}")

    # Plot actual vs. predicted classes
    plt.scatter(y_test, y_pred)
    plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), color='red')

    plt.xlabel('Actual Class')
    plt.ylabel('Predicted Class')
    plt.title('Actual vs. Predicted Classes')
    plt.show()

print(f"Skipped Rows: {skipped_rows}")
