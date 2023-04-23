import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

df = pandas.read_csv('SWE_Training.csv')

df = df.drop(df[df==-9999].dropna(how='all').index)

# Convert the date column to datetime format
df['date'] = pandas.to_datetime(df['date'], format='%d-%b-%y')

# Extract year and cumulative day of the year as new columns
df['year'] = df['date'].dt.year
df['day_of_year'] = df['date'].dt.dayofyear

# Save as new data to be trained with
df.to_csv('Modified_SWE_Training.csv', index=False)

# Step 1: Load data
data = pandas.read_csv('Modified_SWE_Training.csv')

# Separate independent variables and dependent variable
X = df[['year', 'day_of_year', 'precip_cumulative', 'tmean']].values # Features
y = df['swe'].values # Target

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create a Random Forest Regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Step 5: Fit the model on the training data
rf.fit(X_train, y_train)

# Step 6: Make predictions on the testing data
y_pred = rf.predict(X_test)

# Step 7: Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared:', r2)