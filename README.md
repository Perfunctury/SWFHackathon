# SWF Hackathon

# Step 1: Import libraries

To begin executing the model, the following libraries will need to be imported
1) pandas
Used for data cleaning, exploration, manipulation, and overall analysis.

2) sklearn / scikit-learn
An open-source machine learning library to train the model in Python that provides
the tools and algorithms. Our model utilizes supervised learning for time-series forecasting.
In this program, sklearn is used to calculate the accuracy of the model based on the mean squared error and R-squared score.

Overall, scikit-learn is a powerful and versatile library for machine learning tasks in Python. 
It is widely used in data science and machine learning projects and is known for its user-friendly 
API and extensive documentation.

3) datetime
A built-in Python module that provides classes for working with dates and time. This allows the model to manipulate dates, 
times, and intervals.


# Step 2: Load and clean data

Once all libraries and modules are imported data cleaning can begin by loading the data provided from a .csv file named
'SWE_Training.csv' using the pandas read_csv() function. It then drops rows with -9999 value using drop() function, 
and converts the date column to datetime format using to_datetime() function, and extracts year and cumulative day 
of the year as new columns using dt.year and dt.dayofyear functions respectively. It then saves the cleaned data as a 
new CSV file named 'Modified_SWE_Training.csv'.

# Step 3: Split the data
The independent variables (features) and dependent variable (target) are separated from the cleaned data. 
The independent variables are year, day_of_year, precip_cumulative, and tmean. The dependent variable is SWE. 
The data is then split into training and testing sets using the train_test_split() function. The random forest model
is then created using the RandomForestRegressor() function from scikit-learn. The number of estimators is set to 100,  
and the random state is set to 42.

# Step 4: Fit the model
The model is trained on the training data using the 'fit' method. Predictions are made on the testing data 
using the 'predict' method, and the model's performance is evaluated using mean squared error and R-squared.

The standard deviation of the predicted values is calculated using NumPy's 'std' function.

The suspect data is loaded from a CSV file named 'Suspect_SWE_Data.csv'. The date column is converted to a 
datetime format, and new columns are added for the year and day of the year. Any rows containing the value 
'-9999' are removed, and the modified data is saved to a new CSV file named 'Modified_Unfiltered_Suspect_Data.csv'.

# Step 5: Make predictions
Predictions are made on the modified suspect data using the trained Random Forest Regressor model. 
Any entries with residuals greater than 2 standard deviations are filtered out and saved to a new file named 
'suspect_measurements.csv'. The original data, excluding the filtered entries, is saved to a new file named 
'filtered_data.csv'.

# Step 6: Performance evaluation
The performance of the model is evaluated using the mean squared error (MSE) and R-squared (R2) values. 
The mean squared error is calculated using the mean_squared_error() function, and the R-squared value is 
calculated using the r2_score() function. The values are then printed to the console using the print() function.