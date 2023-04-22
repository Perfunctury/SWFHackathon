# SFW Hackathon
Description:
The code is performing a linear regression analysis on Tuolomne Meadows to predict
snow-water-equivalent based on weights: day-of-the-year, precipitation, and the daily-mean temperature.

First the code reads from the .csv dataset using pandas DataFrame. The weights are then converted too
numerical values to be selected such that the snow-water-equivalent can be used for regression analysis.

The program then creates an empty list called "skipped_rows" to store the index of any rows with missing
or invalid data such as "-9999" that appears in the assumed valid dataset. It then loops through each row
in set DataFrame to check if, specifically "-9999" exist to be returned as null. Such that if it does contain
the invalid value that row will be added to skipped_rows list and continues to move onto the next row of data

If a row does contain the targeted invalid value(s) the rows get splits into training and testing using
train_test_split() function from the library scikit-learn. The supervised learning ground is currently
set to a standard of 20% to test/predict based on the sample amount 'n', where 'n' is currently set to 100.

Utilizing scikit-learn library the StandardScaler() function is used to standardize the data. Also, from
scikit-learn the LinearRegression() function trains the linear regression model on the scaled training data.
Thus allowing the model to make prediction based on the scaled testing data.

The code then calculates the mean squared error and R2 score of the model using the mean_squared_error()
and r2_score() functions from scikit-learn to print the metric to onto the console.

Lastly a scatter plot is created used from matplotlib.
