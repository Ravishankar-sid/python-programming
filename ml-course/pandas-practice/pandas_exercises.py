# -*- coding: utf-8 -*-

"""The Complete Machine Learning Course, Pandas Practice Module
"""

# Imports for Exercise
import pandas as pd
import matplotlib.pyplot as plt


# Basics ---------------------------------------------------

# Pandas Series of three different colours
series_of_colours = pd.Series(data=["red", "blue", "green"], dtype=str)

# Printing the series
print(series_of_colours)

# Series of three different cars and printing it
series_of_cars = pd.Series(data=["volkswagen", "porsche", "car_maker101"], dtype=str)
print(series_of_cars)

# Importing a dataset and turning it into a DF
dataframe = pd.read_csv("../data/car-sales.csv")

# Export to CSV
dataframe.to_csv("exported_data.csv")

# Describing the Dataframe
dataframe.describe()

# Different dtypes on the df
print(dataframe.dtypes)

# DF Information
dataframe.info()

# Series of Int and Mean
mean_series = pd.Series([1, 2, 3, 4, 5]).mean()
print(mean_series)


# Series of Int and Sum
sum_series = pd.Series([1, 2, 3, 4, 5]).sum()
print(sum_series)

# Column Names of DF
print(dataframe.columns.values)

# Length of DF
print(len(dataframe))

# First 5 rows of Dataframe
print(dataframe.head())

# First 7 rows of Dataframe
print(dataframe.head(n=7))

# Last 5 rows of Dataframe
print(dataframe.tail())

# .loc to select the row at index 3 of the car sales DataFrame
print(dataframe.loc[3, :])

# .iloc to select the row at index 3 of the car sales DataFrame
print(dataframe.iloc[3, :])

# Select the "Odometer (KM)" column from the car sales DataFrame
col_odometer = dataframe["Odometer (KM)"]

# Find the mean of the "Odometer (KM)" column in the car sales DataFrame
mean_col_odometer = col_odometer.mean()
print(mean_col_odometer)

# Select the rows with over 100,000 kilometers on the Odometer
dataframe.loc[dataframe["Odometer (KM)"] >= 100000]

# Create a crosstab of the Make and Doors columns
pd.crosstab(dataframe["Make"], dataframe["Doors"])

# Group columns of the car sales DataFrame by the Make column and find the average
dataframe.groupby("Make").mean()

# Import Matplotlib and create a plot of the Odometer column
plt.hist(dataframe["Odometer (KM)"])

# Try to plot the Price column using plot()
dataframe["Price"].plot()  # Does not work due to dtype mismatch

# Remove the punctuation from price column
# Remove the two extra zeros at the end of the price column
dataframe["Price"] = (
    dataframe["Price"].str.replace("$", "").str.replace(",", "").astype(float)
)
# Change the datatype of the Price column to integers
dataframe["Price"] = dataframe["Price"].astype(int)

# Lower the strings of the Make column
dataframe["Make"].str.lower()

# Make lowering the case of the Make column permanent
dataframe["Make"] = dataframe["Make"].str.lower()

# Working with missing Data ---------------------------------------------------
missing_dataframe = pd.read_csv("../data/car-sales-missing-data.csv")
print(missing_dataframe)

# Fill the Odometer (KM) column missing values with the mean of the column inplace
missing_dataframe["Odometer"] = missing_dataframe["Odometer"].fillna(
    missing_dataframe["Odometer"].mean()
)
print(missing_dataframe)

# Remove the rest of the missing data inplace
non_missing_dataframe = missing_dataframe.dropna()

# Verify the missing values are removed by viewing the DataFrame
print(non_missing_dataframe)

# Column Manipulations --------------------------------------------------------

# Create a "Seats" column where every row has a value of 5
missing_dataframe["Seats"] = 5

# Create a column called "Engine Size" with random values between 1.3 and 4.5
# Remember: If you're doing it from a Python list, the list has to be the same length
# as the DataFrame
missing_dataframe["Engine Size"] = [1.2, 2.3, 3.4, 4.5, 3.4, 2.3, 1.2, 2.3, 3.4, 4.5]

# Create a column which represents the price of a car per kilometer
# Then view the DataFrame
missing_dataframe["Price"] = (
    missing_dataframe["Price"].str.replace(r"[\$\,\.]", "").astype(float)
)
missing_dataframe["Price Per Km"] = (
    missing_dataframe["Price"] / missing_dataframe["Odometer"]
)

# Remove the last column you added using .drop()
missing_dataframe = missing_dataframe.drop("Price Per Km", axis=1)

# Shuffle the DataFrame using sample() with the frac parameter set to 1
# Save the the shuffled DataFrame to a new variable
shuffled_data = missing_dataframe.sample(frac=1)

# Reset the indexes of the shuffled DataFrame
shuffled_data = shuffled_data.reset_index().drop("index", axis=1)

# Change the Odometer values from kilometers to miles using a Lambda function
# Then view the DataFrame
shuffled_data["Odometer"] = shuffled_data["Odometer"].apply(lambda x: x / 1.609)
shuffled_data["Odometer"] = shuffled_data["Odometer"].map(lambda x: x / 1.609)

# Change the title of the Odometer (KM) to represent miles instead of kilometers
shuffled_data = shuffled_data.rename(columns={"Odometer": "Odometer(Miles)"})
