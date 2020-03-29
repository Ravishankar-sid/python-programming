# -*- coding: utf-8 -*-

"""
Simple Linear Regression Model
"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
data = pd.read_csv("../data/Salary_Data.csv")
data_x = data.iloc[:, :-1].values
data_y = data.iloc[:, 1].values

# Train - Test Split
trainset_data_x, testset_data_x, trainset_data_y, testset_data_y = train_test_split(
    data_x, data_y, test_size=0.33, random_state=0
)

# Model Creation and Fit on Training Set
linear_model = LinearRegression()
linear_model.fit(trainset_data_x, trainset_data_y)

# Prediction with Test Set
predection_y = linear_model.predict(testset_data_x)

# Plotting Training Set Results
plt.scatter(trainset_data_x, trainset_data_y, color="red")

plt.plot(trainset_data_x, linear_model.predict(trainset_data_x), color="blue")
plt.title("Salary Vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")


# Plotting Test Set Results
plt.scatter(testset_data_x, testset_data_y, color="red")

plt.plot(trainset_data_x, linear_model.predict(trainset_data_x), color="blue")
plt.title("Salary Vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
