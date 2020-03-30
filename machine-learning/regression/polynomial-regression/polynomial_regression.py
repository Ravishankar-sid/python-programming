# Base Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Import Data
data = pd.read_csv("../data/Position_Salaries.csv")
data_x = data.iloc[:, 1:2].values
data_y = data.iloc[:, 2].values


# Model Creation and Fit
linear_model = LinearRegression()
linear_model.fit(data_x, data_y)

polynomial_regr = PolynomialFeatures(degree=4)
polynomial_data_x = polynomial_regr.fit_transform(data_x)
polynomial_regr.fit(polynomial_data_x, data_y)
polynomial_model = LinearRegression()
polynomial_model.fit(polynomial_data_x, data_y)

# Plotting Linear Results
plt.scatter(data_x, data_y, color="red")
plt.plot(data_x, linear_model.predict(data_x), color="red")

# Plotting Polynomial Results
plt.scatter(data_x, data_y, color="red")
plt.plot(
    data_x,
    polynomial_model.predict(polynomial_regr.fit_transform(data_x)),
    color="blue",
)


grid_x = np.arange(min(data_x), max(data_x), 0.1)
grid_x = grid_x.reshape(len(grid_x), 1)
plt.scatter(data_x, data_y, color="red")
plt.plot(
    grid_x,
    polynomial_model.predict(polynomial_regr.fit_transform(grid_x)),
    color="blue",
)

# Predicting Values with Linear Regression
linear_model.predict([[6.8]])


# Predicting Values with Polynomial Regression
polynomial_model.predict(polynomial_regr.fit_transform([[6.8]]))
