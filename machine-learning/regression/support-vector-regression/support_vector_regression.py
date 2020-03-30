# Base Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the dataset
data = pd.read_csv("../data/Position_Salaries.csv")
data_x = data.iloc[:, 1:2].values
data_y = data.iloc[:, 2].values

# Feature Scaling
scaler_x = StandardScaler()
scaler_y = StandardScaler()
data_x = scaler_x.fit_transform(data_x)
data_y = scaler_y.fit_transform(data_y.reshape(-1, 1))

# Model Creation and Fitting
support_vector_model = SVR()
support_vector_model.fit(data_x, data_y)

# Predictions
prediction_y = support_vector_model.predict([[6.5]])
prediction_y = scaler_y.inverse_transform(prediction_y)

# Plotting Results
plt.scatter(data_x, data_y, color="red")
plt.plot(data_x, support_vector_model.predict(data_x), color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

# Advanced Plot
grid_x = np.arange(min(data_x), max(data_x), 0.01)  # Choice of 0.01 as data is scaled
grid_x = grid_x.reshape((len(grid_x), 1))
plt.scatter(data_x, data_y, color="red")
plt.plot(grid_x, support_vector_model.predict(grid_x), color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
