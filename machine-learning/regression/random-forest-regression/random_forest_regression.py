# Base Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
data = pd.read_csv("../data/Position_Salaries.csv")
data_x = data.iloc[:, 1:2].values
data_y = data.iloc[:, 2].values

# Model Creation and Fit
rf_model = RandomForestRegressor(n_estimators=10, random_state=0)

rf_model.fit(data_x, data_y)

# Prediction on unknown value
prediction_y = rf_model.predict([[6.5]])

# Plotting Results
grid_x = np.arange(min(data_x), max(data_x), 0.01)  # Choice of 0.01 as data is scaled
grid_x = grid_x.reshape((len(grid_x), 1))
plt.scatter(data_x, data_y, color="red")
plt.plot(grid_x, rf_model.predict(grid_x), color="blue")
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
