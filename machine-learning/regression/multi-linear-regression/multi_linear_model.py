# -*- coding: utf-8 -*-

"""
Multiple Linear Regression Model
"""

# Base Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from statsmodels.regression.linear_model import OLS

# Importing the Data
dataset = pd.read_csv("../data/50_Startups.csv")
data_x = dataset.iloc[:, :-1].values
data_y = dataset.iloc[:, 4].values


# Data Preprocessing
label_encoder_x = LabelEncoder()
data_x[:, 3] = label_encoder_x.fit_transform(data_x[:, 3])

column_transformer = ColumnTransformer(
    transformers=[("3", OneHotEncoder(), [3])], remainder="passthrough"
)

data_x = column_transformer.fit_transform(data_x)

# Avoiding the dummy variable trap
data_x = data_x[:, 1:]

# Training and Testing Data Set Split 80:20
trainset_data_x, testset_data_x, trainset_data_y, testset_data_y = train_test_split(
    data_x, data_y, test_size=0.2, random_state=0
)

# Model Creation and Fitting
multi_linear_model = LinearRegression()
multi_linear_model.fit(trainset_data_x, trainset_data_y)

# Predictions
prediction_y = multi_linear_model.predict(testset_data_x)

# Introduction to Backward Elimination
data_x = np.append(values=data_x, arr=np.ones((50, 1)).astype(int), axis=1)
# First Case
optimal_x = np.array(data_x[:, [0, 1, 2, 3, 4, 5]], dtype=float)

ols_model = OLS(endog=data_y, exog=optimal_x).fit()
ols_model.summary()

# Second Case
optimal_x = np.array(data_x[:, [0, 1, 3, 4, 5]], dtype=float)
ols_model = OLS(endog=data_y, exog=optimal_x).fit()
ols_model.summary()

# Third Case
optimal_x = np.array(data_x[:, [0, 3, 4, 5]], dtype=float)
ols_model = OLS(endog=data_y, exog=optimal_x).fit()
ols_model.summary()

# Fourth Case
optimal_x = np.array(data_x[:, [0, 3, 5]], dtype=float)
ols_model = OLS(endog=data_y, exog=optimal_x).fit()
ols_model.summary()

# Fifth Case
optimal_x = np.array(data_x[:, [0, 3]], dtype=float)
ols_model = OLS(endog=data_y, exog=optimal_x).fit()
ols_model.summary()
