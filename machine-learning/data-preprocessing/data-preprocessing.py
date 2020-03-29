# -*- coding: utf-8 -*-

# Import Libraries
import numpy as np

# import matplotlib.pylot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Import the dataset
data = pd.read_csv("data.csv")
data_x = data.iloc[:, :-1].values
data_y = data.iloc[:, 3].values

# Handling Missing Data
""" Basic Imputation Strategy replacing the NaN values by Mean of the same col.
"""
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

imputer.fit(data_x[:, 1:3])

data_x[:, 1:3] = imputer.transform(data_x[:, 1:3])

# Handling Categorical Data
label_encoder_data_x = LabelEncoder()
data_x[:, 0] = label_encoder_data_x.fit_transform(data_x[:, 0])

label_encoder_data_y = LabelEncoder()
data_y = label_encoder_data_y.fit_transform(data_y)


column_transformer = ColumnTransformer(
    transformers=[("0", OneHotEncoder(), [0])], remainder="passthrough"
)

data_x = column_transformer.fit_transform(data_x)


# Building the training and testing sets with 80-20 split
trainset_data_x, testset_data_x, trainset_data_y, testset_data_y = train_test_split(
    data_x, data_y, test_size=0.2, random_state=0
)

# Scaling Features
standard_scalar_x = StandardScaler()

trainset_data_x = standard_scalar_x.fit_transform(trainset_data_x)

testset_data_x = standard_scalar_x.transform(testset_data_x)

