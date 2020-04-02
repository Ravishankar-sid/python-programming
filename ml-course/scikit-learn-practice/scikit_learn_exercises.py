# Import the pyplot module of Matplotlib as plt
import matplotlib.pyplot as plt

# Import pandas under the abbreviation 'pd'
import pandas as pd

# Import NumPy under the abbreviation 'np'
import numpy as np

# Import Seaborn
import seaborn as sns

# Import Scikit Learn Modules
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    plot_roc_curve,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from joblib import dump, load
from sklearn.pipeline import Pipeline

"""End to End Scikit Learn Classification Pipeline

    * Get a dataset ready
    * Prepare a machine learning model to make predictions
    * Fit the model to the data and make a prediction
    * Evaluate the model's predictions
"""

# Import Data
dataset = pd.read_csv("../data/heart-disease.csv")

# Create X (all columns except target)
X = dataset.drop("target", axis=1)

# Create y (only the target column)
y = dataset["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# View the different shapes of the training and test datasets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Instantiate an instance of RandomForestClassifier as clf
clf = RandomForestClassifier()

# Fit the RandomForestClassifier to the training data
clf.fit(X_train, y_train)

# Use the fitted model to make predictions on the test data and
# save the predictions to a variable called y_preds
y_preds = clf.predict(X_test)

# Evaluate the fitted model on the training set using the score() function
clf.score(X_train, y_train)

# Evaluate the fitted model on the test set using the score() function
clf.score(X_test, y_test)


# Create a dictionary called models which contains all of the classification models we've imported
# Make sure the dictionary is in the same format as example_dict
# The models dictionary should contain 5 models
models = {
    "LinearSVC": LinearSVC(),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(),
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(),
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    results[model_name] = model.score(X_test, y_test)

# Run the same code as above, except this time set a NumPy random seed
# equal to 42

np.random.seed(42)
for model_name, model in models.items():
    model.fit(X_train, y_train)
    results[model_name] = model.score(X_test, y_test)

print(results)

# Create a pandas dataframe with the data as the values of the results dictionary,
# the index as the keys of the results dictionary and a single column called accuracy.
# Be sure to save the dataframe to a variable.
results_df = pd.DataFrame(results.values(), results.keys(), columns=["Accuracy"])

# Create a bar plot of the results dataframe using plot.bar()
results_df.plot.bar()

# Different LogisticRegression hyperparameters
log_reg_grid = {"C": np.logspace(-4, 4, 20), "solver": ["liblinear"]}

# Setup np random seed of 42
np.random.seed(42)

# Setup an instance of RandomizedSearchCV with a LogisticRegression() estimator,
# our log_reg_grid as the param_distributions, a cv of 5 and n_iter of 5.

rs_log_reg = RandomizedSearchCV(
    estimator=LogisticRegression(),
    cv=5,
    n_iter=5,
    param_distributions=log_reg_grid,
    verbose=5,
)

rs_log_reg.fit(X_train, y_train)

# Find the best parameters of the RandomizedSearchCV instance using the best_params_ attribute
rs_log_reg.best_params_

# Score the instance of RandomizedSearchCV using the test data
rs_log_reg.score(X_test, y_test)

"""
Classifier Model Evaluation
    * Confusion matrix - Compares the predicted values with the true values in a
        tabular way, if 100% correct, all values in the matrix will be top left to bottom right (diagnol line).

    * Cross-validation - Splits your dataset into multiple parts and train and
        tests your model on each part and evaluates performance as an average.

    * Precision - Proportion of true positives over total number of samples.
        Higher precision leads to less false positives.

    * Recall - Proportion of true positives over total number of true positives
        and false positives. Higher recall leads to less false negatives.

    * F1 score - Combines precision and recall into one metric. 1 is best, 0 is worst.

    * Classification report - Sklearn has a built-in function called
        classification_report() which returns some of the main classification
        metrics such as precision, recall and f1-score.

    * ROC Curve - Receiver Operating Characterisitc is a plot of true positive
        rate versus false positive rate.

    * Area Under Curve (AUC) - The area underneath the ROC curve. A perfect model
        achieves a score of 1.0.
"""

# Instantiate a LogisticRegression classifier using the best hyperparameters
# from RandomizedSearchCV

clf = LogisticRegression(C=206.913808111479, solver="liblinear")

# Fit the new instance of LogisticRegression with the best hyperparameters on
# the training data
clf.fit(X_train, y_train)

# Make predictions on test data and save them
y_preds = clf.predict(X_test)

# Create a confusion matrix using the confusion_matrix function
confusion_matrix(y_test, y_preds)

# Plotting the Confusion Matrix


def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(
        confusion_matrix(y_test, y_preds), annot=True, cbar=False  # Annotate the boxes
    )
    plt.xlabel("true label")
    plt.ylabel("predicted label")

    # Fix the broken annotations (this happened in Matplotlib 3.1.1)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)


plot_conf_mat(y_test, y_preds)

# Classification Report
print(classification_report(y_test, y_preds))

"""
Classification Report Metrics
-----------------------------

* Precision - Indicates the proportion of positive identifications
    (model predicted class 1) which were actually correct. A model which produces no
    false positives has a precision of 1.0.

* Recall - Indicates the proportion of actual positives which were correctly classified.
    A model which produces no false negatives has a recall of 1.0.

* F1 score - A combination of precision and recall. A perfect model achieves an F1
    score of 1.0.

* Support - The number of samples each metric was calculated on.

* Accuracy - The accuracy of the model in decimal form. Perfect accuracy is equal
    to 1.0.

* Macro avg - Short for macro average, the average precision, recall and F1 score
    between classes. Macro avg doesn’t class imbalance into effort, so if you do have
    class imbalances, pay attention to this metric.

* Weighted avg - Short for weighted average, the weighted average precision,
    recall and F1 score between classes. Weighted means each metric is calculated
    with respect to how many samples there are in each class. This metric will favour
    the majority class (e.g. will give a high value when one class out performs another
    due to having more samples).
"""

# Find the precision score of the model using precision_score()
print(precision_score(y_test, y_preds))

# Find the F1 score
print(f1_score(y_test, y_preds))

# Find the recall score
print(recall_score(y_test, y_preds))

# Plot a ROC curve using our current machine learning model using plot_roc_curve
plot_roc_curve(clf, X_test, y_test)


# By default cross_val_score returns 5 values (cv=5).
cross_val_score(clf, X, y, scoring="accuracy", cv=5)

# Taking the mean of the returned values from cross_val_score gives a
# cross-validated version of the scoring metric.

cross_val_acc = np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=5))

print(cross_val_acc)

# Find the cross-validated precision
cross_val_precision = np.mean(cross_val_score(clf, X, y, scoring="precision", cv=5))
print(cross_val_precision)

# Find the cross-validated recall
cross_val_recall = np.mean(cross_val_score(clf, X, y, scoring="recall", cv=5))
print(cross_val_recall)

# Find the cross-validated f-1 score
cross_val_f1_score = np.mean(cross_val_score(clf, X, y, scoring="f1", cv=5))
print(cross_val_f1_score)

# Use the dump function to export the trained model to file
dump(clf, "trained-classifier.joblib")

# Use the load function to import the trained model you just exported
# Save it to a different variable name to the origial trained model
loaded_clf = load("trained-classifier.joblib")

# Evaluate the loaded trained model on the test data
loaded_clf.score(X_test, y_test)

# Scikit-Learn Regression Practice --------------------------------------------
car_sales_data = pd.read_csv("../data/car-sales-extended-missing-data.csv")

# View the first 5 rows of the car sales data
car_sales_data.head()

# Get information about the car sales DataFrame
car_sales_data.info()

# Find number of missing values in each column
car_sales_data.isna().sum()

# Find the datatypes of each column of car_sales
car_sales_data.dtypes

# Remove rows with no labels (NaN's in the Price column)
car_sales_data.dropna(subset=["Price"], inplace=True)

# Define different categorical features
categorical_features = ["Make", "Colour"]

# Create categorical transformer Pipeline
categorical_transformer = Pipeline(
    steps=[
        # Set SimpleImputer strategy to "constant" and fill value to "missing"
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        # Set OneHotEncoder to ignore the unknowns
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
# Define Doors features
door_feature = ["Doors"]
# Create Doors transformer Pipeline
door_transformer = Pipeline(
    steps=[
        # Set SimpleImputer strategy to "constant" and fill value to 4
        ("imputer", SimpleImputer(strategy="constant", fill_value=4))
    ]
)

# Define numeric features (only the Odometer (KM) column)
numeric_feature = ["Odometer (KM)"]

numeric_transformer = Pipeline(
    steps=[
        # Set SimpleImputer strategy to "constant" and fill value to 4
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Setup preprocessing steps (fill missing values, then convert to numbers)
preprocessor = ColumnTransformer(
    transformers=[
        ("categ", categorical_transformer, categorical_features),
        ("doors", door_transformer, door_feature),
        ("num", numeric_transformer, numeric_feature),
    ]
)
# Create dictionary of model instances, there should be 4 total key, value pairs
# in the form {"model_name": model_instance}.
# Don't forget there's two versions of SVR, one with a "linear" kernel and the
# other with kernel set to "rbf".
regression_models = {
    "ridge": Ridge(),
    "svr_linear": SVR(kernel="linear"),
    "svr_rbf": SVR(kernel="rbf"),
    "random_forest_regressor": RandomForestRegressor(),
}

# Create an empty dictionary for the regression results
regression_results = {}

# Create car sales X data (every column of car_sales except Price)
X = car_sales_data.drop("Price", axis=1)
# Create car sales y data (the Price column of car_sales)
y = car_sales_data["Price"]

# Use train_test_split to split the car_sales_X and car_sales_y data into
# training and test sets.
# Give the test set 20% of the data using the test_size parameter.
# For reproducibility set the random_state parameter to 42.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Check the shapes of the training and test datasets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

for model_name, model in regression_models.items():
    model_pipeline = Pipeline(steps=[("preprocesser", preprocessor), ("model", model)])

    print(f"Fitting {model_name}...")
    model_pipeline.fit(X_train, y_train)

    print(f"Scoring {model_name}...")
    regression_results[model_name] = model_pipeline.score(X_test, y_test)


"""
Regression Metrics
--------------------

* R^2 (pronounced r-squared) or coefficient of determination - Compares your models
    predictions to the mean of the targets. Values can range from negative infinity
    (a very poor model) to 1. For example, if all your model does is predict the mean
    of the targets, its R^2 value would be 0. And if your model perfectly predicts a
    range of numbers it's R^2 value would be 1.

* Mean absolute error (MAE) - The average of the absolute differences between
    predictions and actual values. It gives you an idea of how wrong your
    predictions were.

* Mean squared error (MSE) - The average squared differences between predictions
    and actual values. Squaring the errors removes negative errors. It also
    amplifies outliers (samples which have larger errors).
"""

# Create RidgeRegression Pipeline with preprocessor as the "preprocessor" and
# Ridge() as the "model".

ridge_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", Ridge())])
# Fit the RidgeRegression Pipeline to the car sales training data
ridge_pipeline.fit(X_train, y_train)

# Make predictions on the car sales test data using the RidgeRegression Pipeline
y_preds = ridge_pipeline.predict(X_test)

# View the first 50 predictions
y_preds[:50]

# Find the MSE by comparing the car sales test labels to the car sales predictions
mse = mean_squared_error(y_test, y_preds)
print(mse)

# Find the MAE by comparing the car sales test labels to the car sales predictions
mae = mean_absolute_error(y_test, y_preds)
print(mae)

# Find the R^2 score by comparing the car sales test labels to
# the car sales predictions
r2 = r2_score(y_test, y_preds)
print(r2)
