
from model_training import get_prepared_data, linear_regression, decision_tree, random_forest_reg
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# LINEAR REGRESSION
def lin_reg_metric(model, X, y):
    y_predict = model.predict(X)

    mse = mean_squared_error(y, y_predict)
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")

    lin_scores = cross_val_score(model, X, y,
                                 scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    print("\nCross Validation scores:")
    display_scores(lin_rmse_scores)

# DECISION TREE REGRESSOR
def tree_metric(model, X, y):
    y_predict = model.predict(X)

    tree_mse = mean_squared_error(y, y_predict)
    tree_rmse = np.sqrt(tree_mse)
    print(f"Mean Squared Error: {tree_mse}")
    print(f"Root Mean Squared Error: {tree_rmse}")

    tree_scores = cross_val_score(model, X, y,
                                  scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-tree_scores)
    display_scores(tree_rmse_scores)

# RANDOM FOREST REGRESSOR
def forest_metric(model, X, y):
    y_predict = model.predict(X)

    forest_mse = mean_squared_error(y, y_predict)
    forest_rmse = np.sqrt(forest_mse)
    print(f"Mean Squared Error: {forest_mse}")
    print(f"Root Mean Squared Error: {forest_rmse}")

    forest_scores = cross_val_score(model, X, y,
                                    scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    display_scores(forest_rmse_scores)



X_prepared, y = get_prepared_data()

# Linear Regression
print("\nLINEAR REGRESSION EVALUATION")
trained_lin_reg = linear_regression()
lin_reg_metric(trained_lin_reg, X_prepared, y)

# Decision Tree
print("\nDECISION TREE EVALUATION")
trained_tree_reg = decision_tree()
tree_metric(trained_tree_reg, X_prepared, y)

# Random Forest
print("\nRANDOM FOREST EVALUATION")
trained_forest_reg = random_forest_reg()
forest_metric(trained_forest_reg, X_prepared, y)