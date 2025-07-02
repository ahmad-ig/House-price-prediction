from data_preprocessing import get_prepared_data
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


X_prepared, y = get_prepared_data()

def linear_regression():
    lin_reg = LinearRegression()
    lin_reg.fit(X_prepared, y)
    return lin_reg

def decision_tree():
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(X_prepared, y)
    return tree_reg

def random_forest_reg():
    forest_reg = RandomForestRegressor()
    forest_reg.fit(X_prepared, y)
    return forest_reg


