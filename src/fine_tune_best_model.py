from data_preprocessing import get_prepared_data
from model_training import random_forest_reg
from error_metrics import forest_metric
from sklearn.model_selection import GridSearchCV
import numpy as np

X_prepared, y = get_prepared_data()
forest_reg = random_forest_reg()
# 
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]
# search for the best hyperparameters
grid_search = GridSearchCV(
    forest_reg, param_grid, cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

grid_search.fit(X_prepared, y)
print("Best parameters found:", grid_search.best_params_)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#print("Final Model Test Set Best Parameters:", grid_search.best_params_)
#print("Final Model Test Set Best Estimator:", grid_search.best_estimator_)
#print("Final Model Test Set Best Score:", grid_search.best_score_)
#print("Final Model Test Set CV Results:", grid_search.cv_results_)
#print("Final Model Test Set Feature Importances:", grid_search.best_estimator_.feature_importances_)
#print("Final Model Test Set Feature Names:", X_test_prepared.columns.tolist())