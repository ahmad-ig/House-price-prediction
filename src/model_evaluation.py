import numpy as np
import pandas as pd
from data_preprocessing import full_pipeline
from fine_tune_best_model import grid_search
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import joblib
import cloudpickle
import os

# Load the test set
X_test = pd.read_csv("data/housing_strat_test_set.csv")
y_test = X_test["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_model = grid_search.best_estimator_

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("\nFinal Model Test Set MSE:", final_mse)
print("\nFinal Model Test Set RMSE:", final_rmse)
print("\nFinal Model Test Set MSE:", final_mse)
print("\nFinal Model Test Set Predictions:", final_predictions[:10])
print("\nFinal Model Test Set Actual Values:", y_test.values[:10])


# Save the final model and pipeline
# Create a full end-to-end pipeline: preprocessing + model
#final_pipeline = Pipeline([
#    ("preprocessing", full_pipeline),
#    ("model", final_model)  # your best trained RandomForest model
#])

# Save the final pipeline to together
#joblib.dump(final_pipeline, 'random_forest_model.pkl')
#print("\nMODEL SAVED AS random_forest_model.pkl")

# save model and pipeline separately
#joblib.dump(final_model, "models/rf_model.pkl")
#joblib.dump(full_pipeline, "models/preprocessor.pkl")

with open("models/rf_model_v1.pkl", "wb") as f:
    cloudpickle.dump(final_model, f)

#with open("models/preprocessor_v1.pkl", "wb") as f:
#    cloudpickle.dump(full_pipeline, f)

print("\nMODEL SAVED AS rf_model.pkl")

# Create folder if it doesn't exist
#save_folder = "models"
#os.makedirs(save_folder, exist_ok=True)

# Save to path
#save_path = os.path.join(save_folder, "random_forest_model.pkl")
#joblib.dump(final_pipeline, save_path)

#print(f" Model pipeline saved to {save_path}")