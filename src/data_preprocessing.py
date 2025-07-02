import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import cloudpickle

# Load the training dataset
housing_train = pd.read_csv("data/housing_strat_train_set.csv")

# Separate target from features
X = housing_train.drop("median_house_value", axis=1)
y = housing_train["median_house_value"]


# Separate numerical and categorical features
X_num = X.select_dtypes(include=[np.number])
X_cat = X.select_dtypes(exclude=[np.number])

# Save feature names
X_num_features = list(X_num.columns)
X_cat_features = list(X_cat.columns)

# Index of the numerical features based on the X_num columns
rooms_ix = X_num_features.index("total_rooms")
bedrooms_ix = X_num_features.index("total_bedrooms")
population_ix = X_num_features.index("population")
households_ix = X_num_features.index("households")

# FEATURE ENGINEERING
class AddFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        # This controls whether to include the 'bedrroms per room' feature
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        # No training is needed, just return self
        return self
    
    def transform(self, X):
        total_rooms = X[:, rooms_ix]
        total_bedrooms = X[:, bedrooms_ix]
        population = X[:, population_ix]
        households = X[:, households_ix]

        # Create new features
        rooms_per_household = total_rooms / households
        population_per_household = population / households

        # If bedrooms per room is requested, calculate it
        if self.add_bedrooms_per_room:
            bedrooms_per_room = total_bedrooms / total_rooms
            # Combine original data + new columns
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            # Combine original data + new columns without bedrooms per room
            return np.c_[X, rooms_per_household, population_per_household]
      
      
# NUMERICAL PIPELINE
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("feature_engineering", AddFeatures()),
    ("scaler", StandardScaler())
])

# FULL PIPELINE
# Combine numerical and categorical features
full_pipeline = ColumnTransformer([
    ("num", numerical_pipeline, X_num_features),
    ("cat", OneHotEncoder(), X_cat_features)
])

# FIT & TRANSFORM
# Fit the full pipeline to the training data and transform it
X_prepared = full_pipeline.fit_transform(X)

# Export X_prepared, y for model training
#__all__ = ["X_prepared", "y"]
def get_prepared_data():
    return X_prepared, y

# Create models folder if not exist
os.makedirs("models", exist_ok=True)

with open("models/preprocessor_v1.pkl", "wb") as f:
    cloudpickle.dump(full_pipeline, f)

print("Preprocessing pipeline saved to models/preprocessor.pkl")

# Save the transformed data to a CSV file
#transformed_housing_train = pd.DataFrame(X_prepared, columns=full_pipeline.get_feature_names_out())
#transformed_housing_train["median_house_value"] = y.values
#transformed_housing_train.to_csv('data/housing_transformed_train_set.csv', index=False)
# Save the feature names for later use
